use std::array;
use std::sync::Arc;

use allocative::Allocative;
use ark_std::Zero;

use crate::field::JoltField;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::multilinear_polynomial::PolynomialBinding;
use crate::poly::multilinear_polynomial::{BindingOrder, MultilinearPolynomial};
use crate::poly::opening_proof::{
    OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
    VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
};
use crate::poly::unipoly::UniPoly;
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use crate::subprotocols::sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier};
use crate::transcripts::Transcript;
use crate::utils::math::Math;
use crate::utils::thread::unsafe_allocate_zero_vec;
use crate::zkvm::witness::VirtualPolynomial;
use rayon::prelude::*;
use tracer::instruction::Cycle;

/// Degree bound of the sumcheck round polynomials in [`RegistersClaimReductionSumcheckVerifier`].
const DEGREE_BOUND: usize = 2;

#[derive(Allocative, Clone)]
pub struct RegistersClaimReductionSumcheckParams<F: JoltField> {
    pub gamma: F,
    pub gamma_sqr: F,
    pub n_cycle_vars: usize,
    pub r_spartan: OpeningPoint<BIG_ENDIAN, F>,
}

impl<F: JoltField> RegistersClaimReductionSumcheckParams<F> {
    pub fn new(
        trace_len: usize,
        accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let gamma = transcript.challenge_scalar::<F>();
        let gamma_sqr = gamma.square();
        let (r_spartan, _) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::LookupOutput,
            SumcheckId::SpartanOuter,
        );
        Self {
            gamma,
            gamma_sqr,
            n_cycle_vars: trace_len.log_2(),
            r_spartan,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for RegistersClaimReductionSumcheckParams<F> {
    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let (_, rd_write_value_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RdWriteValue,
            SumcheckId::SpartanOuter,
        );
        let (_, rs1_value_claim) = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::Rs1Value, SumcheckId::SpartanOuter);
        let (_, rs2_value_claim) = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::Rs2Value, SumcheckId::SpartanOuter);
        rd_write_value_claim + self.gamma * rs1_value_claim + self.gamma_sqr * rs2_value_claim
    }

    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.n_cycle_vars
    }

    fn normalize_opening_point(
        &self,
        challenges: &[<F as JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }
}

/// Sumcheck prover for [`RegistersClaimReductionSumcheckVerifier`].
#[derive(Allocative)]
pub struct RegistersClaimReductionSumcheckProver<F: JoltField> {
    phase: RegistersClaimReductionPhase<F>,
    pub params: RegistersClaimReductionSumcheckParams<F>,
}

#[derive(Allocative)]
#[allow(clippy::large_enum_variant)]
enum RegistersClaimReductionPhase<F: JoltField> {
    Phase1(RegistersPhase1State<F>), // 1st half of sumcheck rounds (prefix-suffix sumcheck)
    Phase2(RegistersPhase2State<F>), // 2nd half of sumcheck rounds (regular sumcheck)
}

impl<F: JoltField> RegistersClaimReductionSumcheckProver<F> {
    #[tracing::instrument(skip_all, name = "RegistersClaimReductionSumcheckProver::initialize")]
    pub fn initialize(
        params: RegistersClaimReductionSumcheckParams<F>,
        trace: Arc<Vec<Cycle>>,
    ) -> Self {
        let phase =
            RegistersClaimReductionPhase::Phase1(RegistersPhase1State::initialize(trace, &params));
        Self { phase, params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T>
    for RegistersClaimReductionSumcheckProver<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(
        skip_all,
        name = "RegistersClaimReductionSumcheckProver::compute_message"
    )]
    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        match &self.phase {
            RegistersClaimReductionPhase::Phase1(state) => {
                state.compute_message(&self.params, previous_claim)
            }
            RegistersClaimReductionPhase::Phase2(state) => {
                state.compute_message(&self.params, previous_claim)
            }
        }
    }

    #[tracing::instrument(
        skip_all,
        name = "RegistersClaimReductionSumcheckProver::ingest_challenge"
    )]
    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        match &mut self.phase {
            RegistersClaimReductionPhase::Phase1(state) => {
                if state.should_transition_to_phase2() {
                    let mut sumcheck_challenges = state.sumcheck_challenges.clone();
                    sumcheck_challenges.push(r_j);
                    self.phase = RegistersClaimReductionPhase::Phase2(RegistersPhase2State::gen(
                        &state.trace,
                        &sumcheck_challenges,
                        &self.params,
                    ));
                    return;
                }
                state.bind(r_j);
            }
            RegistersClaimReductionPhase::Phase2(state) => state.bind(r_j),
        }
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let RegistersClaimReductionPhase::Phase2(state) = &self.phase else {
            panic!("Should finish sumcheck on phase 2");
        };

        let opening_point = SumcheckInstanceProver::<F, T>::get_params(self)
            .normalize_opening_point(sumcheck_challenges);

        let rd_write_value_claim = state.rd_write_value_poly.final_sumcheck_claim();
        let rs1_read_value_claim = state.rs1_read_value_poly.final_sumcheck_claim();
        let rs2_read_value_claim = state.rs2_read_value_poly.final_sumcheck_claim();

        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RdWriteValue,
            SumcheckId::RegistersClaimReduction,
            opening_point.clone(),
            rd_write_value_claim,
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::Rs1Value,
            SumcheckId::RegistersClaimReduction,
            opening_point.clone(),
            rs1_read_value_claim,
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::Rs2Value,
            SumcheckId::RegistersClaimReduction,
            opening_point,
            rs2_read_value_claim,
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

#[derive(Allocative)]
struct RegistersPhase1State<F: JoltField> {
    // Prefix-suffix P and Q buffers.
    // See <https://eprint.iacr.org/2025/611.pdf> (Appendix A).
    P: MultilinearPolynomial<F>,
    Q: MultilinearPolynomial<F>,
    #[allocative(skip)]
    trace: Arc<Vec<Cycle>>,
    sumcheck_challenges: Vec<F::Challenge>,
}

impl<F: JoltField> RegistersPhase1State<F> {
    ///Sum(Combined(x) * Eq(x, r)),Combined(x) = Val_rd(x) + gamma * Val_rs1(x) + gamma^2 * Val_rs2(x)
    fn initialize(
        trace: Arc<Vec<Cycle>>,
        params: &RegistersClaimReductionSumcheckParams<F>,
    ) -> Self {
        // ========================================================================
        // 1. 准备 Split Sumcheck 组件 (Prepare Split Sumcheck)
        // ========================================================================
        // [算法原理: Split Sumcheck / Tensor Product]
        // 目标：证明 Sum(Trace(x) * Eq(x, r))。
        // 方法：将随机点 r 拆分为高位 r_hi 和低位 r_lo。
        // Eq(x, r) = Eq(x_hi, r_hi) * Eq(x_lo, r_lo)。

        // 将随机点切分为两半
        let (r_hi, r_lo) = params.r_spartan.split_at(params.r_spartan.len() / 2);

        // 计算 P 向量 (Prefix): 对应 Eq(x_lo, r_lo)
        // 这是一个纯数学构造，只与随机点 r_lo 有关，不依赖 Trace 数据。
        let eq_prefix_evals = EqPolynomial::evals(&r_lo.r);

        // 计算 Suffix 向量: 对应 Eq(x_hi, r_hi)
        // 这将作为权重，用于将 Trace 矩阵沿着高位维度“折叠”。
        let eq_suffix_evals = EqPolynomial::evals(&r_hi.r);

        let prefix_n_vars = r_lo.len();
        let suffix_n_vars = r_hi.len();

        // ========================================================================
        // 2. 初始化 P 和 Q 缓冲区 (Initialize P & Q)
        // ========================================================================
        // Prefix-suffix P and Q buffers.
        // See <https://eprint.iacr.org/2025/611.pdf> (Appendix A).
        // P: 直接就是 prefix 的评估值。
        let P = eq_prefix_evals;

        // Q: Suffix Sum 向量。
        // Q 的大小仅为 2^prefix_len (例如 Trace 是 2^20，Q 可能只有 2^10)。
        // 我们需要遍历整个 Trace 来填充这个 Q。
        let mut Q = unsafe_allocate_zero_vec(1 << prefix_n_vars);

        // [算法原理: Random Linear Combination (RLC)]
        // 我们需要同时提取三个不同的值：rd_write, rs1_read, rs2_read。
        // 为了不跑三次 Sumcheck，我们用随机数 gamma 将它们打包：
        // Combined(x) = Val_rd(x) + gamma * Val_rs1(x) + gamma^2 * Val_rs2(x)
        let gamma = params.gamma;
        let gamma_sqr = params.gamma_sqr;

        // ========================================================================
        // 3. 并行计算 Q 向量 (Compute Q in Parallel)
        // ========================================================================
        const BLOCK_SIZE: usize = 32;
        Q.par_chunks_mut(BLOCK_SIZE)
            .enumerate()
            .for_each(|(chunk_i, q_chunk)| {
                // [算法优化: Delayed Reduction]
                // 为了性能，中间累加过程不进行取模（Reduce）。
                // F::Unreduced 类型允许累加多次乘积结果，利用 CPU 的大寄存器避免频繁取模。
                let mut q_rd_write_value = [F::Unreduced::<6>::zero(); BLOCK_SIZE];
                let mut q_rs1_read_value = [F::Unreduced::<6>::zero(); BLOCK_SIZE];
                let mut q_rs2_read_value = [F::Unreduced::<6>::zero(); BLOCK_SIZE];

                // [核心循环: Marginalization / Folding]
                // 遍历所有的高位组合 (Suffix)。
                // 我们把整个 Trace 看作一个 [Rows x Cols] 的矩阵。
                // Rows 是 x_hi (Suffix), Cols 是 x_lo (Prefix)。
                // 我们要把每一列的所有行加权求和，压缩成一行 (Q)。
                for x_hi in 0..(1 << suffix_n_vars) {
                    for i in 0..q_chunk.len() {
                        // 构造完整的 Trace 索引 x
                        let x_lo = chunk_i * BLOCK_SIZE + i;
                        let x = x_lo + (x_hi << prefix_n_vars);

                        let cycle = &trace[x];

                        // 提取每一行的寄存器操作数数值
                        // .2 代表 rd 的写入值
                        let rd_write_value = cycle.rd_write().unwrap_or_default().2;
                        // .1 代表 rs1 的读取值
                        let rs1_read_value = cycle.rs1_read().unwrap_or_default().1;
                        // .1 代表 rs2 的读取值
                        let rs2_read_value = cycle.rs2_read().unwrap_or_default().1;

                        // 累加到临时缓冲区
                        // Q_temp[i] += Trace[x] * Suffix_Eq[x_hi]
                        q_rd_write_value[i] +=
                            eq_suffix_evals[x_hi].mul_u64_unreduced(rd_write_value);
                        q_rs1_read_value[i] +=
                            eq_suffix_evals[x_hi].mul_u64_unreduced(rs1_read_value);
                        q_rs2_read_value[i] +=
                            eq_suffix_evals[x_hi].mul_u64_unreduced(rs2_read_value);
                    }
                }

                // [约束应用与合并]
                // 将三个独立的累加结果，通过 gamma 进行线性组合，并进行最终的取模。
                for (i, q) in q_chunk.iter_mut().enumerate() {
                    *q = F::from_barrett_reduce(q_rd_write_value[i])
                        + gamma * F::from_barrett_reduce(q_rs1_read_value[i])
                        + gamma_sqr * F::from_barrett_reduce(q_rs2_read_value[i]);
                }
            });

        Self {
            P: P.into(),
            Q: Q.into(),
            trace,
            sumcheck_challenges: Vec::new(),
        }
    }

    fn compute_message(
        &self,
        _params: &RegistersClaimReductionSumcheckParams<F>,
        previous_claim: F,
    ) -> UniPoly<F> {
        let Self { P, Q, .. } = self;
        let mut evals = [F::zero(); DEGREE_BOUND];

        for j in 0..P.len() / 2 {
            let p_evals = P.sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
            let q_evals = Q.sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
            evals = array::from_fn(|i| evals[i] + p_evals[i] * q_evals[i]);
        }

        UniPoly::from_evals_and_hint(previous_claim, &evals)
    }

    fn bind(&mut self, r_j: F::Challenge) {
        assert!(!self.should_transition_to_phase2());
        self.sumcheck_challenges.push(r_j);
        self.P.bind(r_j, BindingOrder::LowToHigh);
        self.Q.bind(r_j, BindingOrder::LowToHigh);
    }

    fn should_transition_to_phase2(&self) -> bool {
        self.P.len().log_2() == 1
    }
}

#[derive(Allocative)]
struct RegistersPhase2State<F: JoltField> {
    rd_write_value_poly: MultilinearPolynomial<F>,
    rs1_read_value_poly: MultilinearPolynomial<F>,
    rs2_read_value_poly: MultilinearPolynomial<F>,
    eq_poly: MultilinearPolynomial<F>,
}

impl<F: JoltField> RegistersPhase2State<F> {
    fn gen(
        trace: &[Cycle],
        sumcheck_challenges: &[F::Challenge],
        params: &RegistersClaimReductionSumcheckParams<F>,
    ) -> Self {
        let n_remaining_rounds = params.r_spartan.len() - sumcheck_challenges.len();
        let r_prefix: OpeningPoint<BIG_ENDIAN, F> =
            OpeningPoint::<LITTLE_ENDIAN, F>::new(sumcheck_challenges.to_vec()).match_endianness();

        let eq_evals = EqPolynomial::evals(&r_prefix.r);
        let mut rd_write_value_poly = unsafe_allocate_zero_vec(1 << n_remaining_rounds);
        let mut rs1_read_value_poly = unsafe_allocate_zero_vec(1 << n_remaining_rounds);
        let mut rs2_read_value_poly = unsafe_allocate_zero_vec(1 << n_remaining_rounds);
        (
            &mut rd_write_value_poly,
            &mut rs1_read_value_poly,
            &mut rs2_read_value_poly,
            trace.par_chunks(eq_evals.len()),
        )
            .into_par_iter()
            .for_each(
                |(rd_write_value_eval, rs1_read_value_eval, rs2_read_value_eval, trace_chunk)| {
                    let mut rd_write_value_eval_unreduced = F::Unreduced::<6>::zero();
                    let mut rs1_read_value_eval_unreduced = F::Unreduced::<6>::zero();
                    let mut rs2_read_value_eval_unreduced = F::Unreduced::<6>::zero();

                    for (i, cycle) in trace_chunk.iter().enumerate() {
                        let rd_write_value = cycle.rd_write().unwrap_or_default().2;
                        let rs1_value_eval = cycle.rs1_read().unwrap_or_default().1;
                        let rs2_value_eval = cycle.rs2_read().unwrap_or_default().1;
                        rd_write_value_eval_unreduced +=
                            eq_evals[i].mul_u64_unreduced(rd_write_value);
                        rs1_read_value_eval_unreduced +=
                            eq_evals[i].mul_u64_unreduced(rs1_value_eval);
                        rs2_read_value_eval_unreduced +=
                            eq_evals[i].mul_u64_unreduced(rs2_value_eval);
                    }

                    *rd_write_value_eval = F::from_barrett_reduce(rd_write_value_eval_unreduced);
                    *rs1_read_value_eval = F::from_barrett_reduce(rs1_read_value_eval_unreduced);
                    *rs2_read_value_eval = F::from_barrett_reduce(rs2_read_value_eval_unreduced);
                },
            );

        let (r_hi, r_lo) = params.r_spartan.split_at(params.r_spartan.len() / 2);
        let eq_prefix_eval = EqPolynomial::mle_endian(&r_prefix, &r_lo);
        let eq_suffix_evals = EqPolynomial::evals_parallel(&r_hi.r, Some(eq_prefix_eval));

        Self {
            rd_write_value_poly: rd_write_value_poly.into(),
            rs1_read_value_poly: rs1_read_value_poly.into(),
            rs2_read_value_poly: rs2_read_value_poly.into(),
            eq_poly: eq_suffix_evals.into(),
        }
    }

    fn compute_message(
        &self,
        params: &RegistersClaimReductionSumcheckParams<F>,
        previous_claim: F,
    ) -> UniPoly<F> {
        let half_n = self.rd_write_value_poly.len() / 2;
        let mut evals = [F::zero(); DEGREE_BOUND];
        for j in 0..half_n {
            let rd_write_value_evals = self
                .rd_write_value_poly
                .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
            let rs1_read_value_evals = self
                .rs1_read_value_poly
                .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
            let rs2_read_value_evals = self
                .rs2_read_value_poly
                .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
            let eq_evals = self
                .eq_poly
                .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
            evals = array::from_fn(|i| {
                evals[i]
                    + eq_evals[i]
                        * (rd_write_value_evals[i]
                            + params.gamma * rs1_read_value_evals[i]
                            + params.gamma_sqr * rs2_read_value_evals[i])
            });
        }
        UniPoly::from_evals_and_hint(previous_claim, &evals)
    }

    fn bind(&mut self, r_j: F::Challenge) {
        self.rd_write_value_poly
            .bind_parallel(r_j, BindingOrder::LowToHigh);
        self.rs1_read_value_poly
            .bind_parallel(r_j, BindingOrder::LowToHigh);
        self.rs2_read_value_poly
            .bind_parallel(r_j, BindingOrder::LowToHigh);
        self.eq_poly.bind_parallel(r_j, BindingOrder::LowToHigh);
    }
}

/// A sumcheck instance for:
///
/// ```text
/// sum_j eq(r_spartan, j) * (RdWriteValue(j) + gamma * Rs1Value(j) + gamma^2 * Rs2Value(j))
/// ```
///
/// where `r_spartan` is the randomness from the log(T) rounds of Spartan outer sumcheck (stage 1).
///
/// The purpose of this sumcheck is to aggregate rd/rs1/rs2 value claims at a single point. It runs in
/// parallel with the Spartan instruction input sumcheck. This facilitates only handling a single set of claims
/// for the instruction input sumcheck.
pub struct RegistersClaimReductionSumcheckVerifier<F: JoltField> {
    params: RegistersClaimReductionSumcheckParams<F>,
}

impl<F: JoltField> RegistersClaimReductionSumcheckVerifier<F> {
    pub fn new(
        trace_len: usize,
        accumulator: &VerifierOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let params = RegistersClaimReductionSumcheckParams::new(trace_len, accumulator, transcript);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
    for RegistersClaimReductionSumcheckVerifier<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let opening_point = SumcheckInstanceVerifier::<F, T>::get_params(self)
            .normalize_opening_point(sumcheck_challenges);

        let (r_spartan, _) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RdWriteValue,
            SumcheckId::SpartanOuter,
        );

        let (_, rd_write_value_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RdWriteValue,
            SumcheckId::RegistersClaimReduction,
        );
        let (_, rs1_read_value_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::Rs1Value,
            SumcheckId::RegistersClaimReduction,
        );
        let (_, rs2_read_value_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::Rs2Value,
            SumcheckId::RegistersClaimReduction,
        );

        EqPolynomial::mle(&opening_point.r, &r_spartan.r)
            * (rd_write_value_claim
                + self.params.gamma * rs1_read_value_claim
                + self.params.gamma_sqr * rs2_read_value_claim)
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = SumcheckInstanceVerifier::<F, T>::get_params(self)
            .normalize_opening_point(sumcheck_challenges);

        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RdWriteValue,
            SumcheckId::RegistersClaimReduction,
            opening_point.clone(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::Rs1Value,
            SumcheckId::RegistersClaimReduction,
            opening_point.clone(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::Rs2Value,
            SumcheckId::RegistersClaimReduction,
            opening_point,
        );
    }
}
