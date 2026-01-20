use std::array;
use std::sync::Arc;

use allocative::Allocative;
use ark_std::Zero;
use common::constants::XLEN;

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
use crate::zkvm::instruction::LookupQuery;
use crate::zkvm::witness::VirtualPolynomial;
use rayon::prelude::*;
use tracer::instruction::Cycle;

/// Degree bound of the sumcheck round polynomials in [`InstructionLookupsClaimReductionSumcheckVerifier`].
const DEGREE_BOUND: usize = 2;

#[derive(Allocative, Clone)]
pub struct InstructionLookupsClaimReductionSumcheckParams<F: JoltField> {
    pub gamma: F,
    pub gamma_sqr: F,
    pub n_cycle_vars: usize,
    pub r_spartan: OpeningPoint<BIG_ENDIAN, F>,
}

impl<F: JoltField> InstructionLookupsClaimReductionSumcheckParams<F> {
    /// 初始化 `InstructionLookupsClaimReductionSumcheckParams` 实例。
    ///
    /// # 作用
    /// 为“指令查找表主张归约”（Instruction Lookups Claim Reduction）协议准备参数。
    /// 这个协议运行在一个更大规模的 Spartan Sumcheck 协议之后（或与之并行）。
    /// 它的目的是将针对三个不同多项式 (`LookupOutput`, `LeftLookupOperand`, `RightLookupOperand`)
    /// 在随机点 `r_spartan` 上的评估检查，归约为针对单个线性组合多项式的 Sumcheck。
    ///
    /// # 核心逻辑
    /// 1. **生成随机挑战 $\gamma$**:
    ///    从 Fiat-Shamir Transcript 中获取一个随机数 $\gamma$。
    ///    这个系数用于将三个查找表相关的多项式进行随机线性组合：
    ///    $P_{combined} = \text{Output} + \gamma \cdot \text{Left} + \gamma^2 \cdot \text{Right}$。
    ///
    /// 2. **获取上下文挑战 $r_{spartan}$**:
    ///    这个 Sumcheck 并非独立存在，而是为了验证 Spartan 协议外层 Sumcheck (SpartanOuter) 产生的某个主张。
    ///    我们需要从 `accumulator` 中提取出 Spartan 协议使用的随机挑战点 $r_{spartan}$。
    ///    当前的 Sumcheck 将证明：在 $r_{spartan}$ 这一点上，各查找多项式的加权和是正确的。
    ///
    /// 3. **计算轮数**:
    ///    Sumcheck 的轮数由执行轨迹长度（`trace_len`）的对数决定。
    ///
    /// # 参数
    /// * `trace_len`: 执行轨迹的长度（Cycle 数）。必须是 2 的幂。
    /// * `accumulator`: 累加器，存储了之前协议阶段生成的挑战点和多项式开启信息。
    /// * `transcript`: Fiat-Shamir Transcript，用于生成安全的伪随机数。
    pub fn new(
        trace_len: usize,
        accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        // 1. 生成随机挑战 gamma，用于线性组合多个多项式。
        let gamma = transcript.challenge_scalar::<F>();
        // 预计算 gamma 的平方，对应于第三个多项式（RightOperand）的系数。
        let gamma_sqr = gamma.square();

        // 2. 从累加器中获取 Spartan Outer Sumcheck 阶段生成的随机挑战点 r_spartan。
        // 这个点定义了 Eq 多项式的求值位置：eq(r_spartan, x)。
        // 我们只关心挑战点 r_spartan 本身 (Tuple 的第一个元素)，忽略具体的评估值。
        let (r_spartan, _) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::LookupOutput,
            SumcheckId::SpartanOuter,
        );

        Self {
            gamma,
            gamma_sqr,
            // 3. 计算 Sumcheck 需要运行的轮数 (即变量个数)。
            n_cycle_vars: trace_len.log_2(),
            r_spartan,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for InstructionLookupsClaimReductionSumcheckParams<F> {
    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let (_, lookup_output_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::LookupOutput,
            SumcheckId::SpartanOuter,
        );
        let (_, left_operand_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::LeftLookupOperand,
            SumcheckId::SpartanOuter,
        );
        let (_, right_operand_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RightLookupOperand,
            SumcheckId::SpartanOuter,
        );
        lookup_output_claim + self.gamma * left_operand_claim + self.gamma_sqr * right_operand_claim
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

/// Sumcheck prover for [`InstructionLookupsClaimReductionSumcheckVerifier`].
#[derive(Allocative)]
pub struct InstructionLookupsClaimReductionSumcheckProver<F: JoltField> {
    phase: InstructionLookupsClaimReductionPhase<F>,
    pub params: InstructionLookupsClaimReductionSumcheckParams<F>,
}

#[derive(Allocative)]
#[allow(clippy::large_enum_variant)]
enum InstructionLookupsClaimReductionPhase<F: JoltField> {
    Phase1(InstructionLookupsPhase1State<F>), // 1st half of sumcheck rounds (prefix-suffix sumcheck)
    Phase2(InstructionLookupsPhase2State<F>), // 2nd half of sumcheck rounds (regular sumcheck)
}

impl<F: JoltField> InstructionLookupsClaimReductionSumcheckProver<F> {
    /// 初始化 `InstructionLookupsClaimReductionSumcheckProver` 实例。
    ///
    /// # 作用
    /// 启动指令查找表主张归约的 Sumcheck Prover。
    /// 该方法的关键任务是创建初始阶段（Phase 1）的状态。
    ///
    /// # 背景：混合 Sumcheck 策略 (Hybrid Sumcheck)
    /// 为了处理这种特定的 Sumcheck 实例（涉及前缀/后缀拆分的大规模多线性多项式计算），
    /// Jolt采用了两阶段策略（参考代码中的注释引用 <https://eprint.iacr.org/2025/611.pdf>）：
    /// 1. **Phase 1 (Prefix-Suffix Sumcheck)**:
    ///    处理 Sumcheck 的前几轮。在这个阶段，多项式被隐式表示为前缀（Eq 多项式部分）和后缀（Trace 数据部分）的组合。
    ///    这种方式避免了立即物化巨大的密集多项式，节省内存并允许并行计算。
    ///    `InstructionLookupsPhase1State::initialize` 会负责预计算 Phase 1 所需的 $P$ 和 $Q$ 缓冲区。
    /// 2. **Phase 2 (Regular Sumcheck)**:
    ///    在 Phase 1 缩减了足够多的变量后，协议会转换到 Phase 2。
    ///    此时剩余的问题规模足够小，可以将多项式完全物化（Materialize）为密集向量，并使用常规的线性时间 Sumcheck 算法完成剩余轮次。
    ///
    /// # 参数
    /// * `params`: 包含了随机挑战因子 $\gamma$ 和目标点 $r_{spartan}$ 的参数集。
    /// * `trace`: 完整的执行轨迹（Trace），包含了每个 Cycle 的指令和查找表信息。
    #[tracing::instrument(skip_all, name = "InstructionClaimReductionSumcheckProver::initialize")]
    pub fn initialize(
        params: InstructionLookupsClaimReductionSumcheckParams<F>,
        trace: Arc<Vec<Cycle>>,
    ) -> Self {
        // 初始化 Phase 1 状态。
        // 这一步会预计算 Eq 多项式的前缀部分，并扫描 Trace 计算后缀部分的线性组合 Q。
        // 这是这类 Sumcheck 性能优化的核心起点。
        let phase = InstructionLookupsClaimReductionPhase::Phase1(
            InstructionLookupsPhase1State::initialize(trace, &params),
        );
        
        Self { phase, params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T>
    for InstructionLookupsClaimReductionSumcheckProver<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(
        skip_all,
        name = "InstructionClaimReductionSumcheckProver::compute_message"
    )]
    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        match &self.phase {
            InstructionLookupsClaimReductionPhase::Phase1(state) => {
                state.compute_message(&self.params, previous_claim)
            }
            InstructionLookupsClaimReductionPhase::Phase2(state) => {
                state.compute_message(&self.params, previous_claim)
            }
        }
    }

    #[tracing::instrument(
        skip_all,
        name = "InstructionClaimReductionSumcheckProver::ingest_challenge"
    )]
    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        match &mut self.phase {
            InstructionLookupsClaimReductionPhase::Phase1(state) => {
                if state.should_transition_to_phase2() {
                    let mut sumcheck_challenges = state.sumcheck_challenges.clone();
                    sumcheck_challenges.push(r_j);
                    self.phase = InstructionLookupsClaimReductionPhase::Phase2(
                        InstructionLookupsPhase2State::gen(
                            &state.trace,
                            &sumcheck_challenges,
                            &self.params,
                        ),
                    );
                    return;
                }
                state.bind(r_j);
            }
            InstructionLookupsClaimReductionPhase::Phase2(state) => state.bind(r_j),
        }
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let InstructionLookupsClaimReductionPhase::Phase2(state) = &self.phase else {
            panic!("Should finish sumcheck on phase 2");
        };

        let opening_point = SumcheckInstanceProver::<F, T>::get_params(self)
            .normalize_opening_point(sumcheck_challenges);

        let lookup_output_claim = state.lookup_output_poly.final_sumcheck_claim();
        let left_lookup_operand_claim = state.left_lookup_operand_poly.final_sumcheck_claim();
        let right_lookup_operand_claim = state.right_lookup_operand_poly.final_sumcheck_claim();

        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::LookupOutput,
            SumcheckId::InstructionClaimReduction,
            opening_point.clone(),
            lookup_output_claim,
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::LeftLookupOperand,
            SumcheckId::InstructionClaimReduction,
            opening_point.clone(),
            left_lookup_operand_claim,
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RightLookupOperand,
            SumcheckId::InstructionClaimReduction,
            opening_point,
            right_lookup_operand_claim,
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

#[derive(Allocative)]
struct InstructionLookupsPhase1State<F: JoltField> {
    // Prefix-suffix P and Q buffers.
    // See <https://eprint.iacr.org/2025/611.pdf> (Appendix A).
    P: MultilinearPolynomial<F>,
    Q: MultilinearPolynomial<F>,
    #[allocative(skip)]
    trace: Arc<Vec<Cycle>>,
    sumcheck_challenges: Vec<F::Challenge>,
}

impl<F: JoltField> InstructionLookupsPhase1State<F> {
    fn initialize(
        trace: Arc<Vec<Cycle>>,
        params: &InstructionLookupsClaimReductionSumcheckParams<F>,
    ) -> Self {
        let (r_hi, r_lo) = params.r_spartan.split_at(params.r_spartan.len() / 2);
        let eq_prefix_evals = EqPolynomial::evals(&r_lo.r);
        let eq_suffix_evals = EqPolynomial::evals(&r_hi.r);
        let prefix_n_vars = r_lo.len();
        let suffix_n_vars = r_hi.len();

        // Prefix-suffix P and Q buffers.
        // See <https://eprint.iacr.org/2025/611.pdf> (Appendix A).
        let P = eq_prefix_evals;
        let mut Q = unsafe_allocate_zero_vec(1 << prefix_n_vars);

        let gamma = params.gamma;
        let gamma_sqr = params.gamma_sqr;

        const BLOCK_SIZE: usize = 32;
        Q.par_chunks_mut(BLOCK_SIZE)
            .enumerate()
            .for_each(|(chunk_i, q_chunk)| {
                let mut q_lookup_output = [F::Unreduced::<6>::zero(); BLOCK_SIZE];
                let mut q_left_lookup_operand = [F::Unreduced::<6>::zero(); BLOCK_SIZE];
                let mut q_right_lookup_operand = [F::Unreduced::<7>::zero(); BLOCK_SIZE];

                for x_hi in 0..(1 << suffix_n_vars) {
                    for i in 0..q_chunk.len() {
                        let x_lo = chunk_i * BLOCK_SIZE + i;
                        let x = x_lo + (x_hi << prefix_n_vars);
                        let cycle = &trace[x];
                        let (left_lookup, right_lookup) =
                            LookupQuery::<XLEN>::to_lookup_operands(cycle);
                        let lookup_output = LookupQuery::<XLEN>::to_lookup_output(cycle);

                        q_lookup_output[i] +=
                            eq_suffix_evals[x_hi].mul_u64_unreduced(lookup_output);
                        q_left_lookup_operand[i] +=
                            eq_suffix_evals[x_hi].mul_u64_unreduced(left_lookup);
                        q_right_lookup_operand[i] +=
                            eq_suffix_evals[x_hi].mul_u128_unreduced(right_lookup);
                    }
                }

                for (i, q) in q_chunk.iter_mut().enumerate() {
                    *q = F::from_barrett_reduce(q_lookup_output[i])
                        + gamma * F::from_barrett_reduce(q_left_lookup_operand[i])
                        + gamma_sqr * F::from_barrett_reduce(q_right_lookup_operand[i]);
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
        _params: &InstructionLookupsClaimReductionSumcheckParams<F>,
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
struct InstructionLookupsPhase2State<F: JoltField> {
    lookup_output_poly: MultilinearPolynomial<F>,
    left_lookup_operand_poly: MultilinearPolynomial<F>,
    right_lookup_operand_poly: MultilinearPolynomial<F>,
    eq_poly: MultilinearPolynomial<F>,
}

impl<F: JoltField> InstructionLookupsPhase2State<F> {
    fn gen(
        trace: &[Cycle],
        sumcheck_challenges: &[F::Challenge],
        params: &InstructionLookupsClaimReductionSumcheckParams<F>,
    ) -> Self {
        let n_remaining_rounds = params.r_spartan.len() - sumcheck_challenges.len();
        let r_prefix: OpeningPoint<BIG_ENDIAN, F> =
            OpeningPoint::<LITTLE_ENDIAN, F>::new(sumcheck_challenges.to_vec()).match_endianness();

        let eq_evals = EqPolynomial::evals(&r_prefix.r);
        let mut lookup_output_poly = unsafe_allocate_zero_vec(1 << n_remaining_rounds);
        let mut left_lookup_operand_poly = unsafe_allocate_zero_vec(1 << n_remaining_rounds);
        let mut right_lookup_operand_poly = unsafe_allocate_zero_vec(1 << n_remaining_rounds);
        (
            &mut lookup_output_poly,
            &mut left_lookup_operand_poly,
            &mut right_lookup_operand_poly,
            trace.par_chunks(eq_evals.len()),
        )
            .into_par_iter()
            .for_each(
                |(
                    lookup_output_eval,
                    left_lookup_operand_eval,
                    right_lookup_operand_eval,
                    trace_chunk,
                )| {
                    let mut lookup_output_eval_unreduced = F::Unreduced::<6>::zero();
                    let mut left_lookup_operand_eval_unreduced = F::Unreduced::<6>::zero();
                    let mut right_lookup_operand_eval_unreduced = F::Unreduced::<7>::zero();

                    for (i, cycle) in trace_chunk.iter().enumerate() {
                        let (left_lookup, right_lookup) =
                            LookupQuery::<XLEN>::to_lookup_operands(cycle);
                        let lookup_output = LookupQuery::<XLEN>::to_lookup_output(cycle);

                        lookup_output_eval_unreduced +=
                            eq_evals[i].mul_u64_unreduced(lookup_output);
                        left_lookup_operand_eval_unreduced +=
                            eq_evals[i].mul_u64_unreduced(left_lookup);
                        right_lookup_operand_eval_unreduced +=
                            eq_evals[i].mul_u128_unreduced(right_lookup);
                    }

                    *lookup_output_eval = F::from_barrett_reduce(lookup_output_eval_unreduced);
                    *left_lookup_operand_eval =
                        F::from_barrett_reduce(left_lookup_operand_eval_unreduced);
                    *right_lookup_operand_eval =
                        F::from_barrett_reduce(right_lookup_operand_eval_unreduced);
                },
            );

        let (r_hi, r_lo) = params.r_spartan.split_at(params.r_spartan.len() / 2);
        let eq_prefix_eval = EqPolynomial::mle_endian(&r_prefix, &r_lo);
        let eq_suffix_evals = EqPolynomial::evals_parallel(&r_hi.r, Some(eq_prefix_eval));

        Self {
            lookup_output_poly: lookup_output_poly.into(),
            left_lookup_operand_poly: left_lookup_operand_poly.into(),
            right_lookup_operand_poly: right_lookup_operand_poly.into(),
            eq_poly: eq_suffix_evals.into(),
        }
    }

    fn compute_message(
        &self,
        params: &InstructionLookupsClaimReductionSumcheckParams<F>,
        previous_claim: F,
    ) -> UniPoly<F> {
        let half_n = self.lookup_output_poly.len() / 2;
        let mut evals = [F::zero(); DEGREE_BOUND];
        for j in 0..half_n {
            let lookup_output_evals = self
                .lookup_output_poly
                .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
            let left_lookup_operand_evals = self
                .left_lookup_operand_poly
                .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
            let right_lookup_operand_evals = self
                .right_lookup_operand_poly
                .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
            let eq_evals = self
                .eq_poly
                .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
            evals = array::from_fn(|i| {
                evals[i]
                    + eq_evals[i]
                        * (lookup_output_evals[i]
                            + params.gamma * left_lookup_operand_evals[i]
                            + params.gamma_sqr * right_lookup_operand_evals[i])
            });
        }
        UniPoly::from_evals_and_hint(previous_claim, &evals)
    }

    fn bind(&mut self, r_j: F::Challenge) {
        self.lookup_output_poly
            .bind_parallel(r_j, BindingOrder::LowToHigh);
        self.left_lookup_operand_poly
            .bind_parallel(r_j, BindingOrder::LowToHigh);
        self.right_lookup_operand_poly
            .bind_parallel(r_j, BindingOrder::LowToHigh);
        self.eq_poly.bind_parallel(r_j, BindingOrder::LowToHigh);
    }
}

/// A sumcheck instance for:
///
/// ```text
/// sum_j eq(r_spartan, j) * (LookupOutput(j) + gamma * RightLookupOperand(j) + gamma^2 * LeftLookupOperand(j))
/// ```
///
/// where `r_spartan` is the randomness from the log(T) rounds of Spartan outer sumcheck (stage 1).
///
/// The purpose of this sumcheck is to aggregate instruction lookup claims into a single claim. It runs in
/// parallel with the Spartan product sumcheck. This optimization eliminates the need for a separate opening
/// of [`VirtualPolynomial::LookupOutput`] at `r_spartan`, leaving only the opening at `r_product` required.
pub struct InstructionLookupsClaimReductionSumcheckVerifier<F: JoltField> {
    params: InstructionLookupsClaimReductionSumcheckParams<F>,
}

impl<F: JoltField> InstructionLookupsClaimReductionSumcheckVerifier<F> {
    pub fn new(
        trace_len: usize,
        accumulator: &VerifierOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let params =
            InstructionLookupsClaimReductionSumcheckParams::new(trace_len, accumulator, transcript);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
    for InstructionLookupsClaimReductionSumcheckVerifier<F>
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
            VirtualPolynomial::LookupOutput,
            SumcheckId::SpartanOuter,
        );

        let (_, lookup_output_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::LookupOutput,
            SumcheckId::InstructionClaimReduction,
        );
        let (_, left_lookup_operand_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::LeftLookupOperand,
            SumcheckId::InstructionClaimReduction,
        );
        let (_, right_lookup_operand_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RightLookupOperand,
            SumcheckId::InstructionClaimReduction,
        );

        EqPolynomial::mle(&opening_point.r, &r_spartan.r)
            * (lookup_output_claim
                + self.params.gamma * left_lookup_operand_claim
                + self.params.gamma_sqr * right_lookup_operand_claim)
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
            VirtualPolynomial::LookupOutput,
            SumcheckId::InstructionClaimReduction,
            opening_point.clone(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::LeftLookupOperand,
            SumcheckId::InstructionClaimReduction,
            opening_point.clone(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RightLookupOperand,
            SumcheckId::InstructionClaimReduction,
            opening_point,
        );
    }
}
