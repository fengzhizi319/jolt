use std::array;
use std::sync::Arc;

use allocative::Allocative;
use ark_std::Zero;
use itertools::chain;
use tracer::instruction::Cycle;

use crate::field::JoltField;
use crate::poly::eq_plus_one_poly::{EqPlusOnePolynomial, EqPlusOnePrefixSuffixPoly};
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::multilinear_polynomial::{
    BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
};
use crate::poly::opening_proof::{
    OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
    VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
};
use crate::poly::unipoly::UniPoly;
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use crate::subprotocols::sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier};
use crate::transcripts::Transcript;
use crate::zkvm::bytecode::BytecodePreprocessing;
use crate::zkvm::instruction::{CircuitFlags, InstructionFlags};
use crate::zkvm::r1cs::inputs::ShiftSumcheckCycleState;
use crate::zkvm::witness::VirtualPolynomial;
use rayon::prelude::*;

// Spartan PC sumcheck
//
// Proves the batched identity over cycles j:
//   Σ_j EqPlusOne(r_outer, j) ⋅ (UnexpandedPC_shift(j) + γ·PC_shift(j) + γ²·IsNoop_shift(j))
//   = NextUnexpandedPC(r_outer) + γ·NextPC(r_outer) + γ²·NextIsNoop(r_outer),
//
// where:
// - EqPlusOne(r_outer, j): MLE of the function that,
//     on (i,j) returns 1 iff i = j + 1; no wrap-around at j = 2^{log T} − 1
// - UnexpandedPC_shift(j), PC_shift(j), IsNoop_shift(j):
//     SpartanShift MLEs encoding f(j+1) aligned at cycle j
// - NextUnexpandedPC(r_outer), NextPC(r_outer), NextIsNoop(r_outer)
//     are claims from Spartan outer sumcheck
// - γ: batching scalar drawn from the transcript

/// Degree bound of the sumcheck round polynomials in [`ShiftSumcheckVerifier`].
const DEGREE_BOUND: usize = 2;

#[derive(Allocative, Clone)]
pub struct ShiftSumcheckParams<F: JoltField> {
    pub gamma_powers: [F; 5],
    pub n_cycle_vars: usize, // = log(T)
    pub r_outer: OpeningPoint<BIG_ENDIAN, F>,
    pub r_product: OpeningPoint<BIG_ENDIAN, F>,
}

impl<F: JoltField> ShiftSumcheckParams<F> {
    /// 初始化移位检查参数
    ///
    /// # 参数
    /// - `n_cycle_vars`: 周期变量的数量 (即 log2(Trace长度))。
    ///   如果 Trace 有 2^k 行，那么就需要 k 个布尔变量来索引行号。
    /// - `opening_accumulator`: 之前阶段 (Stage 1/2) 的累加器。
    ///   它存储了之前所有 Sumcheck 产生的随机点 $r$ 和多项式评估值。
    /// - `transcript`: Fiat-Shamir 副本，用于生成新的密码学随机数。
    pub fn new(
        n_cycle_vars: usize,
        opening_accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        // [算法 1: Fiat-Shamir Batching]
        // 我们有多个不同的移位约束要检查（例如：PC更新、RAM指针移动等）。
        // 也就是要同时证明 C1(x)=0, C2(x)=0, ...
        // 为了高效，Verifer 提供一个随机数 gamma。
        // 我们将这些约束线性组合：Final(x) = C1(x) + gamma*C2(x) + gamma^2*C3(x)...
        //
        // 这里请求了 gamma 的 0 到 4 次幂 (共5个)，暗示可能有 5 类移位约束。
        let gamma_powers = transcript.challenge_scalar_powers(5).try_into().unwrap();

        // [算法 2: 获取 Spartan Outer 阶段的随机点]
        // 在 Stage 1 (Spartan Proof) 中，Verifier 已经挑选了一个随机向量 r_outer。
        // NextPC 是一个虚拟多项式，代表 "PC 列的下一个值"。
        // 我们需要获取之前针对这个多项式生成的随机点，以确保 Stage 3 的证明与 Stage 1 是关联的。
        let (outer_sumcheck_r, _) = opening_accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::NextPC, SumcheckId::SpartanOuter);

        // [算法 3: 变量分割 (Variable Splitting)]
        // r_outer 是一个长向量，包含 [周期变量 (Cycle Vars) | 其他变量 (Data Vars)]。
        // 移位检查 (Shift Check) 主要关心的是 "行与行" 的关系，即周期变量。
        //
        // 举例：多项式 P(x_time, x_reg)。
        // 我们只关心 x_time 部分，因为 Shift 是在时间维度上发生的 (t -> t+1)。
        let (r_outer, _rx_var) = outer_sumcheck_r.split_at(n_cycle_vars);

        // [算法 4: 获取 Product Virtualization 阶段的随机点]
        // 类似于上面，这是从另一个子协议 (Product Check) 中获取随机点。
        // NextIsNoop 可能用于处理填充行或空指令的逻辑。
        let (product_sumcheck_r, _) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::NextIsNoop,
            SumcheckId::SpartanProductVirtualization,
        );

        // 同样提取针对 "周期/时间" 维度的随机点部分。
        let (r_product, _) = product_sumcheck_r.split_at(n_cycle_vars);

        Self {
            gamma_powers,
            n_cycle_vars,
            r_outer,   // 用于评估主逻辑移位的随机点
            r_product, // 用于评估辅助逻辑移位的随机点
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for ShiftSumcheckParams<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.n_cycle_vars
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let (_, input_claim_next_pc) = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::NextPC, SumcheckId::SpartanOuter);
        let (_, input_claim_next_unexpanded_pc) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::NextUnexpandedPC,
            SumcheckId::SpartanOuter,
        );
        let (_, input_claim_next_is_virtual) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::NextIsVirtual,
            SumcheckId::SpartanOuter,
        );
        let (_, input_claim_next_is_first_in_sequence) = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::NextIsFirstInSequence,
                SumcheckId::SpartanOuter,
            );
        let (_, input_claim_next_is_noop) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::NextIsNoop,
            SumcheckId::SpartanProductVirtualization,
        );

        input_claim_next_unexpanded_pc
            + input_claim_next_pc * self.gamma_powers[1]
            + input_claim_next_is_virtual * self.gamma_powers[2]
            + input_claim_next_is_first_in_sequence * self.gamma_powers[3]
            + (F::one() - input_claim_next_is_noop) * self.gamma_powers[4]
    }

    fn normalize_opening_point(
        &self,
        challenges: &[<F as JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        normalize_opening_point(challenges)
    }
}

fn normalize_opening_point<F: JoltField>(
    challenges: &[F::Challenge],
) -> OpeningPoint<BIG_ENDIAN, F> {
    OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
}

/// Sumcheck prover for [`ShiftSumcheckVerifier`].
#[derive(Allocative)]
pub struct ShiftSumcheckProver<F: JoltField> {
    phase: ShiftSumcheckPhase<F>,
    pub params: ShiftSumcheckParams<F>,
}

#[derive(Allocative)]
#[allow(clippy::large_enum_variant)]
enum ShiftSumcheckPhase<F: JoltField> {
    Phase1(Phase1State<F>), // 1st half (prefix-suffix sc)
    Phase2(Phase2State<F>), // 2nd half (regular sc)
}

impl<F: JoltField> ShiftSumcheckProver<F> {
    #[tracing::instrument(skip_all, name = "ShiftSumcheckProver::initialize")]
    pub fn initialize(
        params: ShiftSumcheckParams<F>,
        trace: Arc<Vec<Cycle>>,
        bytecode_preprocessing: &BytecodePreprocessing,
    ) -> Self {
        let phase =
            ShiftSumcheckPhase::Phase1(Phase1State::gen(trace, bytecode_preprocessing, &params));
        Self { phase, params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for ShiftSumcheckProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(skip_all, name = "ShiftSumcheckProver::compute_message")]
    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        match &self.phase {
            ShiftSumcheckPhase::Phase1(state) => {
                state.compute_message(&self.params, previous_claim)
            }
            ShiftSumcheckPhase::Phase2(state) => {
                state.compute_message(&self.params, previous_claim)
            }
        }
    }

    #[tracing::instrument(skip_all, name = "ShiftSumcheckProver::ingest_challenge")]
    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        match &mut self.phase {
            ShiftSumcheckPhase::Phase1(state) => {
                if state.should_transition_to_phase2() {
                    let mut sumcheck_challenges = state.sumcheck_challenges.clone();
                    sumcheck_challenges.push(r_j);
                    self.phase = ShiftSumcheckPhase::Phase2(Phase2State::gen(
                        &state.trace,
                        &state.bytecode_preprocessing,
                        &sumcheck_challenges,
                        &self.params,
                    ));
                    return;
                }

                state.bind(r_j);
            }
            ShiftSumcheckPhase::Phase2(state) => state.bind(r_j),
        }
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[<F as JoltField>::Challenge],
    ) {
        let ShiftSumcheckPhase::Phase2(state) = &self.phase else {
            panic!("Should finish sumcheck on phase 2");
        };

        let unexpanded_pc_eval = state.unexpanded_pc_poly.final_sumcheck_claim();
        let pc_eval = state.pc_poly.final_sumcheck_claim();
        let is_virtual_eval = state.is_virtual_poly.final_sumcheck_claim();
        let is_first_in_sequence_eval = state.is_first_in_sequence_poly.final_sumcheck_claim();
        let is_noop_eval = state.is_noop_poly.final_sumcheck_claim();

        let opening_point = normalize_opening_point(sumcheck_challenges);
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::UnexpandedPC,
            SumcheckId::SpartanShift,
            opening_point.clone(),
            unexpanded_pc_eval,
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::PC,
            SumcheckId::SpartanShift,
            opening_point.clone(),
            pc_eval,
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::OpFlags(CircuitFlags::VirtualInstruction),
            SumcheckId::SpartanShift,
            opening_point.clone(),
            is_virtual_eval,
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::OpFlags(CircuitFlags::IsFirstInSequence),
            SumcheckId::SpartanShift,
            opening_point.clone(),
            is_first_in_sequence_eval,
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::InstructionFlags(InstructionFlags::IsNoop),
            SumcheckId::SpartanShift,
            opening_point,
            is_noop_eval,
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

pub struct ShiftSumcheckVerifier<F: JoltField> {
    params: ShiftSumcheckParams<F>,
}

impl<F: JoltField> ShiftSumcheckVerifier<F> {
    pub fn new(
        n_cycle_vars: usize,
        opening_accumulator: &VerifierOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let params = ShiftSumcheckParams::new(n_cycle_vars, opening_accumulator, transcript);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for ShiftSumcheckVerifier<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        // Get the shift evaluations from the accumulator
        let (_, unexpanded_pc_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::UnexpandedPC,
            SumcheckId::SpartanShift,
        );
        let (_, pc_claim) = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::PC, SumcheckId::SpartanShift);
        let (_, is_virtual_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::OpFlags(CircuitFlags::VirtualInstruction),
            SumcheckId::SpartanShift,
        );
        let (_, is_first_in_sequence_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::OpFlags(CircuitFlags::IsFirstInSequence),
            SumcheckId::SpartanShift,
        );
        let (_, is_noop_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::InstructionFlags(InstructionFlags::IsNoop),
            SumcheckId::SpartanShift,
        );

        let r = normalize_opening_point::<F>(sumcheck_challenges);
        let eq_plus_one_r_outer_at_shift =
            EqPlusOnePolynomial::<F>::new(self.params.r_outer.r.to_vec()).evaluate(&r.r);
        let eq_plus_one_r_product_at_shift =
            EqPlusOnePolynomial::<F>::new(self.params.r_product.r.to_vec()).evaluate(&r.r);

        [
            unexpanded_pc_claim,
            pc_claim,
            is_virtual_claim,
            is_first_in_sequence_claim,
        ]
        .iter()
        .zip(&self.params.gamma_powers)
        .map(|(eval, gamma)| *gamma * eval)
        .sum::<F>()
            * eq_plus_one_r_outer_at_shift
            + self.params.gamma_powers[4]
                * (F::one() - is_noop_claim)
                * eq_plus_one_r_product_at_shift
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[<F as JoltField>::Challenge],
    ) {
        let opening_point = normalize_opening_point(sumcheck_challenges);
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::UnexpandedPC,
            SumcheckId::SpartanShift,
            opening_point.clone(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::PC,
            SumcheckId::SpartanShift,
            opening_point.clone(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::OpFlags(CircuitFlags::VirtualInstruction),
            SumcheckId::SpartanShift,
            opening_point.clone(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::OpFlags(CircuitFlags::IsFirstInSequence),
            SumcheckId::SpartanShift,
            opening_point.clone(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::InstructionFlags(InstructionFlags::IsNoop),
            SumcheckId::SpartanShift,
            opening_point,
        );
    }
}

/// State for 1st half of the rounds.
///
/// Performs prefix-suffix sumcheck. See <https://eprint.iacr.org/2025/611.pdf> (Appendix A).
#[derive(Allocative)]
struct Phase1State<F: JoltField> {
    // All prefix-suffix (P, Q) buffers for this sumcheck.
    prefix_suffix_pairs: Vec<(MultilinearPolynomial<F>, MultilinearPolynomial<F>)>,
    // Below all stored to gen phase 2 state.
    #[allocative(skip)]
    trace: Arc<Vec<Cycle>>,
    #[allocative(skip)]
    bytecode_preprocessing: BytecodePreprocessing,
    sumcheck_challenges: Vec<F::Challenge>,
}

impl<F: JoltField> Phase1State<F> {
    /// 生成 Phase 1 的证明状态 (Pre-computation Step)
    ///
    // [算法原理: Split Sumcheck / Tensor Product]
    // 这里的核心思想是将多线性多项式 Eq(x, r) 分解为两部分：
    // Eq(x, r) = Eq(x_lo, r_lo) * Eq(x_hi, r_hi)
    //
    // - `P` (Prefix): 对应 Eq(x_lo, r_lo)，只与低位变量有关。
    // - `Q` (Suffix Sum): 对应 Σ [ Trace(x) * Eq(x_hi, r_hi) ]。
    //
    // 这里的 `gen` 函数主要负责计算向量 `Q`。
    // 它遍历整个 Trace，把 Trace 的值与 "后缀多项式" 进行内积，压缩成一个较小的向量 `Q`。
    // 这样，后续的 Sumcheck 只需要在较小的 `P` 和 `Q` 上进行 (大小为 2^prefix_vars)。
    fn gen(
        trace: Arc<Vec<Cycle>>,
        bytecode_preprocessing: &BytecodePreprocessing,
        params: &ShiftSumcheckParams<F>,
    ) -> Self {
        // 1. 初始化 Eq 多项式的 Prefix 和 Suffix 部分
        // EqPlusOnePrefixSuffixPoly 是一个专门处理 "Shift" (x -> x+1) 逻辑的工具。
        // 它不仅计算 Eq(x, r)，还能处理 Eq(x+1, r) 的情况。
        let EqPlusOnePrefixSuffixPoly {
            prefix_0: prefix_0_for_r_outer, // 对应 P 向量 (outer check)
            suffix_0: suffix_0_for_r_outer, // 对应 Suffix (outer check)
            prefix_1: prefix_1_for_r_outer,
            suffix_1: suffix_1_for_r_outer,
        } = EqPlusOnePrefixSuffixPoly::new(&params.r_outer);

        // 同样的逻辑，处理另一个校验 (Product Check)
        let EqPlusOnePrefixSuffixPoly {
            prefix_0: prefix_0_for_r_prod,
            suffix_0: suffix_0_for_r_prod,
            prefix_1: prefix_1_for_r_prod,
            suffix_1: suffix_1_for_r_prod,
        } = EqPlusOnePrefixSuffixPoly::new(&params.r_product);

        let prefix_n_vars = prefix_0_for_r_outer.len().ilog2(); // 低位变量数
        let suffix_n_vars = suffix_0_for_r_outer.len().ilog2(); // 高位变量数

        // [变量绑定]
        // P 向量直接来自 Prefix 多项式评估值 (大小 2^prefix)。
        let P_0_for_r_outer = prefix_0_for_r_outer;
        let P_1_for_r_outer = prefix_1_for_r_outer;
        let P_0_for_r_prod = prefix_0_for_r_prod;
        let P_1_for_r_prod = prefix_1_for_r_prod;

        // Q 向量用于存储 "Trace数据 * Suffix多项式" 的部分和。
        // 初始化大小为 2^prefix，这比原始 Trace (2^total) 小得多。
        let mut Q_0_for_r_outer = vec![F::zero(); 1 << prefix_n_vars];
        let mut Q_1_for_r_outer = vec![F::zero(); 1 << prefix_n_vars];
        let mut Q_0_for_r_prod = vec![F::zero(); 1 << prefix_n_vars];
        let mut Q_1_for_r_prod = vec![F::zero(); 1 << prefix_n_vars];

        const BLOCK_SIZE: usize = 32; // 并行处理块大小

        // 2. 并行计算 Q 向量 (The Heavy Lifting)
        // 这个循环计算：Q[x_lo] = Σ_{x_hi} ( Trace(x_lo, x_hi) * Suffix(x_hi) )
        // 这相当于把巨大的 Trace 矩阵沿着 "高位" 维度压缩（Marginalization）。
        (
            Q_0_for_r_outer.par_chunks_mut(BLOCK_SIZE),
            Q_1_for_r_outer.par_chunks_mut(BLOCK_SIZE),
            Q_0_for_r_prod.par_chunks_mut(BLOCK_SIZE),
            Q_1_for_r_prod.par_chunks_mut(BLOCK_SIZE),
        )
            .into_par_iter()
            .enumerate()
            .for_each(
                |(
                     chunk_i, // 对应 x_lo 的分块索引
                     (
                         Q_0_for_r_outer_chunk,
                         Q_1_for_r_outer_chunk,
                         Q_0_for_r_prod_chunk,
                         Q_1_for_r_prod_chunk,
                     ),
                 )| {
                    let chunk_len = Q_0_for_r_outer_chunk.len();

                    // [算法优化: Delayed Reduction]
                    // 为了减少昂贵的模运算 (Modular Reduction)，我们使用 F::Unreduced 类型。
                    // 它可以累加多次乘法结果，直到达到上限（如9次）才取模一次。
                    let mut Q_0_for_r_outer_unreduced = [F::Unreduced::<9>::zero(); BLOCK_SIZE];
                    let mut Q_1_for_r_outer_unreduced = [F::Unreduced::<9>::zero(); BLOCK_SIZE];
                    let mut Q_0_for_r_prod_unreduced = [F::Unreduced::<5>::zero(); BLOCK_SIZE];
                    let mut Q_1_for_r_prod_unreduced = [F::Unreduced::<5>::zero(); BLOCK_SIZE];

                    // 遍历所有的高位组合 (Suffix part)
                    for x_hi in 0..1 << suffix_n_vars {
                        for i in 0..chunk_len {
                            // 构造完整的 Trace 索引 x
                            // x = x_lo + (x_hi * 2^prefix_len)
                            // 这里 chunk_i * BLOCK_SIZE + i 就是 x_lo (低位)
                            let x_lo = chunk_i * BLOCK_SIZE + i;
                            let x = x_lo + (x_hi << prefix_n_vars);

                            // 从 Trace 中解析第 x 行的状态
                            let ShiftSumcheckCycleState {
                                unexpanded_pc,
                                pc,
                                is_virtual,
                                is_first_in_sequence,
                                is_noop,
                            } = ShiftSumcheckCycleState::new(&trace[x], bytecode_preprocessing);

                            // [算法原理: Random Linear Combination (RLC)]
                            // 将多个检查项 (PC正确性, 虚拟标志等) 用 gamma 的幂次加权合并。
                            // v 代表了第 x 行 Trace 数据的 "指纹"。
                            let mut v =
                                F::from_u64(unexpanded_pc) + params.gamma_powers[1].mul_u64(pc);
                            if is_virtual {
                                v += params.gamma_powers[2];
                            }
                            if is_first_in_sequence {
                                v += params.gamma_powers[3];
                            }

                            // 累加到 Q 向量中
                            // Q_unreduced[i] += v * Suffix[x_hi]
                            // 注意：这里用的是 x_hi 作为 Suffix 的索引
                            Q_0_for_r_outer_unreduced[i] +=
                                v.mul_unreduced::<9>(suffix_0_for_r_outer[x_hi]);
                            Q_1_for_r_outer_unreduced[i] +=
                                v.mul_unreduced::<9>(suffix_1_for_r_outer[x_hi]);

                            // 处理 Product Check 的部分 (Noop 逻辑)
                            if !is_noop {
                                Q_0_for_r_prod_unreduced[i] +=
                                    *suffix_0_for_r_prod[x_hi].as_unreduced_ref();
                                Q_1_for_r_prod_unreduced[i] +=
                                    *suffix_1_for_r_prod[x_hi].as_unreduced_ref();
                            }
                        }
                    }

                    // 循环结束后，执行一次统一的取模还原 (Reduction)
                    for i in 0..chunk_len {
                        // Montgomery Reduction 适用于一般的域乘法结果
                        Q_0_for_r_outer_chunk[i] =
                            F::from_montgomery_reduce(Q_0_for_r_outer_unreduced[i]);
                        Q_1_for_r_outer_chunk[i] =
                            F::from_montgomery_reduce(Q_1_for_r_outer_unreduced[i]);
                        // Barrett Reduction 适用于特定的加法累积结果
                        Q_0_for_r_prod_chunk[i] =
                            F::from_barrett_reduce(Q_0_for_r_prod_unreduced[i]);
                        Q_1_for_r_prod_chunk[i] =
                            F::from_barrett_reduce(Q_1_for_r_prod_unreduced[i]);
                    }
                },
            );

        // 对 Q_prod 应用 gamma 权重 (RLC 的最后一部分)
        chain!(&mut Q_0_for_r_prod, &mut Q_1_for_r_prod).for_each(|v| *v *= params.gamma_powers[4]);

        let prefix_suffix_pairs = vec![
            (P_0_for_r_outer.into(), Q_0_for_r_outer.into()),
            (P_1_for_r_outer.into(), Q_1_for_r_outer.into()),
            (P_0_for_r_prod.into(), Q_0_for_r_prod.into()),
            (P_1_for_r_prod.into(), Q_1_for_r_prod.into()),
        ];

        Self {
            prefix_suffix_pairs,
            trace,
            bytecode_preprocessing: bytecode_preprocessing.clone(),
            sumcheck_challenges: Vec::new(),
        }
    }

    /// 计算当前 Sumcheck 轮次的消息 (多项式评估值)
    ///
    /// [算法原理: Sumcheck Round Execution]
    /// 在 Phase 1 中，我们要证明的式子变成了 Σ_{x_lo} P(x_lo) * Q(x_lo)。
    /// 每一轮 Sumcheck，Prover 都要发送一个单变量多项式（通常是二次或三次）。
    /// 这个函数计算该多项式在特定点（0, 1, ...）的评估值。
    fn compute_message(&self, _params: &ShiftSumcheckParams<F>, previous_claim: F) -> UniPoly<F> {
        let evals = self
            .prefix_suffix_pairs
            .par_iter()
            .map(|(p, q)| {
                // 并行处理每一对 (P, Q)
                let mut evals = [F::zero(); DEGREE_BOUND];
                // 遍历 P 和 Q 的前半部分（折叠操作）
                for i in 0..p.len() / 2 {
                    // [算法: Linear Time Sumcheck]
                    // 计算 sumcheck_evals_array 是为了获得多项式在下一轮折叠需要的点。
                    // BindingOrder::LowToHigh 表示我们正在从低位变量开始折叠 x_0, x_1...
                    let p_evals =
                        p.sumcheck_evals_array::<DEGREE_BOUND>(i, BindingOrder::LowToHigh);
                    let q_evals =
                        q.sumcheck_evals_array::<DEGREE_BOUND>(i, BindingOrder::LowToHigh);

                    // 对应点相乘并累加： g(r) = P(r) * Q(r)
                    evals = array::from_fn(|i| evals[i] + p_evals[i] * q_evals[i]);
                }
                evals
            })
            // 归约求和所有分块的结果
            .reduce(
                || [F::zero(); DEGREE_BOUND],
                |a, b| array::from_fn(|i| a[i] + b[i]),
            );

        // 使用之前的 Claim 和计算出的评估点构造单变量多项式
        UniPoly::from_evals_and_hint(previous_claim, &evals)
    }

    /// 绑定变量 (Bind / Fold)
    ///
    /// [算法原理: Polynomial Folding]
    /// 当 Verifier 发送挑战 r_j 后，Prover 需要将多变量多项式 P(x_0, ..., x_k) 和 Q
    /// 在变量 x_j = r_j 处进行求值/折叠，使其变数减少 1。
    /// P_new(x) = P(x, 0) + r_j * (P(x, 1) - P(x, 0))
    fn bind(&mut self, r_j: F::Challenge) {
        assert!(!self.should_transition_to_phase2());
        self.sumcheck_challenges.push(r_j);
        self.prefix_suffix_pairs.iter_mut().for_each(|(p, q)| {
            // 对 P 和 Q 向量分别进行折叠
            // 长度减半： 2^k -> 2^{k-1}
            p.bind(r_j, BindingOrder::LowToHigh);
            q.bind(r_j, BindingOrder::LowToHigh);
        });
    }

    /// 检查是否应转换到 Phase 2
    ///
    /// 当 P 和 Q 向量被折叠到只剩常数项（长度为1，即 log2(len)==0 实际上通常保留到最后一层）时，
    /// Phase 1 (Prefix Sumcheck) 结束。
    /// 接下来的 Phase 2 将处理 Suffix 部分的验证。
    fn should_transition_to_phase2(&self) -> bool {
        self.prefix_suffix_pairs[0].0.len().ilog2() == 1
    }
}

/// State for 2nd half of the rounds.
#[derive(Allocative)]
struct Phase2State<F: JoltField> {
    unexpanded_pc_poly: MultilinearPolynomial<F>,
    pc_poly: MultilinearPolynomial<F>,
    is_virtual_poly: MultilinearPolynomial<F>,
    is_first_in_sequence_poly: MultilinearPolynomial<F>,
    is_noop_poly: MultilinearPolynomial<F>,
    eq_plus_one_r_outer: MultilinearPolynomial<F>,
    eq_plus_one_r_product: MultilinearPolynomial<F>,
}

impl<F: JoltField> Phase2State<F> {
    fn gen(
        trace: &[Cycle],
        bytecode_preprocessing: &BytecodePreprocessing,
        sumcheck_challenges: &[F::Challenge],
        params: &ShiftSumcheckParams<F>,
    ) -> Self {
        let n_remaining_rounds = params.r_outer.len() - sumcheck_challenges.len();
        let r_prefix: OpeningPoint<BIG_ENDIAN, F> =
            OpeningPoint::<LITTLE_ENDIAN, F>::new(sumcheck_challenges.to_vec()).match_endianness();

        // Gen eq+1(r_outer, (r_prefix, j)) for all j.
        let EqPlusOnePrefixSuffixPoly {
            prefix_0,
            suffix_0,
            prefix_1,
            suffix_1,
        } = EqPlusOnePrefixSuffixPoly::new(&params.r_outer);
        let prefix_0_eval = MultilinearPolynomial::from(prefix_0).evaluate(&r_prefix.r);
        let prefix_1_eval = MultilinearPolynomial::from(prefix_1).evaluate(&r_prefix.r);
        let eq_plus_one_r_outer: MultilinearPolynomial<F> = (0..suffix_0.len())
            .map(|i| prefix_0_eval * suffix_0[i] + prefix_1_eval * suffix_1[i])
            .collect::<Vec<F>>()
            .into();

        // Gen eq+1(r_product, (r_prefix, j)) for all j.
        let EqPlusOnePrefixSuffixPoly {
            prefix_0,
            suffix_0,
            prefix_1,
            suffix_1,
        } = EqPlusOnePrefixSuffixPoly::new(&params.r_product);
        let prefix_0_eval = MultilinearPolynomial::from(prefix_0).evaluate(&r_prefix.r);
        let prefix_1_eval = MultilinearPolynomial::from(prefix_1).evaluate(&r_prefix.r);
        let eq_plus_one_r_product: MultilinearPolynomial<F> = (0..suffix_0.len())
            .map(|i| prefix_0_eval * suffix_0[i] + prefix_1_eval * suffix_1[i])
            .collect::<Vec<F>>()
            .into();

        // Gen MLEs: UnexpandedPc(r_prefix, j), Pc(r_prefix, j), ...
        let eq_evals = EqPolynomial::evals(&r_prefix.r);
        let mut unexpanded_pc_poly = vec![F::zero(); 1 << n_remaining_rounds];
        let mut pc_poly = vec![F::zero(); 1 << n_remaining_rounds];
        let mut is_virtual_poly = vec![F::zero(); 1 << n_remaining_rounds];
        let mut is_first_in_sequence_poly = vec![F::zero(); 1 << n_remaining_rounds];
        let mut is_noop_poly = vec![F::zero(); 1 << n_remaining_rounds];
        (
            &mut unexpanded_pc_poly,
            &mut pc_poly,
            &mut is_virtual_poly,
            &mut is_first_in_sequence_poly,
            &mut is_noop_poly,
            trace.par_chunks(eq_evals.len()),
        )
            .into_par_iter()
            .for_each(
                |(
                    unexpanded_pc_eval,
                    pc_eval,
                    is_virtual_eval,
                    is_first_in_sequence_eval,
                    is_noop_eval,
                    trace_chunk,
                )| {
                    let mut unexpanded_pc_eval_unreduced = F::Unreduced::<5>::zero();
                    let mut pc_eval_unreduced = F::Unreduced::<6>::zero();
                    let mut is_virtual_eval_unreduced = F::Unreduced::<5>::zero();
                    let mut is_first_in_sequence_eval_unreduced = F::Unreduced::<5>::zero();
                    let mut is_noop_eval_unreduced = F::Unreduced::<5>::zero();

                    for (i, cycle) in trace_chunk.iter().enumerate() {
                        let ShiftSumcheckCycleState {
                            unexpanded_pc,
                            pc,
                            is_virtual,
                            is_first_in_sequence,
                            is_noop,
                        } = ShiftSumcheckCycleState::new(cycle, bytecode_preprocessing);
                        let eq_eval = eq_evals[i];
                        unexpanded_pc_eval_unreduced += eq_eval.mul_u64_unreduced(unexpanded_pc);
                        pc_eval_unreduced += eq_eval.mul_u64_unreduced(pc);
                        if is_virtual {
                            is_virtual_eval_unreduced += *eq_eval.as_unreduced_ref();
                        }
                        if is_first_in_sequence {
                            is_first_in_sequence_eval_unreduced += *eq_eval.as_unreduced_ref();
                        }
                        if is_noop {
                            is_noop_eval_unreduced += *eq_eval.as_unreduced_ref();
                        }
                    }

                    *unexpanded_pc_eval = F::from_barrett_reduce(unexpanded_pc_eval_unreduced);
                    *pc_eval = F::from_barrett_reduce(pc_eval_unreduced);
                    *is_virtual_eval = F::from_barrett_reduce(is_virtual_eval_unreduced);
                    *is_first_in_sequence_eval =
                        F::from_barrett_reduce(is_first_in_sequence_eval_unreduced);
                    *is_noop_eval = F::from_barrett_reduce(is_noop_eval_unreduced);
                },
            );

        Self {
            unexpanded_pc_poly: unexpanded_pc_poly.into(),
            pc_poly: pc_poly.into(),
            is_virtual_poly: is_virtual_poly.into(),
            is_first_in_sequence_poly: is_first_in_sequence_poly.into(),
            is_noop_poly: is_noop_poly.into(),
            eq_plus_one_r_outer,
            eq_plus_one_r_product,
        }
    }

    fn compute_message(&self, params: &ShiftSumcheckParams<F>, previous_claim: F) -> UniPoly<F> {
        let half_n = self.unexpanded_pc_poly.len() / 2;
        let mut evals = [F::zero(); DEGREE_BOUND];
        for j in 0..half_n {
            let unexpanded_pc_evals = self
                .unexpanded_pc_poly
                .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
            let pc_evals = self
                .pc_poly
                .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
            let is_virtual_evals = self
                .is_virtual_poly
                .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
            let is_first_in_sequence_evals = self
                .is_first_in_sequence_poly
                .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
            let is_noop_evals = self
                .is_noop_poly
                .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
            let eq_plus_one_r_outer_evals = self
                .eq_plus_one_r_outer
                .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
            let eq_plus_one_r_product_evals = self
                .eq_plus_one_r_product
                .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
            evals = array::from_fn(|i| {
                evals[i]
                    + eq_plus_one_r_outer_evals[i]
                        * (unexpanded_pc_evals[i]
                            + params.gamma_powers[1] * pc_evals[i]
                            + params.gamma_powers[2] * is_virtual_evals[i]
                            + params.gamma_powers[3] * is_first_in_sequence_evals[i])
                    + params.gamma_powers[4]
                        * eq_plus_one_r_product_evals[i]
                        * (F::one() - is_noop_evals[i])
            });
        }

        UniPoly::from_evals_and_hint(previous_claim, &evals)
    }

    fn bind(&mut self, r_j: F::Challenge) {
        let Self {
            unexpanded_pc_poly,
            pc_poly,
            is_virtual_poly,
            is_first_in_sequence_poly,
            is_noop_poly,
            eq_plus_one_r_outer,
            eq_plus_one_r_product,
        } = self;
        unexpanded_pc_poly.bind(r_j, BindingOrder::LowToHigh);
        pc_poly.bind(r_j, BindingOrder::LowToHigh);
        is_virtual_poly.bind(r_j, BindingOrder::LowToHigh);
        is_first_in_sequence_poly.bind(r_j, BindingOrder::LowToHigh);
        is_noop_poly.bind(r_j, BindingOrder::LowToHigh);
        eq_plus_one_r_outer.bind(r_j, BindingOrder::LowToHigh);
        eq_plus_one_r_product.bind(r_j, BindingOrder::LowToHigh);
    }
}
