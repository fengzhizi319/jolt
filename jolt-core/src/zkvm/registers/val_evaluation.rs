use num_traits::Zero;
use std::{array, sync::Arc};
use tracer::instruction::Cycle;

use crate::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        lt_poly::LtPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
        },
        ra_poly::RaPolynomial,
        unipoly::UniPoly,
    },
    subprotocols::{
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
    zkvm::{
        bytecode::BytecodePreprocessing,
        witness::{CommittedPolynomial, VirtualPolynomial},
    },
};
use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use common::{constants::REGISTER_COUNT, jolt_device::MemoryLayout};
use rayon::prelude::*;

// Register value evaluation sumcheck
//
// Proves the relation:
//   Val(r) = Σ_{j=0}^{T-1} inc(j) ⋅ wa(r_address, j) ⋅ LT(r_cycle, j)
// where:
// - r = (r_address, r_cycle) is the evaluation point from the read-write checking sumcheck.
// - Val(r) is the claimed value of register r_address at time r_cycle.
// - inc(j) is the change in value at cycle j if a write occurs, and 0 otherwise.
// - wa is the MLE of the write-indicator (1 on matching {0,1}-points).
// - LT is the MLE of strict less-than on bitstrings; evaluated at (r_cycle, j) as field points.
//
// This sumcheck ensures that the claimed final value of a register is consistent
// with all the writes that occurred to it over time (assuming initial value of 0).

const LOG_K: usize = REGISTER_COUNT.ilog2() as usize;

/// Degree bound of the sumcheck round polynomials in [`ValEvaluationSumcheckVerifier`].
const DEGREE_BOUND: usize = 3;

#[derive(Allocative, Clone)]
pub struct RegistersValEvaluationSumcheckParams<F: JoltField> {
    pub r_address: OpeningPoint<BIG_ENDIAN, F>,
    pub r_cycle: OpeningPoint<BIG_ENDIAN, F>,
}

impl<F: JoltField> RegistersValEvaluationSumcheckParams<F> {

    /// 该函数负责从上一阶段（Stage 4: 读写一致性检查）的Registers验证状态中提取随机挑战点 `r`，
    /// 并将其分解为“地址挑战”和“时间周期挑战”，以便在当前阶段（Stage 5）证明寄存器的值的正确性。
    ///
    /// # 背景逻辑
    /// 在 Stage 4 中，Verifier 针对多项式 `RegistersVal(r)` 提出了一个 claim。
    /// `r` 是一个随机点，涵盖了所有变量（地址位 + 时间位）。
    /// 为了验证这个 claim (即证明寄存器当时的值确实是那个数)，我们需要用同样的 `r` 来运行新的 Sumcheck。
    /// # 返回值
    /// 返回包含分离后的挑战点 `r_address` 和 `r_cycle` 的参数结构体
    pub fn new(opening_accumulator: &dyn OpeningAccumulator<F>) -> Self {
        // 1. 从累加器中获取与 `RegistersVal` 多项式关联的开启点 `r`。
        //    这个 `r` 是在 RegistersReadWriteChecking (Stage 4) 结束时生成的挑战点。
        //    它是一个由随机数组成的向量，维度 = log(寄存器数) + log(Trace长度)。
        //    (虽然这里只需要 r 向量,不需要对应的值 claim,所以用 _ 忽略返回值)
        let (r, _) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RegistersVal,
            SumcheckId::RegistersReadWriteChecking,
        );

        // 2. 将挑战点 `r` 切分为两部分：
        //    - `r_address`: 前 LOG_K (例如 32 个寄存器对应 5 位) 个随机数，
        //                 对应寄存器地址的二进制位。用于构造 eq(r_address, address) 多项式。
        //    - `r_cycle`: 剩余的随机数，对应时间周期 (Trace cycle) 的二进制位。
        //               用于构造 lt(r_cycle, cycle) 多项式。
        let (r_address, r_cycle) = r.split_at(LOG_K);

        Self { r_address, r_cycle }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for RegistersValEvaluationSumcheckParams<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.r_cycle.len()
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let (_, registers_val_input_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RegistersVal,
            SumcheckId::RegistersReadWriteChecking,
        );
        registers_val_input_claim
    }

    fn normalize_opening_point(
        &self,
        challenges: &[<F as JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }
}

#[derive(Allocative)]
pub struct ValEvaluationSumcheckProver<F: JoltField> {
    inc: MultilinearPolynomial<F>,
    wa: RaPolynomial<u8, F>,
    lt: LtPolynomial<F>,
    pub params: RegistersValEvaluationSumcheckParams<F>,
}

impl<F: JoltField> ValEvaluationSumcheckProver<F> {
    /// 初始化寄存器值评估 Sumcheck 的 Prover 实例
    ///
    /// # 功能
    ///
    /// 该函数负责构建当前 Sumcheck 协议所需的三个核心多项式，用于证明某个特定的寄存器（由 `params.r_address` 指定）
    /// 在特定的时间点（由 `params.r_cycle` 指定）的值是正确的。
    ///
    /// # 核心逻辑
    ///
    /// 验证公式为：$Val(r) = \sum_{j=0}^{T-1} inc(j) \cdot wa(r_{address}, j) \cdot LT(r_{cycle}, j)$
    /// 也就是：某寄存器在时刻 $t$ 的值 = 所有时刻 $j < t$ 中该寄存器增量的累加。
    ///
    /// 该函数主要构建以下多项式：
    /// 1. **inc (Increment)**: 描述每个 Cycle 中寄存器数值的变化量 (新值 - 旧值)。
    /// 2. **wa (Write Access)**: 描述每个 Cycle 操作的目标寄存器是否就是我们要查询的那个寄存器 `r_address`。
    /// 3. **lt (Less Than)**: 描述时间 $j$ 是否小于查询时间 `r_cycle`。
    ///
    /// # 参数
    ///
    /// * `params` - 包含查询挑战点 `r_address` (哪个寄存器) 和 `r_cycle` (哪个时间点)。
    /// * `trace` - 完整的执行轨迹，包含每个 Cycle 的具体指令信息。
    /// * `bytecode_preprocessing` - 字节码预处理信息，用于辅助生成 Witness。
    /// * `memory_layout` - 内存布局配置。
    #[tracing::instrument(skip_all, name = "RegistersValEvaluationSumcheckProver::initialize")]
    pub fn initialize(
        params: RegistersValEvaluationSumcheckParams<F>,
        trace: &[Cycle],
        bytecode_preprocessing: &BytecodePreprocessing,
        memory_layout: &MemoryLayout,
    ) -> Self {
        // 1. 生成 inc (Increment) 多项式
        //    这是一个 Committed Polynomial，记录了每个 Cycle 写入寄存器时值的变化量。
        //    如果某 Cycle 没有写寄存器或写入值为 0，则 inc(j) = 0。
        let inc = CommittedPolynomial::RdInc.generate_witness(
            bytecode_preprocessing,
            memory_layout,
            trace,
            None,
        );

        // 2. 预计算 eq(r_address, x) 的评估值
        //    这是一个辅助向量，用于稍后构建 wa 多项式。
        //    它表示：对于任意寄存器地址 x，它是否等于查询的目标地址 r_address。
        let eq_r_address = EqPolynomial::evals(&params.r_address.r);

        // 3. 提取 Trace 中每个步骤的目标寄存器索引 (rd)
        //    wa (Write Access) 向量存储了每个 Cycle 指令所写入的目标寄存器 ID。
        let wa: Vec<Option<u8>> = trace
            .par_iter()
            .map(|cycle| {
                let instr = cycle.instruction().normalize();
                instr.operands.rd
            })
            .collect();

        // 4. 构建 RaPolynomial (Random Access Polynomial)
        //    这个特殊的多项式结合了 Trace 中的实际寄存器操作 (wa 向量) 和查询目标 (eq_r_address)。
        //    这就构成了公式中的 wa(r_address, j) 部分：当且仅当第 j 个 Cycle 操作的寄存器等于 r_address 时，值为 1。
        let wa = RaPolynomial::new(Arc::new(wa), eq_r_address);

        // 5. 构建 LtPolynomial (Less Than Polynomial)
        //    这个多项式用于过滤时间。它接收查询时间点 r_cycle。
        //    在 Sumcheck 过程中，它保证我们只累加那些发生在查询时间点 *之前* (j < r_cycle) 的 inc 值。
        let lt = LtPolynomial::new(&params.r_cycle);

        Self {
            inc,
            wa,
            lt,
            params,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for ValEvaluationSumcheckProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(
        skip_all,
        name = "RegistersValEvaluationSumcheckProver::compute_message"
    )]
    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let [eval_at_1, eval_at_2, eval_at_inf] = (0..self.inc.len() / 2)
            .into_par_iter()
            .map(|j| {
                let inc_at_1_j = self.inc.get_bound_coeff(2 * j + 1);
                let inc_at_inf_j = inc_at_1_j - self.inc.get_bound_coeff(2 * j);
                let inc_at_2_j = inc_at_1_j + inc_at_inf_j;

                let wa_at_1_j = self.wa.get_bound_coeff(2 * j + 1);
                let wa_at_inf_j = wa_at_1_j - self.wa.get_bound_coeff(2 * j);
                let wa_at_2_j = wa_at_1_j + wa_at_inf_j;

                let lt_at_1_j = self.lt.get_bound_coeff(2 * j + 1);
                let lt_at_inf_j = lt_at_1_j - self.lt.get_bound_coeff(2 * j);
                let lt_at_2_j = lt_at_1_j + lt_at_inf_j;

                // Eval inc * wa * lt.
                [
                    (inc_at_1_j * wa_at_1_j).mul_unreduced::<9>(lt_at_1_j),
                    (inc_at_2_j * wa_at_2_j).mul_unreduced::<9>(lt_at_2_j),
                    (inc_at_inf_j * wa_at_inf_j).mul_unreduced::<9>(lt_at_inf_j),
                ]
            })
            .reduce(
                || [F::Unreduced::zero(); DEGREE_BOUND],
                |a, b| array::from_fn(|i| a[i] + b[i]),
            )
            .map(F::from_montgomery_reduce);

        let eval_at_0 = previous_claim - eval_at_1;
        UniPoly::from_evals_toom(&[eval_at_0, eval_at_1, eval_at_2, eval_at_inf])
    }

    #[tracing::instrument(
        skip_all,
        name = "RegistersValEvaluationSumcheckProver::ingest_challenge"
    )]
    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.inc.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.wa.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.lt.bind(r_j, BindingOrder::LowToHigh);
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let r_cycle: OpeningPoint<BIG_ENDIAN, F> =
            self.params.normalize_opening_point(sumcheck_challenges);
        let registers_val_input_sample = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RegistersVal,
            SumcheckId::RegistersReadWriteChecking,
        );
        let (r_address, _) = registers_val_input_sample.0.split_at(LOG_K);

        let inc_claim = self.inc.final_sumcheck_claim();
        let wa_claim = self.wa.final_sumcheck_claim();

        accumulator.append_dense(
            transcript,
            CommittedPolynomial::RdInc,
            SumcheckId::RegistersValEvaluation,
            r_cycle.r.clone(),
            inc_claim,
        );

        let r = [r_address.r.as_slice(), r_cycle.r.as_slice()].concat();
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RdWa,
            SumcheckId::RegistersValEvaluation,
            OpeningPoint::new(r),
            wa_claim,
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

pub struct ValEvaluationSumcheckVerifier<F: JoltField> {
    params: RegistersValEvaluationSumcheckParams<F>,
}

impl<F: JoltField> ValEvaluationSumcheckVerifier<F> {
    pub fn new(opening_accumulator: &VerifierOpeningAccumulator<F>) -> Self {
        let params = RegistersValEvaluationSumcheckParams::new(opening_accumulator);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
for ValEvaluationSumcheckVerifier<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let registers_val_input_sample = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RegistersVal,
            SumcheckId::RegistersReadWriteChecking,
        );
        let (_, r_cycle) = registers_val_input_sample.0.split_at(LOG_K);

        // Compute LT(r_cycle', r_cycle)
        let mut lt_eval = F::zero();
        let mut eq_term = F::one();

        let r: OpeningPoint<BIG_ENDIAN, F> =
            self.params.normalize_opening_point(sumcheck_challenges);
        for (x, y) in r.r.iter().zip(r_cycle.r.iter()) {
            lt_eval += (F::one() - x) * y * eq_term;
            eq_term *= F::one() - x - y + *x * y + *x * y;
        }

        let (_, inc_claim) = accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::RdInc,
            SumcheckId::RegistersValEvaluation,
        );
        let (_, wa_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RdWa,
            SumcheckId::RegistersValEvaluation,
        );

        // Return inc_claim * wa_claim * lt_eval
        inc_claim * wa_claim * lt_eval
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let r_cycle: OpeningPoint<BIG_ENDIAN, F> =
            self.params.normalize_opening_point(sumcheck_challenges);
        let registers_val_input_sample = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RegistersVal,
            SumcheckId::RegistersReadWriteChecking,
        );
        let (r_address, _) = registers_val_input_sample.0.split_at(LOG_K);

        // Append claims to accumulator
        accumulator.append_dense(
            transcript,
            CommittedPolynomial::RdInc,
            SumcheckId::RegistersValEvaluation,
            r_cycle.r.clone(),
        );

        let r = [r_address.r.as_slice(), r_cycle.r.as_slice()].concat();
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RdWa,
            SumcheckId::RegistersValEvaluation,
            OpeningPoint::new(r),
        );
    }
}
