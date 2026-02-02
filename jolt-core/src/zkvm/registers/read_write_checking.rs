use std::sync::Arc;

use crate::poly::multilinear_polynomial::PolynomialEvaluation;
use crate::subprotocols::read_write_matrix::{
    AddressMajorMatrixEntry, ReadWriteMatrixAddressMajor, ReadWriteMatrixCycleMajor,
    RegistersAddressMajorEntry, RegistersCycleMajorEntry,
};
use crate::zkvm::bytecode::BytecodePreprocessing;
use crate::zkvm::config::ReadWriteConfig;
use crate::zkvm::witness::VirtualPolynomial;
use crate::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN,
        },
        split_eq_poly::GruenSplitEqPolynomial,
        unipoly::UniPoly,
    },
    subprotocols::{
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
    utils::math::Math,
    zkvm::witness::CommittedPolynomial,
};
use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use common::constants::REGISTER_COUNT;
use common::jolt_device::MemoryLayout;
use num::Integer;
use num_traits::Zero;
use rayon::prelude::*;
use tracing::info;
use tracer::instruction::Cycle;

// Register read-write checking sumcheck
//
// Proves the combined relation
//   Σ_j eq(r_cycle, j) ⋅ ( RdWriteValue(j) + γ⋅ReadVals(j) )
//     = rd_wv_claim + γ⋅rs1_rv_claim + γ²⋅rs2_rv_claim
// where:
// - eq(r_cycle, ·) is the equality MLE over the cycle index j, evaluated at challenge point r_cycle.
// - RdWriteValue(j)   = Σ_k wa(k,j)⋅(inc(j)+Val(k,j));
// - ReadVals(j)       = Σ_k [ ra1(k,j)⋅Val(k,j) + γ⋅ra2(k,j)⋅Val(k,j) ];
// - wa(k,j) = 1 if register k is written at cycle j (rd = k), 0 otherwise;
// - ra1(k,j) = 1 if register k is read at cycle j (rs1 = k), 0 otherwise;
// - ra2(k,j) = 1 if register k is read at cycle j (rs2 = k), 0 otherwise;
// - Val(k,j) is the value of register k right before cycle j;
// - inc(j) is the change in value at cycle j if a write occurs, and 0 otherwise.
//
// This sumcheck ensures that the values read from and written to registers are consistent
// with the execution trace.

const K: usize = REGISTER_COUNT as usize;
const LOG_K: usize = REGISTER_COUNT.ilog2() as usize;

/// Degree bound of the sumcheck round polynomials in [`RegistersReadWriteCheckingVerifier`].
const DEGREE_BOUND: usize = 3;

#[derive(Allocative, Clone)]
pub struct RegistersReadWriteCheckingParams<F: JoltField> {
    pub gamma: F,
    pub T: usize,
    pub r_cycle: OpeningPoint<BIG_ENDIAN, F>,
    /// Number of cycle variables to bind in phase 1.
    pub phase1_num_rounds: usize,
    /// Number of address variables to bind in phase 2.
    pub phase2_num_rounds: usize,
}

impl<F: JoltField> RegistersReadWriteCheckingParams<F> {
    pub fn new(
        trace_length: usize,
        opening_accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
        config: &ReadWriteConfig,
    ) -> Self {
        let gamma = transcript.challenge_scalar::<F>();
        let (r_cycle, _) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RdWriteValue,
            SumcheckId::RegistersClaimReduction,
        );
        Self {
            gamma,
            T: trace_length,
            r_cycle,
            phase1_num_rounds: config.registers_rw_phase1_num_rounds as usize,
            phase2_num_rounds: config.registers_rw_phase2_num_rounds as usize,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for RegistersReadWriteCheckingParams<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        LOG_K + self.T.log_2()
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let (_, rd_wv_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RdWriteValue,
            SumcheckId::RegistersClaimReduction,
        );
        let (_, rs1_rv_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::Rs1Value,
            SumcheckId::RegistersClaimReduction,
        );
        let (_, rs1_rv_claim_instruction_input) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::Rs1Value,
            SumcheckId::InstructionInputVirtualization,
        );
        // TODO: Make error and move to more appropriate place.
        assert_eq!(rs1_rv_claim, rs1_rv_claim_instruction_input);
        let (_, rs2_rv_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::Rs2Value,
            SumcheckId::RegistersClaimReduction,
        );
        let (_, rs2_rv_claim_instruction_input) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::Rs2Value,
            SumcheckId::InstructionInputVirtualization,
        );
        // TODO: Make error and move to more appropriate place.
        assert_eq!(rs2_rv_claim, rs2_rv_claim_instruction_input);

        rd_wv_claim + self.gamma * (rs1_rv_claim + self.gamma * rs2_rv_claim)
    }

    // Invariant: we want big-endian, with address variables being "higher" than cycle variables
    fn normalize_opening_point(
        &self,
        sumcheck_challenges: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        // Cycle variables are bound low-to-high in phase 1
        let (phase1_challenges, sumcheck_challenges) =
            sumcheck_challenges.split_at(self.phase1_num_rounds);
        // Address variables are bound low-to-high in phase 2
        let (phase2_challenges, sumcheck_challenges) =
            sumcheck_challenges.split_at(self.phase2_num_rounds);
        // Remaining cycle variables, then address variables are
        // bound low-to-high in phase 3
        let (phase3_cycle_challenges, phase3_address_challenges) =
            sumcheck_challenges.split_at(self.T.log_2() - self.phase1_num_rounds);

        // Both Phase 1/2 (GruenSplitEqPolynomial LowToHigh) and Phase 3 (dense LowToHigh)
        // bind variables from the "bottom" (last w component) to "top" (first w component).
        // So all challenges need to be reversed to get big-endian [w[0], w[1], ...] order.
        let r_cycle: Vec<_> = phase3_cycle_challenges
            .iter()
            .rev()
            .copied()
            .chain(phase1_challenges.iter().rev().copied())
            .collect();
        let r_address: Vec<_> = phase3_address_challenges
            .iter()
            .rev()
            .copied()
            .chain(phase2_challenges.iter().rev().copied())
            .collect();

        [r_address, r_cycle].concat().into()
    }
}

/// Sumcheck prover for [`RegistersReadWriteCheckingVerifier`].
#[derive(Allocative)]
pub struct RegistersReadWriteCheckingProver<F: JoltField> {
    sparse_matrix_phase1: ReadWriteMatrixCycleMajor<F, RegistersCycleMajorEntry<F>>,
    sparse_matrix_phase2: ReadWriteMatrixAddressMajor<F, RegistersAddressMajorEntry<F>>,
    gruen_eq: Option<GruenSplitEqPolynomial<F>>,
    inc: MultilinearPolynomial<F>,
    #[allocative(skip)]
    trace: Arc<Vec<Cycle>>,
    // The following polynomials are instantiated after
    // the second phase
    ra: Option<MultilinearPolynomial<F>>,
    wa: Option<MultilinearPolynomial<F>>,
    val: Option<MultilinearPolynomial<F>>,
    merged_eq: Option<MultilinearPolynomial<F>>,
    pub params: RegistersReadWriteCheckingParams<F>,
}

impl<F: JoltField> RegistersReadWriteCheckingProver<F> {
    #[tracing::instrument(skip_all, name = "RegistersReadWriteCheckingProver::initialize")]
    pub fn initialize(
        params: RegistersReadWriteCheckingParams<F>,
        trace: Arc<Vec<Cycle>>, // 完整的执行轨迹
        bytecode_preprocessing: &BytecodePreprocessing,
        memory_layout: &MemoryLayout,
    ) -> Self {
        let r_prime = &params.r_cycle; // Verifier 提供的随机挑战点 r

        // ========================================================================
        // 1. 初始化 Eq 多项式 (Eq Polynomial Setup)
        // ========================================================================
        // [算法原理: Gruen's Split Sumcheck Optimization]
        // 这里的逻辑是为了支持 Split Sumcheck (分块求和检查)。
        // 如果 phase1_num_rounds > 0，说明使用了分块优化，我们需要构造 GruenSplitEqPolynomial。
        // 这允许我们将巨大的 Trace 分解为 Tensor Product (张量积) 形式，加速证明。
        let (gruen_eq, merged_eq) = if params.phase1_num_rounds > 0 {
            (
                Some(GruenSplitEqPolynomial::new(
                    &r_prime.r,
                    BindingOrder::LowToHigh,
                )),
                None,
            )
        } else {
            // 如果没有 Phase 1 (小规模测试或特定配置)，则退化为标准的多线性扩展 Eq。
            (
                None,
                Some(MultilinearPolynomial::from(EqPolynomial::evals(&r_prime.r))),
            )
        };

        // ========================================================================
        // 2. 生成 Inc 多项式 (Increment Polynomial)
        // ========================================================================
        // [约束原理: Timestamp / Counter Logic]
        // 内存检查需要区分操作的先后顺序。通常使用全局计数器。
        // 但在 Jolt 中，"Inc" 用于标记寄存器是否发生变化的增量位，每个trace会得到一个变化值，没有变化的为0。trace.len()=inc.len()
        // 此多项式由 CommittedPolynomial 系统自动生成，代表了每一行 Trace 的辅助状态。

        info!("trace.len(): {:?}", trace.len());
        let inc = CommittedPolynomial::RdInc.generate_witness(
            bytecode_preprocessing,
            memory_layout,
            &trace,
            None,
        );


        // ========================================================================
        // 3. 构建稀疏读写矩阵 (Sparse Read-Write Matrix)
        // ========================================================================
        // [算法原理: Sparse Matrix Representation for Memory Checking]
        // 这是核心数据结构。
        // 内存检查本质上是比较两个集合：{Writes + Init} 和 {Reads + Final}。
        // 但寄存器访问是稀疏的（相对于整个 RAM 空间，或者相对于多变量域）。
        //
        // `ReadWriteMatrixCycleMajor` 负责将 Trace 转换为适合 Grand Product 计算的矩阵形式。
        // 它使用 params.gamma 将 (Address, Value, Time) 压缩成单一指纹。
        let sparse_matrix =
            ReadWriteMatrixCycleMajor::<_, RegistersCycleMajorEntry<F>>::new(&trace, params.gamma);

        // ========================================================================
        // 4. 阶段划分 (Phase Splitting)
        // ========================================================================
        // [算法原理: Multi-Phase Sumcheck]
        // 类似于 Stage 3，Stage 4 也可能将证明拆分为多个阶段以优化内存或计算。
        // 这里根据配置将稀疏矩阵移动到 Phase 1 或 Phase 2 的槽位中。
        let phase1_rounds = params.phase1_num_rounds;
        let phase2_rounds = params.phase2_num_rounds;

        let (sparse_matrix_phase1, sparse_matrix_phase2) = if phase1_rounds > 0 {
            (sparse_matrix, Default::default())
        } else if phase2_rounds > 0 {
            (Default::default(), sparse_matrix.into())
        } else {
            unimplemented!("Unsupported configuration: both phase 1 and phase 2 are 0 rounds")
        };

        Self {
            sparse_matrix_phase1,
            sparse_matrix_phase2,
            gruen_eq,
            merged_eq,
            inc,
            // 下面这些字段 (ra, wa, val) 是 Grand Product 的中间值，
            // 将在 Sumcheck 过程中动态计算，此处初始化为 None。
            ra: None,
            wa: None,
            val: None,
            params,
            trace,
        }
    }


    /// 计算 Phase 1 (Cycle 绑定阶段) 的 Sumcheck 消息多项式。
    ///
    /// 此阶段的目标是消除时间维度 (Trace Cycles)。
    /// Sumcheck 公式的每一轮都需要 Prover 发送一个单变量多项式 $M_j(X)$。
    ///
    /// ### 数学原理: Gruen's Split Sumcheck Optimization
    /// 为了处理巨大的 Trace (例如 $2^{20}$)，我们不物化完整的 $Eq(r, x)$ 表。
    /// 而是利用 $Eq$ 的张量积结构: $Eq(r, x) = E_{out}(x_{out}) \cdot E_{in}(x_{in})$。
    ///
    /// 总和计算公式:
    /// $$ \sum_{x} Eq(r, x) \cdot G(x) = \sum_{x_{out}} E_{out}(x_{out}) \cdot \left( \sum_{x_{in}} E_{in}(x_{in}) \cdot G(x_{out}, x_{in}) \right) $$
    fn phase1_compute_message(&mut self, previous_claim: F) -> UniPoly<F> {
        let Self {
            inc,
            gruen_eq,
            params,
            sparse_matrix_phase1: sparse_matrix,
            ..
        } = self;
        let gruen_eq = gruen_eq.as_ref().unwrap();

        // 1. 准备 Gruen 优化参数
        // E_in_current 返回当前未绑定的内部变量的 Eq 评估表。
        // 当 e_in_len <= 1 时，意味着内部变量已被完全绑定，此时只能进行普通的行处理。
        let e_in = gruen_eq.E_in_current();
        let e_in_len = e_in.len();
        let num_x_in_bits = e_in_len.max(1).log_2(); // 计算 x_in 覆盖的比特数
        let x_bitmask = (1 << num_x_in_bits) - 1;

        // 2. 并行计算二次项系数 (Quadratic Coefficients)
        // 这里的 entries 是稀疏矩阵的行条目 (Cycle Major)。
        // 我们的目标是计算当前轮变量 X 取 0 和 1 时的加权和 (evals[0], evals[1])。
        let quadratic_coeffs: [F; DEGREE_BOUND - 1] = sparse_matrix
            .entries
            // [并行分块 A]: 按 x_out (Cycle 高位) 进行分块处理
            // ((a.row / 2) >> num_x_in_bits) 提取了 x_out 的索引。
            // 每一块对应 Sumcheck 公式外层求和的一项。
            .par_chunk_by(|a, b| ((a.row / 2) >> num_x_in_bits) == ((b.row / 2) >> num_x_in_bits))
            .map(|entries| {
                // 获取外层权重 E_out(x_out)
                let x_out = (entries[0].row / 2) >> num_x_in_bits;
                let E_out_eval = gruen_eq.E_out_current()[x_out];

                // [并行分块 B]: 处理具体的行对 (Row Pairs), 对应内层求和
                // a.row / 2 == b.row / 2 意味着这两行属于同一对 (2k, 2k+1)。
                // 它们分别对应当前 Sumcheck 轮次变量 X=0 和 X=1 的情况。
                let outer_sum_evals = entries
                    .par_chunk_by(|a, b| a.row / 2 == b.row / 2)
                    .map(|entries| {
                        // 分离偶数行 (X=0) 和 奇数行 (X=1)
                        let odd_row_start_index = entries.partition_point(|entry| entry.row.is_even());
                        let (even_row, odd_row) = entries.split_at(odd_row_start_index);
                        let j_prime = 2 * (entries[0].row / 2);

                        // 计算内层权重 E_in(x_in)
                        // x_in 是 Cycle 索引的低位部分
                        let E_in_eval = if e_in_len <= 1 {
                            F::one()
                        } else {
                            let x_in = (j_prime / 2) & x_bitmask;
                            e_in[x_in]
                        };

                        // 获取 inc 多项式的系数 (代表 Cycle j 的数值增量)
                        // inc(j) 用于重构写入值: New_Val = Old_Val + inc
                        let inc_evals = {
                            let inc_0 = inc.get_bound_coeff(j_prime);
                            let inc_1 = inc.get_bound_coeff(j_prime + 1);
                            let inc_infty = inc_1 - inc_0;
                            [inc_0, inc_infty]
                        };

                        // [核心公式计算]
                        // 计算 G(x) 部分的贡献:
                        // Term = wa * (inc + val) + gamma * (ra * val) ...
                        // wa, ra 是由稀疏矩阵的条目隐式给出的 (存在即为1)。
                        let inner_sum_evals = ReadWriteMatrixCycleMajor::prover_message_contribution(
                            even_row,
                            odd_row,
                            inc_evals,
                            params.gamma,
                        );

                        // 累加带权重的内部结果: E_in * Inner_Constraint
                        [
                            E_in_eval.mul_unreduced::<9>(inner_sum_evals[0]),
                            E_in_eval.mul_unreduced::<9>(inner_sum_evals[1]),
                        ]
                    })
                    .reduce(
                        || [F::Unreduced::<9>::zero(); DEGREE_BOUND - 1],
                        |running, new| [running[0] + new[0], running[1] + new[1]],
                    )
                    .map(F::from_montgomery_reduce);

                // 累加外部结果: E_out * Outer_Sum
                [
                    E_out_eval.mul_unreduced::<9>(outer_sum_evals[0]),
                    E_out_eval.mul_unreduced::<9>(outer_sum_evals[1]),
                ]
            })
            .reduce(
                || [F::Unreduced::<9>::zero(); DEGREE_BOUND - 1],
                |running, new| [running[0] + new[0], running[1] + new[1]],
            )
            .map(F::from_montgomery_reduce);

        // 3. 构造单变量多项式
        // 利用 Sumcheck 性质: Eval(0) + Eval(1) = Previous_Claim
        // 结合计算出的各点评估值，插值得到 3 次多项式 (Degree 3)。
        // 为什么是 3 次? 公式的每一项涉及三个多线性多项式的乘积: Eq * wa * val
        gruen_eq.gruen_poly_deg_3(quadratic_coeffs[0], quadratic_coeffs[1], previous_claim)
    }

    /// 计算 Phase 2 (Address/Register 绑定阶段) 的 Sumcheck 消息多项式。
    ///
    /// 此时 Cycle 维度已在 Phase 1 绑定完毕。
    /// `merged_eq` 包含了绑定后的 Cycle 维度 Eq 多项式的评估值 (作为系数)。
    /// 我们现在的目标是对 **寄存器地址维度 (Address)** 进行 Sumcheck。
    /// 数据源切换为 `sparse_matrix_phase2` (按列/地址排序)。
    fn phase2_compute_message(&self, previous_claim: F) -> UniPoly<F> {
        let Self {
            inc,
            merged_eq,
            sparse_matrix_phase2,
            params,
            ..
        } = self;
        let merged_eq = merged_eq.as_ref().unwrap();

        let evals = sparse_matrix_phase2
            .entries
            // 并行分块: 按列对 (Column Pairs) 聚合
            // x.column() / 2 == y.column() / 2 表示两个条目属于同一对寄存器地址 (对应当前变量的 0/1)。
            .par_chunk_by(|x, y| x.column() / 2 == y.column() / 2)
            .map(|entries| {
                // 分离偶数列 (X=0) 和 奇数列 (X=1)
                let odd_col_start_index = entries.partition_point(|entry| entry.column().is_even());
                let (even_col, odd_col) = entries.split_at(odd_col_start_index);
                let even_col_idx = 2 * (entries[0].column() / 2);
                let odd_col_idx = even_col_idx + 1;

                // [核心公式计算]
                // 计算每个寄存器列 (Column) 的贡献。
                // 此时需要在该列包含的所有 Cycle (Rows) 上进行求和。
                //
                // 公式示意:
                // Sum_Col(k) = Σ_{row \in entries} Eq_merged(row) * Constraint(row, k)
                //
                // inc 和 merged_eq 现在都是关于 Cycle 的密集多项式数组，可以直接索引。
                // val_init 提供该寄存器的初始值 (Initial Value)。
                ReadWriteMatrixAddressMajor::prover_message_contribution(
                    even_col,
                    odd_col,
                    sparse_matrix_phase2.val_init.get_bound_coeff(even_col_idx),
                    sparse_matrix_phase2.val_init.get_bound_coeff(odd_col_idx),
                    inc,
                    merged_eq,
                    params.gamma,
                )
            })
            // 累加所有列对的贡献
            .fold_with([F::Unreduced::<5>::zero(); 2], |running, new| {
                [
                    running[0] + new[0].as_unreduced_ref(),
                    running[1] + new[1].as_unreduced_ref(),
                ]
            })
            .reduce(
                || [F::Unreduced::<5>::zero(); 2],
                |running, new| [running[0] + new[0], running[1] + new[1]],
            );

        // 使用评估值和 Hint (Previous Claim) 构造单变量多项式
        UniPoly::from_evals_and_hint(
            previous_claim,
            &[
                F::from_barrett_reduce(evals[0]),
                F::from_barrett_reduce(evals[1]),
            ],
        )
    }


    /// 计算 Phase 3 (Dense Base Phase) 的 Sumcheck 消息多项式。
    ///
    /// 在 Phase 1 和 Phase 2 完成了稀疏矩阵和部分变量的绑定后，
    /// Phase 3 处理剩余的“密集”计算部分，直到所有变量被绑定。
    ///
    /// ### 数学原理
    /// 我们需要计算当前 Sumcheck Round 的单变量多项式 $M(X)$。
    /// 目标多项式 $G$ 的形式为：
    ///
    /// $$ G(j, k) = eq_{cycle}(j) \cdot \left[ wa(k, j) \cdot (val(k, j) + inc(j)) + ra(k, j) \cdot val(k, j) \right] $$
    ///
    /// 其中:
    /// - $j$: Cycle Index (时间维度)
    /// - $k$: Register Index (地址维度)
    /// - $wa, ra$: 写/读指示器 (Write/Read Address indicators)
    /// - $val$: 寄存器值状态 (Value)
    /// - $inc$: 增量值 (Increment)
    ///
    /// 根据剩余变量的类型（是属于 Cycle 还是 Address），代码分为两个分支处理。
    fn phase3_compute_message(&self, previous_claim: F) -> UniPoly<F> {
        let Self {
            inc,
            merged_eq,
            ra,
            wa,
            val,
            params,
            ..
        } = self;
        let ra = ra.as_ref().unwrap();
        let wa = wa.as_ref().unwrap();
        let val = val.as_ref().unwrap();
        let merged_eq = merged_eq.as_ref().unwrap();

        // [Case 1]: 仍有 Cycle 变量没被绑定 (Cycle Variables Remaining)
        // 此时由于 inc 长度大于 1，说明当前的 Sumcheck 轮次正在处理 Cycle 维度的变量。
        if inc.len() > 1 {
            // Cycle variables remaining
            const DEGREE: usize = 3; // 结果多项式次数为 3 (eq * wa * val 导致)
            let K_prime = K >> params.phase2_num_rounds; // 剩余的有效寄存器数量
            let T_prime = inc.len(); // 剩余的时间周期长度
            debug_assert_eq!(ra.len(), K_prime * inc.len());

            // 并行遍历剩余的 Cycle 空间 (的一半，因为当前变量 X 取 0/1 会对折空间)
            let evals = (0..inc.len() / 2)
                .into_par_iter()
                .map(|j| {
                    // 获取 Cycle 相关多项式在 (..., j, X) 处的评估值
                    // inc(X) 和 eq(X) 仅依赖于 cycle 变量
                    let inc_evals = inc.sumcheck_evals(j, DEGREE, BindingOrder::LowToHigh);
                    let eq_evals = merged_eq.sumcheck_evals(j, DEGREE, BindingOrder::LowToHigh);

                    // 内层求和 (Inner Sum):
                    // 遍历所有寄存器 k, 累加该时刻 j 下所有寄存器的读写一致性贡献。
                    // 相当于公式中的 $\sum_k$ 部分。
                    let inner = (0..K_prime)
                        .into_par_iter()
                        .map(|k| {
                            // 计算展平后的索引。布局暗示了 [Register][Cycle] 也就是 Cycle-Major inside。
                            let idx = k * T_prime / 2 + j;

                            // 获取寄存器读写相关密集多项式 (ra, wa, val) 的评估值
                            let ra_evals = ra.sumcheck_evals(idx, DEGREE, BindingOrder::LowToHigh);
                            let wa_evals = wa.sumcheck_evals(idx, DEGREE, BindingOrder::LowToHigh);
                            let val_evals =
                                val.sumcheck_evals(idx, DEGREE, BindingOrder::LowToHigh);

                            // [核心公式]
                            // Contribution = ra * val + wa * (val + inc)
                            // 分别计算 X=0, 1, 2 插值点的值
                            [
                                ra_evals[0] * val_evals[0]
                                    + wa_evals[0] * (val_evals[0] + inc_evals[0]),
                                ra_evals[1] * val_evals[1]
                                    + wa_evals[1] * (val_evals[1] + inc_evals[1]),
                                ra_evals[2] * val_evals[2]
                                    + wa_evals[2] * (val_evals[2] + inc_evals[2]),
                            ]
                        })
                        // 使用折叠归约 (Fold-Reduce) 模式并行聚合结果
                        .fold_with([F::Unreduced::<5>::zero(); DEGREE], |running, new| {
                            [
                                running[0] + new[0].as_unreduced_ref(),
                                running[1] + new[1].as_unreduced_ref(),
                                running[2] + new[2].as_unreduced_ref(),
                            ]
                        })
                        .reduce(
                            || [F::Unreduced::<5>::zero(); DEGREE],
                            |running, new| {
                                [
                                    running[0] + new[0],
                                    running[1] + new[1],
                                    running[2] + new[2],
                                ]
                            },
                        );

                    // 最后乘以外层的 Eq 系数
                    // Total(X) = Eq(j, X) * Inner_Sum(X)
                    [
                        eq_evals[0] * F::from_barrett_reduce(inner[0]),
                        eq_evals[1] * F::from_barrett_reduce(inner[1]),
                        eq_evals[2] * F::from_barrett_reduce(inner[2]),
                    ]
                })
                .fold_with([F::Unreduced::<5>::zero(); DEGREE], |running, new| {
                    [
                        running[0] + new[0].as_unreduced_ref(),
                        running[1] + new[1].as_unreduced_ref(),
                        running[2] + new[2].as_unreduced_ref(),
                    ]
                })
                .reduce(
                    || [F::Unreduced::<5>::zero(); DEGREE],
                    |running, new| {
                        [
                            running[0] + new[0],
                            running[1] + new[1],
                            running[2] + new[2],
                        ]
                    },
                );

            // 从计算出的评估值 [P(0), P(1), P(2)] 构造单变量多项式
            UniPoly::from_evals_and_hint(
                previous_claim,
                &[
                    F::from_barrett_reduce(evals[0]),
                    F::from_barrett_reduce(evals[1]),
                    F::from_barrett_reduce(evals[2]),
                ],
            )
        } else {
            // [Case 2]: Cycle 变量已完全绑定 (Cycle variables are fully bound)
            // 此时 inc 和 eq 都变成了常数 (Scalar)。
            // 我们现在正在对 Address (Register) 维度的变量进行 Sumcheck。
            const DEGREE: usize = 2; // eq 为常数，剩下的 wa*val 是 2 次
            // Cycle variables are fully bound
            let inc_eval = inc.final_sumcheck_claim();
            let eq_eval = merged_eq.final_sumcheck_claim();

            // 遍历剩余的 Address 空间
            let evals = (0..ra.len() / 2)
                .into_par_iter()
                .map(|k| {
                    // 获取当前 Register 相关的 ra, wa, val 评估值
                    let ra_evals = ra.sumcheck_evals_array::<DEGREE>(k, BindingOrder::LowToHigh);
                    let wa_evals = wa.sumcheck_evals_array::<DEGREE>(k, BindingOrder::LowToHigh);
                    let val_evals = val.sumcheck_evals_array::<DEGREE>(k, BindingOrder::LowToHigh);

                    // [核心公式]
                    // Term = ra * val + wa * (val + inc_constant)
                    [
                        ra_evals[0] * val_evals[0] + wa_evals[0] * (val_evals[0] + inc_eval),
                        ra_evals[1] * val_evals[1] + wa_evals[1] * (val_evals[1] + inc_eval),
                    ]
                })
                .fold_with([F::Unreduced::<5>::zero(); DEGREE], |running, new| {
                    [
                        running[0] + new[0].as_unreduced_ref(),
                        running[1] + new[1].as_unreduced_ref(),
                    ]
                })
                .reduce(
                    || [F::Unreduced::<5>::zero(); DEGREE],
                    |running, new| [running[0] + new[0], running[1] + new[1]],
                );

            // 最终结果需要乘上全局 Eq 常数 (因为 Eq 在 Register 维度上没有变化)
            UniPoly::from_evals_and_hint(
                previous_claim,
                &[
                    eq_eval * F::from_barrett_reduce(evals[0]),
                    eq_eval * F::from_barrett_reduce(evals[1]),
                ],
            )
        }
    }

    /// 执行 Sumcheck Phase 1 的变量绑定 (Binding)。
    ///
    /// 当 Verifier 发送挑战点 $r_j$ 时，Prover 需要将多项式的变量 $x_j$ 固定为 $r_j$。
    /// 这将把 $n$ 元多项式降阶为 $n-1$ 元多项式。
    fn phase1_bind(&mut self, r_j: F::Challenge, round: usize) {
        let Self {
            sparse_matrix_phase1: sparse_matrix,
            inc,
            gruen_eq,
            params,
            ..
        } = self;
        let gruen_eq = gruen_eq.as_mut().unwrap();

        // 1. 绑定稀疏矩阵 (逻辑上折叠 Cycle 维度)
        sparse_matrix.bind(r_j);
        // 2. 绑定 Gruen Eq 多项式 (Split Eq 优化部分)
        gruen_eq.bind(r_j);
        // 3. 绑定 Inc 多项式 (标准 MLE 绑定: $P(r) = P(0) + r(P(1) - P(0))$)
        inc.bind_parallel(r_j, BindingOrder::LowToHigh);

        // [阶段转换逻辑]
        // 如果 Phase 1 结束 (Cycle 维度部分绑定完成):
        if round == params.phase1_num_rounds - 1 {
            // 将优化的 Gruen Eq 合并为普通密集多项式
            self.merged_eq = Some(MultilinearPolynomial::LargeScalars(gruen_eq.merge()));

            // 拿走稀疏矩阵的所有权进行转换
            let sparse_matrix = std::mem::take(sparse_matrix);
            if params.phase2_num_rounds > 0 {
                // 如果配置了 Phase 2，将矩阵转入 Phase 2 结构
                self.sparse_matrix_phase2 = sparse_matrix.into();
            } else {
                // 如果跳过 Phase 2，直接物化 (Materialize) 密集的 ra, wa, val 多项式
                // 此时所有 Cycle 变量已处理，准备直接进入 Phase 3 处理地址变量
                let T_prime = params.T >> params.phase1_num_rounds;
                let [ra, wa, val] = sparse_matrix.materialize(K, T_prime);
                self.ra = Some(ra);
                self.wa = Some(wa);
                self.val = Some(val);
            }
        }
    }

    /// 执行 Sumcheck Phase 2 的变量绑定。
    ///
    /// 这里的 $r_j$ 对应 Address/Register 维度的变量。
    fn phase2_bind(&mut self, r_j: F::Challenge, round: usize) {
        let Self {
            params,
            sparse_matrix_phase2: sparse_matrix,
            ..
        } = self;

        // 绑定稀疏矩阵 (逻辑上折叠 Address 维度)
        sparse_matrix.bind(r_j);

        // [阶段转换逻辑]
        // 如果 Phase 2 结束:
        if round == params.phase1_num_rounds + params.phase2_num_rounds - 1 {
            let sparse_matrix = std::mem::take(sparse_matrix);
            // 将稀疏矩阵完全转换为密集多线性多项式 (MLE)
            // 此时所有稀疏优化阶段已结束，进入最终的 Phase 3 密集计算
            // K_prime: 剩余的寄存器空间
            // T_prime: 剩余的周期空间 (由 Phase 1 决定)
            let [ra, wa, val] = sparse_matrix.materialize(
                K >> params.phase2_num_rounds,
                params.T >> params.phase1_num_rounds,
            );
            self.ra = Some(ra);
            self.wa = Some(wa);
            self.val = Some(val);
        }
    }

    /// 执行 Sumcheck Phase 3 的变量绑定。
    ///
    /// 此时我们已经拥有了密集 MLE 多项式 (ra, wa, val, inc, merged_eq)，直接进行标准绑定。
    fn phase3_bind(&mut self, r_j: F::Challenge) {
        let Self {
            ra,
            wa,
            val,
            inc,
            merged_eq,
            ..
        } = self;
        let ra = ra.as_mut().unwrap();
        let wa = wa.as_mut().unwrap();
        let val = val.as_mut().unwrap();
        let merged_eq = merged_eq.as_mut().unwrap();

        // 绑定 ra, wa, val。
        // 注意：ra/wa/val 是联合分布 (Cycle x Address)，无论绑定哪种变量都会折叠。
        [ra, wa, val]
            .into_par_iter()
            .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::LowToHigh));

        // 只有当还有 Cycle 变量未绑定时 (inc.len() > 1)，才需要绑定 inc 和 merged_eq。
        // 因为 inc 和 merged_eq 仅仅依赖于 Cycle 维度。
        // 如果我们正在绑定 Address 变量，且 Cycle 变量早就在 Phase 1 绑完了，那么它们早就是常数了。
        if inc.len() > 1 {
            // Cycle variables remaining
            inc.bind_parallel(r_j, BindingOrder::LowToHigh);
            merged_eq.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
    }

    /// Compute rs2_ra(r_address, r_cycle) = Σ_j [has_rs2[j]] * eq(r_address, rs2[j]) * eq(r_cycle, j)
    ///
    /// We compute rs2 (not rs1) because fewer cycles have rs2 reads:
    /// - rs2 is NOT read by: FormatI (ADDI, etc.), FormatLoad (LB, LW, etc.), FormatU, FormatJ
    /// - rs1 is NOT read by: only FormatU, FormatJ
    ///
    /// Uses a 2-way split-eq optimization over the joint (cycle, address) space:
    /// - Order: r_joint = [r_cycle..., r_address...] so cycle vars are MSB
    /// - Total bits: n = log_T + 7 (address)
    /// - hi_bits = min(log_T, (n+1)/2) ensures hi part contains only cycle bits
    /// - This enables clean double outer/inner sum: outer over cycle blocks, inner sums E_lo
    ///
    /// EqPolynomial bit ordering: bit i of index → r[n-1-i] (reverse order, r[0] is MSB)
    /// - For r_joint = [r_cycle, r_address]:
    ///   - bits 0..(addr_bits-1) of joint_index → r_address (LSB part)
    ///   - bits addr_bits..(n-1) of joint_index → r_cycle (MSB part)
    /// - So joint_index = (j << addr_bits) | rs2
    /// 计算 rs2 读取操作的 Claim 值。
    ///
    /// ### 目标公式
    /// 我们需要计算多项式 $Q(r_{cycle}, r_{address})$ 的值：
    /// $$ Q(r_{cycle}, r_{address}) = \sum_{j=0}^{T-1} \mathbb{I}(has\_rs2_j) \cdot eq(r_{cycle}, j) \cdot eq(r_{address}, rs2_j) $$
    ///
    /// 其中：
    /// - $\mathbb{I}(has\_rs2_j)$：如果第 $j$ 个 cycle 读取了 rs2，则为 1，否则为 0。
    /// - $rs2_j$：第 $j$ 个 cycle 读取的寄存器索引。
    ///
    /// ### 优化算法：2-way Split-Eq (双路分裂求和)
    /// 直接计算上述公式需要 $O(T)$ 次椭圆曲线乘法。为了加速，我们利用 $eq$ 函数的张量积性质：
    /// $$ eq(r, x) = eq(r_{hi}, x_{hi}) \cdot eq(r_{lo}, x_{lo}) $$
    ///
    /// 我们将联合空间 $(Cycle \times Address)$ 视为一个大向量，将其切分为高位 ($hi$) 和低位 ($lo$)。
    /// - **Outer Sum (Parallel)**: 遍历 $E_{hi}$ (代表时间块 Block)。
    /// - **Inner Sum (Sequential)**: 在每个块内遍历具体的 Cycle。
    #[tracing::instrument(
        skip_all,
        name = "RegistersReadWriteCheckingProver::compute_rs2_ra_claim"
    )]
    fn compute_rs2_ra_claim(
        trace: &[Cycle],
        r_address: &[F::Challenge], // 寄存器地址的随机挑战点 (低位)
        r_cycle: &[F::Challenge],   // 时间周期的随机挑战点 (高位)
    ) -> F {
        let log_T = r_cycle.len();
        let addr_bits = r_address.len(); // 例如 128 个寄存器对应 7 bits

        // [联合空间定义]
        // 我们在一个联合的虚拟空间上进行评估，总维度为 n。
        // 变量顺序 (大端序): r_joint = [r_cycle..., r_address...]
        // 这意味着 Cycle 变量位于高位 (MSB)，Address 变量位于低位 (LSB)。
        let n = log_T + addr_bits;

        // [切分策略]
        // 将联合空间切分为 Hi 和 Lo 两部分。
        // 关键约束: hi_bits 必须只包含 Cycle 变量，不能包含 Address 变量。
        // 原因：为了让内层循环能够按顺序遍历 Trace 中的 Cycle，Address 变量必须完全包含在 Lo 部分。
        let hi_bits = std::cmp::min(log_T, n.div_ceil(2));
        let lo_bits = n - hi_bits;

        // 拼接随机点: r_joint = [r_cycle || r_address]
        let r_joint: Vec<F::Challenge> = r_cycle.iter().chain(r_address.iter()).copied().collect();
        let (r_hi, r_lo) = r_joint.split_at(hi_bits);

        // [预计算 Eq 表]
        // E_hi 大小为 2^hi_bits，针对 Cycle 的高位部分。
        // E_lo 大小为 2^lo_bits，针对 Cycle 的低位部分 + 所有 Address 位。
        let (E_hi, E_lo) = rayon::join(
            || EqPolynomial::<F>::evals(r_hi),
            || EqPolynomial::<F>::evals(r_lo),
        );

        // [位操作掩码计算]
        // 联合索引公式: joint_index = (cycle << addr_bits) | rs2
        //
        // 分解为 Hi/Lo 索引:
        // idx_hi = joint_index >> lo_bits = cycle >> (lo_bits - addr_bits)
        // idx_lo = joint_index & lo_mask
        let cycle_bits_in_lo = lo_bits - addr_bits; // Lo 部分包含的时间位数
        let cycles_per_block = 1usize << cycle_bits_in_lo; // 每个 Hi 索引对应的 Block 大小 (包含多少个 cycle)
        let cycle_lo_mask = cycles_per_block - 1;

        // [双层求和执行]
        // Formula: Sum = Σ_{block} E_hi[block] * ( Σ_{j ∈ block} E_lo[local_idx(j)] )
        (0..E_hi.len())
            .into_par_iter()
            .map(|idx_hi| {
                let e_hi_val = E_hi[idx_hi];
                // 计算当前 Block 在 Trace 中的起始和结束位置
                let block_start = idx_hi << cycle_bits_in_lo;
                let block_end = std::cmp::min(block_start + cycles_per_block, trace.len());

                // 边界检查：防止索引越界
                if block_start >= trace.len() {
                    return F::zero();
                }

                // Inner Sum: 在当前 Block 内遍历 Cycle
                // `filter_map` 自动处理了稀疏性：只有存在 rs2 读取的指令才会贡献值。
                let inner_sum: F = (block_start..block_end)
                    .filter_map(|j| {
                        trace[j].rs2_read().map(|(rs2, _)| {
                            // [Lo 索引映射公式]
                            // 我们需要构造联合索引的低位部分。
                            // idx_lo = (Cycle_Low_Bits || Register_Bits)
                            // 1. 获取当前 cycle 在 block 内的偏移: j_in_block = j & cycle_lo_mask
                            // 2. 左移腾出寄存器位: j_in_block << addr_bits
                            // 3. 填入寄存器地址: | rs2
                            let j_in_block = j & cycle_lo_mask;
                            let idx_lo = (j_in_block << addr_bits) | (rs2 as usize);
                            E_lo[idx_lo]
                        })
                    })
                    .sum();

                // 累加项: hi_part * sum(lo_part)
                e_hi_val * inner_sum
            })
            .sum()
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T>
for RegistersReadWriteCheckingProver<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(skip_all, name = "RegistersReadWriteCheckingProver::compute_message")]
    fn compute_message(&mut self, round: usize, previous_claim: F) -> UniPoly<F> {
        if round < self.params.phase1_num_rounds {
            self.phase1_compute_message(previous_claim)
        } else if round < self.params.phase1_num_rounds + self.params.phase2_num_rounds {
            self.phase2_compute_message(previous_claim)
        } else {
            self.phase3_compute_message(previous_claim)
        }
    }

    #[tracing::instrument(skip_all, name = "RegistersReadWriteCheckingProver::ingest_challenge")]
    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        if round < self.params.phase1_num_rounds {
            self.phase1_bind(r_j, round);
        } else if round < self.params.phase1_num_rounds + self.params.phase2_num_rounds {
            self.phase2_bind(r_j, round);
        } else {
            self.phase3_bind(r_j);
        }
    }

    #[tracing::instrument(skip_all, name = "RegistersReadWriteCheckingProver::cache_openings")]
    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[<F as JoltField>::Challenge],
    ) {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        let (r_address, r_cycle) = opening_point.split_at(LOG_K);

        let val_claim = self.val.as_ref().unwrap().final_sumcheck_claim();
        let rd_wa_claim = self.wa.as_ref().unwrap().final_sumcheck_claim();
        let inc_claim = self.inc.final_sumcheck_claim();
        let combined_ra_claim = self.ra.as_ref().unwrap().final_sumcheck_claim();
        // In order to obtain the individual claims rs1_ra(r) and rs2_ra(r),
        // we compute rs2_ra(r) directly (fewer cycles have rs2 reads than rs1 reads):
        let rs2_ra_claim: F = Self::compute_rs2_ra_claim(&self.trace, &r_address.r, &r_cycle.r);

        // Now compute rs1_ra(r) from combined_ra_claim and rs2_ra_claim. Recall that:
        // combined_ra_claim = gamma * rs1_ra(r) + gamma^2 * rs2_ra(r)
        // => rs1_ra(r) = (combined_ra_claim - gamma^2 * rs2_ra(r)) / gamma
        let gamma = self.params.gamma;
        let gamma_inverse = gamma.inverse().unwrap();
        let rs1_ra_claim = (combined_ra_claim - gamma * gamma * rs2_ra_claim) * gamma_inverse;

        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RegistersVal,
            SumcheckId::RegistersReadWriteChecking,
            opening_point.clone(),
            val_claim,
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::Rs1Ra,
            SumcheckId::RegistersReadWriteChecking,
            opening_point.clone(),
            rs1_ra_claim,
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::Rs2Ra,
            SumcheckId::RegistersReadWriteChecking,
            opening_point.clone(),
            rs2_ra_claim,
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RdWa,
            SumcheckId::RegistersReadWriteChecking,
            opening_point.clone(),
            rd_wa_claim,
        );

        accumulator.append_dense(
            transcript,
            CommittedPolynomial::RdInc,
            SumcheckId::RegistersReadWriteChecking,
            r_cycle.r,
            inc_claim,
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

/// A sumcheck instance for:
///
/// ```text
/// sum_j eq(r_cycle, j) * (RdWriteValue(x) + gamma * Rs1Value(j) + gamma^2 * Rs2Value(j))
/// ```
///
/// Where
///
/// ```text
/// RdWriteValue(x) = RdWa(x) * (Inc(x) + Val(x))
/// Rs1Value(x) = Rs1Ra(x) * Val(x)
/// Rs2Value(x) = Rs2Ra(x) * Val(x)
/// ```
pub struct RegistersReadWriteCheckingVerifier<F: JoltField> {
    params: RegistersReadWriteCheckingParams<F>,
}

impl<F: JoltField> RegistersReadWriteCheckingVerifier<F> {
    pub fn new(
        trace_len: usize,
        opening_accumulator: &VerifierOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
        config: &ReadWriteConfig,
    ) -> Self {
        let params = RegistersReadWriteCheckingParams::new(
            trace_len,
            opening_accumulator,
            transcript,
            config,
        );
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
for RegistersReadWriteCheckingVerifier<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let r = self.params.normalize_opening_point(sumcheck_challenges);
        let (_, r_cycle) = r.split_at(LOG_K);

        let (_, val_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RegistersVal,
            SumcheckId::RegistersReadWriteChecking,
        );
        let (_, rs1_ra_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::Rs1Ra,
            SumcheckId::RegistersReadWriteChecking,
        );
        let (_, rs2_ra_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::Rs2Ra,
            SumcheckId::RegistersReadWriteChecking,
        );
        let (_, rd_wa_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RdWa,
            SumcheckId::RegistersReadWriteChecking,
        );
        let (_, inc_claim) = accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::RdInc,
            SumcheckId::RegistersReadWriteChecking,
        );

        let rd_write_value_claim = rd_wa_claim * (inc_claim + val_claim);
        let rs1_value_claim = rs1_ra_claim * val_claim;
        let rs2_value_claim = rs2_ra_claim * val_claim;

        EqPolynomial::mle_endian(&r_cycle, &self.params.r_cycle)
            * (rd_write_value_claim
            + self.params.gamma * (rs1_value_claim + self.params.gamma * rs2_value_claim))
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[<F as JoltField>::Challenge],
    ) {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RegistersVal,
            SumcheckId::RegistersReadWriteChecking,
            opening_point.clone(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::Rs1Ra,
            SumcheckId::RegistersReadWriteChecking,
            opening_point.clone(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::Rs2Ra,
            SumcheckId::RegistersReadWriteChecking,
            opening_point.clone(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RdWa,
            SumcheckId::RegistersReadWriteChecking,
            opening_point.clone(),
        );

        let (_, r_cycle) = opening_point.split_at(LOG_K);
        accumulator.append_dense(
            transcript,
            CommittedPolynomial::RdInc,
            SumcheckId::RegistersReadWriteChecking,
            r_cycle.r,
        );
    }
}
