use common::jolt_device::MemoryLayout;
use num::Integer;
use num_traits::Zero;

use crate::poly::multilinear_polynomial::PolynomialEvaluation;
use crate::poly::opening_proof::OpeningAccumulator;
use crate::poly::split_eq_poly::GruenSplitEqPolynomial;

use crate::poly::unipoly::UniPoly;
use crate::subprotocols::read_write_matrix::{
    AddressMajorMatrixEntry, RamAddressMajorEntry, RamCycleMajorEntry, ReadWriteMatrixAddressMajor,
    ReadWriteMatrixCycleMajor,
};
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use crate::subprotocols::sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier};
use crate::zkvm::bytecode::BytecodePreprocessing;
use crate::zkvm::config::{OneHotParams, ReadWriteConfig};
use crate::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{
            OpeningPoint, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator,
            BIG_ENDIAN,
        },
    },
    transcripts::Transcript,
    utils::math::Math,
    zkvm::witness::{CommittedPolynomial, VirtualPolynomial},
};
use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use rayon::prelude::*;
use tracer::instruction::Cycle;

// RAM read-write checking sumcheck
//
// Proves the relation:
//   Σ_{k,j} eq(r_cycle, j) ⋅ ra(k, j) ⋅ (Val(k, j) + γ ⋅ (inc(j) + Val(k, j)))
//   = rv_claim + γ ⋅ wv_claim
// where:
// - r_cycle are the challenges for the cycle variables in this sumcheck (from Spartan outer)
// - ra(k, j) = 1 if memory address k is accessed at cycle j, and 0 otherwise
// - Val(k, j) is the value at memory address k right before cycle j
// - inc(j) is the change in value at cycle j if a write occurs, and 0 otherwise
// - rv_claim and wv_claim are the claimed read and write values from the Spartan outer sumcheck.
//
// This sumcheck ensures that the values read from and written to RAM are consistent
// with the memory trace and the initial/final memory states.

/// Degree bound of the sumcheck round polynomials in [`RamReadWriteCheckingVerifier`].
const DEGREE_BOUND: usize = 3;

#[derive(Allocative, Clone)]
pub struct RamReadWriteCheckingParams<F: JoltField> {
    pub K: usize,
    pub T: usize,
    pub gamma: F,
    pub r_cycle: OpeningPoint<BIG_ENDIAN, F>,
    /// Number of cycle variables to bind in phase 1.
    pub phase1_num_rounds: usize,
    /// Number of address variables to bind in phase 2.
    pub phase2_num_rounds: usize,
}

impl<F: JoltField> RamReadWriteCheckingParams<F> {
    /// 初始化 `RamReadWriteCheckingParams` 结构体。
    ///
    /// # 作用
    /// 准备 RAM 读写一致性检查（RAM Read-Write Consistency Check）Sumcheck 协议所需的参数。
    /// 此协议用于证明内存的读取值和写入值与内存的历史状态变化（Trace）是一致的。
    ///
    /// # 核心逻辑
    /// 1. **生成挑战因子 $\gamma$**: 从 Transcript 中采样。该因子用于在验证等式中线性组合初始值和更新后的值：
    ///    $Val + \gamma \cdot (Inc + Val)$。这本质上是在进行一种类似于排列检查（Permutation Check）或多重集合一致性的随机线性组合。
    /// 2. **继承挑战点 $r_{cycle}$**: 从 Spartan Outer Sumcheck 获取周期变量的挑战点。
    ///    这是为了将当前的 RAM 检查协议“锚定”到外部的主证明协议上。
    ///
    /// # 参数
    /// * `opening_accumulator`: 用于获取上一阶段产生的 Challenges。
    /// * `transcript`: Fiat-Shamir Transcript，用于生成新的随机挑战。
    /// * `one_hot_params`: 包含 RAM 大小 $K$ 等系统参数。
    /// * `trace_length`: 执行轨迹的长度 ($T$)。
    /// * `config`: 包含关于求和阶段（Phase 1/2）具体轮数的配置。
    pub fn new(
        // 1. Opening Accumulator (上一阶段的遗产)
        // 包含 Stage 1 (CPU 执行) 结束后的所有多项式评估值和挑战点。
        opening_accumulator: &dyn OpeningAccumulator<F>,
        // 2. Transcript (随机数生成源)
        // 用于生成 Fiat-Shamir 挑战，确保交互的安全性。
        transcript: &mut impl Transcript,
        // 3. One-Hot Params (系统参数)
        // 包含内存大小 K 等配置信息。
        one_hot_params: &OneHotParams,
        // 4. Trace Length (执行轨迹长度 T)
        // CPU 执行的总步数（指令数），通常对应时间戳 (Timestamp)。
        trace_length: usize,
        // 5. Config (读写检查配置)
        // 定义了 Grand Product 协议的轮数分配 (Phase 1 / Phase 2)。
        config: &ReadWriteConfig,
    ) -> Self {
        // =================================================================
        // 步骤 A: 生成随机挑战 Gamma (γ)
        // =================================================================
        // 这是一个至关重要的随机数，用于 Random Linear Combination (RLC)。
        //
        // 背景：
        // 内存检查需要验证元组 (Address, Value, Timestamp) 的集合一致性。
        // 为了将这三个数压缩成一个数进行连乘 (Grand Product)，我们需要一个随机权重 γ。
        //
        // 压缩公式通常为：
        // Fingerprint = Address + γ * Value + γ^2 * Timestamp
        //
        // 只有当两个集合的连乘积相等时，才能证明内存读写是一致的。
        let gamma = transcript.challenge_scalar();

        // =================================================================
        // 步骤 B: 获取 Stage 1 的绑定点 r_cycle
        // =================================================================
        // 这里的 get_virtual_polynomial_opening 并不是为了获取“值”，
        // 而是为了获取 CPU 阶段结束时锁定的那个随机点坐标 `r_cycle`。
        //
        // 为什么需要它？
        // 1. Stage 1 (SpartanOuter) 证明了指令的执行，生成了关于 RamReadValue 的多项式承诺。
        // 2. Stage 2 (本阶段) 要证明这些 ReadValue 是合法的（符合读写逻辑）。
        // 3. 必须确保 Stage 2 检查的多项式，和 Stage 1 使用的多项式是“同一个”。
        // 4. 通过共享同一个评估点 `r_cycle`，我们将两个独立的 Sum-check 协议“胶合”在了一起。
        //
        // VirtualPolynomial::RamReadValue: 指明我们要关联的是“内存读数值”这一列。
        let (r_cycle, _) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamReadValue,
            SumcheckId::SpartanOuter,
        );

        // =================================================================
        // 步骤 C: 提取系统规模参数
        // =================================================================

        // K: 内存操作的总容量 (RAM Size / Operations)。
        // 这通常是按地址排序后的 Trace 长度 (Sorted Trace)。
        let K = one_hot_params.ram_k;

        // T: CPU 执行的时间步总数 (Time Steps)。
        // 这是按时间顺序的 Trace 长度 (Execution Trace)。
        // 在 Jolt 中，通常 K >= T，因为可能会有 Padding。
        let T = trace_length;

        // =================================================================
        // 步骤 D: 构造参数结构体
        // =================================================================
        RamReadWriteCheckingParams {
            K,
            T,
            gamma,// 用于 RLC 压缩指纹
            r_cycle,// 用于与 CPU 阶段绑定
            // 从配置中加载 Phase 1 (Cycle 也就是时间维度) 和 Phase 2 (Address 也就是空间维度) 的绑定轮数。
            // 这通常是为了优化 Proof 的生成性能，允许部分变量先绑定，分阶段处理。
            phase1_num_rounds: config.ram_rw_phase1_num_rounds as usize,//9
            phase2_num_rounds: config.ram_rw_phase2_num_rounds as usize,//13
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for RamReadWriteCheckingParams<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.K.log_2() + self.T.log_2()
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let (_, rv_input_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamReadValue,
            SumcheckId::SpartanOuter,
        );
        let (_, wv_input_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamWriteValue,
            SumcheckId::SpartanOuter,
        );
        rv_input_claim + self.gamma * wv_input_claim
    }

    /// 将 Sumcheck 过程中产生的原始挑战点序列重组为规范的评估点。
    ///
    /// # 作用
    /// Sumcheck 协议是按特定顺序（Phase 1 -> Phase 2 -> Phase 3）逐轮产生挑战点 $r_i$ 的。
    /// 为了优化 Prover 的性能，变量绑定的顺序（Binding Order）通常是混合且经过优化的（例如先绑定低位）。
    /// 此函数负责将这些按时间顺序产生的随机数，重新排列组合成标准的 Big-Endian 坐标 $(r_{address} || r_{cycle})$。
    ///
    /// # 变量绑定流程
    /// 这里的 Sumcheck 分为三个阶段，但在逻辑上是连续的：
    /// 1. **Phase 1**: 绑定 Cycle (时间) 变量的低位部分 (Low bits)。
    /// 2. **Phase 2**: 绑定 Address (空间) 变量的低位部分 (Low bits)。
    /// 3. **Phase 3**: 先绑定剩余的 Cycle 变量 (High bits)，再绑定剩余的 Address 变量 (High bits)。
    ///
    /// # 逻辑步骤
    /// 1. 根据 phases 的轮数，将线性的 `sumcheck_challenges` 切片分割。
    /// 2. 由于所有绑定操作都是 `LowToHigh` (从 LSB 到 MSB)，为了得到 Big-Endian (从 MSB 到 LSB, 即 $x_0, x_1...$)，
    ///    需要对每一部分的挑战点进行 `.rev()` 反转。
    /// 3. 组合：
    ///    * $r_{cycle}$ = Phase 3 (高位) + Phase 1 (低位)。
    ///    * $r_{address}$ = Phase 3 (高位) + Phase 2 (低位)。
    /// 4. 最终返回拼接后的点 $[r_{address}, r_{cycle}]$。
    fn normalize_opening_point(
        &self,
        sumcheck_challenges: &[F::Challenge], // 原始的、按轮次顺序生成的挑战点列表
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        // 1. 切分 Phase 1 的挑战点 (Cycle 低位)
        // Cycle variables are bound low-to-high in phase 1
        let (phase1_challenges, sumcheck_challenges) =
            sumcheck_challenges.split_at(self.phase1_num_rounds);

        // 2. 切分 Phase 2 的挑战点 (Address 低位)
        // Address variables are bound low-to-high in phase 2
        let (phase2_challenges, sumcheck_challenges) =
            sumcheck_challenges.split_at(self.phase2_num_rounds);

        // 3. 切分 Phase 3 的挑战点
        // Phase 3 剩下的部分先是 Cycle 高位，然后是 Address 高位。
        // Calculate remaining cycle rounds: Total Cycle Rounds - Phase 1 Rounds
        let (phase3_cycle_challenges, phase3_address_challenges) =
            sumcheck_challenges.split_at(self.T.log_2() - self.phase1_num_rounds);

        // 4. 重组 r_cycle (周期/时间坐标)
        // 结构要求：Big-Endian [MSB ... LSB]
        // Phase 3 包含 Cycle 高位 (MSB)，Phase 1 包含 Cycle 低位 (LSB)。
        // 且因为绑定顺序是 LowToHigh (Last variable first)，所以每段内部都需要 reverse。
        // Result order: [Phase3_rev (High), Phase1_rev (Low)]
        let r_cycle: Vec<_> = phase3_cycle_challenges
            .iter()
            .rev()
            .copied()
            .chain(phase1_challenges.iter().rev().copied())
            .collect();

        // 5. 重组 r_address (内存地址坐标)
        // 结构要求：Big-Endian [MSB ... LSB]
        // Phase 3 剩余部分包含 Address 高位，Phase 2 包含 Address 低位。
        // Result order: [Phase3_rev (High), Phase2_rev (Low)]
        let r_address: Vec<_> = phase3_address_challenges
            .iter()
            .rev()
            .copied()
            .chain(phase2_challenges.iter().rev().copied())
            .collect();

        // 6. 拼接最终结果并转换为 OpeningPoint
        // 最终多项式的变量顺序通常是 Address 也就是 Memory 部分在前，Cycle 在后。
        // 此处返回 [r_address || r_cycle]。
        [r_address, r_cycle].concat().into()
    }
}

#[derive(Allocative)]
pub struct RamReadWriteCheckingProver<F: JoltField> {
    sparse_matrix_phase1: ReadWriteMatrixCycleMajor<F, RamCycleMajorEntry<F>>,
    sparse_matrix_phase2: ReadWriteMatrixAddressMajor<F, RamAddressMajorEntry<F>>,
    gruen_eq: Option<GruenSplitEqPolynomial<F>>,
    inc: MultilinearPolynomial<F>,
    // The following polynomials are instantiated after
    // the second phase
    ra: Option<MultilinearPolynomial<F>>,
    val: Option<MultilinearPolynomial<F>>,
    merged_eq: Option<MultilinearPolynomial<F>>,
    pub params: RamReadWriteCheckingParams<F>,
}

impl<F: JoltField> RamReadWriteCheckingProver<F> {
    /// 初始化 `RamReadWriteCheckingProver`。
    ///
    /// # 作用
    /// 构建 RAM 读写一致性检查的证明者实例。此过程主要涉及将执行轨迹（Trace）转换为
    /// 适合 Sumcheck 协议处理的多项式和稀疏矩阵形式。
    ///
    /// # 关键步骤
    /// 1. **初始化 Eq 多项式**:
    ///    根据是否进行分阶段绑定（Phase 1），决定是使用优化的 Gruen 分裂 Eq 多项式，
    ///    还是直接计算完整的 Eq 多项式评估值。这里的 Eq 针对的是 $r_{cycle}$。
    /// 2. **生成 Inc 多项式**:
    ///    调用 `CommittedPolynomial::RamInc` 生成 `inc` 列的 Witness 数据。
    ///    `inc` 代表如果发生了写入操作，内存值的变化量。
    /// 3. **构建读写稀疏矩阵**:
    ///    将 Trace 数据转换为 `ReadWriteMatrix`。这是一个稀疏表示，记录了哪些 (Address, Cycle)
    ///    对发生了交互以及相应的值。
    /// 4. **配置矩阵状态**:
    ///    根据 `phase1_num_rounds` 的配置，决定初始持有的矩阵形态：
    ///    * 如果有 Phase 1 (Cycle 绑定阶段)，保持 `CycleMajor` 格式。
    ///    * 如果跳过 Phase 1 直接进入 Phase 2，则转换为 `AddressMajor` 格式。
    #[tracing::instrument(skip_all, name = "RamReadWriteCheckingProver::initialize")]
    pub fn initialize(
        params: RamReadWriteCheckingParams<F>,
        trace: &[Cycle],
        bytecode_preprocessing: &BytecodePreprocessing,
        memory_layout: &MemoryLayout,
        initial_ram_state: &[u64],
    ) -> Self {
        let r_prime = &params.r_cycle;

        // 1. 准备 Eq 多项式 (用于 Sumcheck 中的 eq(r_cycle, j) 项)
        // 如果 Phase 1 轮数大于 0，我们使用 Gruen 的优化算法（Split Eq），允许逐轮绑定变量。
        // 否则，如果直接从后续阶段开始，或没有 Phase 1，我们可能直接计算出完整的 Eq 表。
        let (gruen_eq, merged_eq) = if params.phase1_num_rounds > 0 {
            (
                Some(GruenSplitEqPolynomial::new(
                    &r_prime.r,
                    BindingOrder::LowToHigh,
                )),
                None, // 尚未合并，处于分裂状态
            )
        } else {
            (
                None,
                Some(MultilinearPolynomial::from(EqPolynomial::evals(&r_prime.r))),
            )
        };

        // 2. 生成 Inc (Increment) 多项式 Witness
        // Inc 表示在特定周期写入发生时值的增量。
        let inc = CommittedPolynomial::RamInc.generate_witness(
            bytecode_preprocessing,
            memory_layout,
            trace,
            None,
        );

        // 3. 处理初始 RAM 状态
        // 将 u64 类型的初始内存值转换为有限域元素。
        let val_init: Vec<_> = initial_ram_state
            .par_iter()
            .map(|x| F::from_u64(*x))
            .collect();

        // 4. 构建读写稀疏矩阵 (Cycle-Major)
        // 这是证明的核心数据结构，包含了所有的内存访问记录。
        // 初始构建时通常按照时间顺序 (Cycle-Major)。
        let sparse_matrix = ReadWriteMatrixCycleMajor::<_, RamCycleMajorEntry<F>>::new(
            trace,
            val_init,
            memory_layout,
        );

        let phase1_rounds = params.phase1_num_rounds;
        let phase2_rounds = params.phase2_num_rounds;

        // 5. 根据阶段配置设置矩阵存储
        // Jolt 的 Sumcheck 分为不同阶段绑定不同类型的变量（先时间后空间，或反之）。
        // 不同的绑定顺序对矩阵的存储布局（行优先/列优先）有不同要求以优化性能。
        let (sparse_matrix_phase1, sparse_matrix_phase2, ra, val) = if phase1_rounds > 0 {
            // 如果有 Phase 1 (绑定 Cycle 变量)，我们需要 Cycle-Major 的矩阵。
            (sparse_matrix, Default::default(), None, None)
        } else if phase2_rounds > 0 {
            // 如果跳过 Phase 1 直接进入 Phase 2 (绑定 Address 变量)，
            // 我们需要将矩阵转置/转换为 Address-Major 格式。
            (Default::default(), sparse_matrix.into(), None, None)
        } else {
            // 理论上不支持两个阶段都是 0 的情况（这意味着没有变量需要 Sumcheck 绑定）。
            unimplemented!("Unsupported configuration: both phase 1 and phase 2 are 0 rounds")
        };

        Self {
            sparse_matrix_phase1,
            sparse_matrix_phase2,
            gruen_eq,
            merged_eq,
            inc,
            ra,
            val,
            params,
        }
    }

    /// 计算 Phase 1 (Cycle 变量绑定阶段) 的当前轮次 Sumcheck 消息。
    ///
    /// # 作用
    /// 在 Sumcheck 协议的每一轮，Prover 需要发送一个单变量多项式给 Verifier。
    /// 该多项式代表了将多元多项式 $P(x_1, \dots, x_v)$ 固定除当前变量 $x_j$ 外的所有变量后，
    /// 对剩余布尔超立方体求和的结果。
    ///
    /// # 核心逻辑
    /// 1. **Gruen 优化配置**:
    ///    本阶段使用了 Gruen 的 `SplitEqPolynomial` 优化。`Eq` 多项式被视为两个较小多项式 $E_{in}$ 和 $E_{out}$ 的张量积。
    ///    这种结构允许我们在遍历稀疏矩阵时，利用分块（Chunking）复用 $E_{out}$ 的评估值，从而减少计算量。
    ///
    /// 2. **并行遍历稀疏矩阵**:
    ///    矩阵按 Cycle-Major 存储。我们将矩阵条目按 `x_out` 索引分组，这样同一组内的条目共享相同的 $E_{out}$ 值。
    ///
    /// 3. **计算子项贡献**:
    ///    对于每一组内的数据，进一步按具体的行（Row/Cycle）分组。
    ///    计算每一行的贡献：$E_{in}(r) \cdot \text{MatrixTerm}(r, \text{val}, \text{inc})$。
    ///    其中 `MatrixTerm` 结合了读写矩阵的值和 `Inc` 多项式的值。
    ///
    /// 4. **聚合与插值**:
    ///    将所有行的贡献累加，并乘以外层的 $E_{out}$。
    ///    最终利用 `previous_claim`（即上一轮的评估和）和计算出的系数，构造出三次单变量多项式返回。
    fn phase1_compute_message(&mut self, previous_claim: F) -> UniPoly<F> {
        let Self {
            inc,
            gruen_eq,
            params,
            sparse_matrix_phase1: sparse_matrix,
            ..
        } = self;
        let gruen_eq = gruen_eq.as_ref().unwrap();

        // 计算 Gruen 优化所需的二次系数。
        // Gruen 优化利用 Eq 多项式的张量结构：Eq(x, y) = E_out(x) * E_in(y)。
        // 当 E_in 已经被之前的轮次完全绑定 (len <= 1) 时，
        // 说明当前的变量在 E_out 部分，此时 E_in_eval = 1，内部求和退化。
        let e_in = gruen_eq.E_in_current();
        let e_in_len = e_in.len();
        // 计算用于分割 x_in 和 x_out 的位宽。
        // 如果 e_in_len 是 1，log_2 为 0；max(1) 保证 log_2 至少对 1 操作。
        let num_x_in_bits = e_in_len.max(1).log_2();
        let x_bitmask = (1 << num_x_in_bits) - 1;

        // 并行计算每一块的贡献，最终规约为 [coeff_0, coeff_1]
        let quadratic_coeffs: [F; DEGREE_BOUND - 1] = sparse_matrix
            .entries
            // 1. 外层分块：根据 x_out 进行分组。
            // 稀疏矩阵已经按行排序，(row / 2) 是为了忽略当前轮次正在求和的最低位（Pairing）。
            // 右移 num_x_in_bits 后，剩下的高位即为 x_out。
            // 属于同一个 Chunk 的条目共享相同的 E_out 值。
            .par_chunk_by(|a, b| ((a.row / 2) >> num_x_in_bits) == ((b.row / 2) >> num_x_in_bits))
            .map(|entries| {
                // 计算当前块的 E_out 值
                let x_out = (entries[0].row / 2) >> num_x_in_bits;
                let E_out_eval = gruen_eq.E_out_current()[x_out];

                // 2. 内层处理：计算块内各行的贡献并求和
                let outer_sum_evals = entries
                    // 按具体的“对行”（row pair）分组。
                    // 每一组包含 row 和 row^1 (即当前变量取 0 和 1 的位置) 的条目。
                    .par_chunk_by(|a, b| a.row / 2 == b.row / 2)
                    .map(|entries| {
                        // 将条目根据当前变量位是 0 (even) 还是 1 (odd) 分割。
                        let odd_row_start_index = entries.partition_point(|entry| entry.row.is_even());
                        let (even_row, odd_row) = entries.split_at(odd_row_start_index);

                        // 获取当前正在处理的基础索引（去除最低位）。
                        let j_prime = 2 * (entries[0].row / 2);

                        // 获取 E_in 的评估值。
                        // 如果 E_in 已经完全绑定，则贡献为 1。
                        let E_in_eval = if e_in_len <= 1 {
                            F::one()
                        } else {
                            let x_in = (j_prime / 2) & x_bitmask;
                            e_in[x_in]
                        };

                        // 获取 Inc 多项式的评估值。
                        // Inc 是多线性的（但在当前维度上看是线性的）。
                        // 我们获取它在当前变量为 0 和 1 时的值，并转换为 [常数项, 斜率] 的形式供 helper 使用。
                        let inc_evals = {
                            let inc_0 = inc.get_bound_coeff(j_prime);     // Inc(..., 0)
                            let inc_1 = inc.get_bound_coeff(j_prime + 1); // Inc(..., 1)
                            let inc_infty = inc_1 - inc_0;                // 实际上是斜率 slope
                            [inc_0, inc_infty]
                        };

                        // 计算具体的矩阵贡献。
                        // RAM 检查公式的核心项：ra * (val + gamma * (inc + val))
                        let inner_sum_evals = ReadWriteMatrixCycleMajor::prover_message_contribution(
                            even_row,
                            odd_row,
                            inc_evals,
                            params.gamma,
                        );

                        // 乘以 E_in 分量
                        [
                            E_in_eval.mul_unreduced::<9>(inner_sum_evals[0]),
                            E_in_eval.mul_unreduced::<9>(inner_sum_evals[1]),
                        ]
                    })
                    // 累加块内所有行的结果
                    .reduce(
                        || [F::Unreduced::<9>::zero(); DEGREE_BOUND - 1],
                        |running, new| [running[0] + new[0], running[1] + new[1]],
                    )
                    .map(F::from_montgomery_reduce);

                // 3. 乘以 E_out 分量
                [
                    E_out_eval.mul_unreduced::<9>(outer_sum_evals[0]),
                    E_out_eval.mul_unreduced::<9>(outer_sum_evals[1]),
                ]
            })
            // 4. 全局归约：累加所有块的结果
            .reduce(
                || [F::Unreduced::<9>::zero(); DEGREE_BOUND - 1],
                |running, new| [running[0] + new[0], running[1] + new[1]],
            )
            .map(F::from_montgomery_reduce);

        // 5. 构造最终的多项式
        // 根据计算出的系数点和 previous_claim（它隐含了 sum(0)+sum(1) 的信息），
        // 恢复出一个三次多项式 (Degree 3)。
        gruen_eq.gruen_poly_deg_3(quadratic_coeffs[0], quadratic_coeffs[1], previous_claim)
    }

    /// 计算 Phase 2 (Address 变量绑定阶段) 的当前轮次 Sumcheck 消息。
    ///
    /// # 作用
    /// 在这一阶段，Sumcheck 正在处理内存地址 ($A_0, \dots, A_m$) 相关的变量。
    /// 函数需要计算并返回一个单变量多项式，该多项式描述了当前轮次被固定的变量 $x_j$ 的边际贡献。
    ///
    /// # 核心逻辑
    /// 1. **基于列（Address）的分组**:
    ///    使用的是 `sparse_matrix_phase2`，它是按 `Address` (列) 优先存储的。
    ///    我们将矩阵条目按列索引对 `(col / 2)` 进行分组。这意味着列 $2k$ 和 $2k+1$ 会被分到一组，
    ///    这对应于当前 Sumcheck 变量取 0 和 取 1 的情况。
    ///
    /// 2. **获取初始值**:
    ///    从 `val_init` 中提取当前处理的两个地址上的初始内存值。
    ///
    /// 3. **计算子项贡献**:
    ///    调用 `prover_message_contribution` 计算这些条目对多项式的贡献。
    ///    由于 Cycle 变量（Phase 1）已经绑定完成，此时 `inc` 和 `merged_eq` 多项式已经被部分求值或作为系数传入。
    ///
    /// 4. **并行聚合**:
    ///    通过 `fold` 和 `reduce` 将所有地址对的贡献累加，得到评估点的值。
    ///
    /// 5. **构造多项式**:
    ///    使用计算得到的点值和上一轮的 Claim 构造单变量多项式。
    fn phase2_compute_message(&self, previous_claim: F) -> UniPoly<F> {
        let Self {
            inc,
            merged_eq,
            sparse_matrix_phase2,
            params,
            ..
        } = self;
        // Phase 1 之后 Eq 多项式应该已经完成了合并
        let merged_eq = merged_eq.as_ref().unwrap();

        // 并行遍历稀疏矩阵，计算当前轮次多项式的评估值
        let evals = sparse_matrix_phase2
            .entries
            // 1. 数据分块：按“列对”分组。
            // 稀疏矩阵按 Address (column) 排序。我们将 column 和 column^1 (即最低位翻转) 的条目分在一组。
            // 这样同一组数据包含了当前变量 x_j = 0 和 x_j = 1 的所有相关项。
            .par_chunk_by(|x, y| x.column() / 2 == y.column() / 2)
            .map(|entries| {
                // 2. 组内拆分：区分偶数列 (x_j=0) 和 奇数列 (x_j=1)
                let odd_col_start_index = entries.partition_point(|entry| entry.column().is_even());
                let (even_col, odd_col) = entries.split_at(odd_col_start_index);

                // 计算实际的列索引
                let even_col_idx = 2 * (entries[0].column() / 2);
                let odd_col_idx = even_col_idx + 1;

                // 3. 计算贡献：调用 Address-Major 矩阵特有的贡献计算函数。
                // * val_init: 获取对应地址的初始内存值。
                // * inc, merged_eq: 传入相关的多项式（包含时间维度的信息）。
                // * params.gamma: 读写检查的随机挑战因子。
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
            // 4. 并行归约 (Fold & Reduce)
            // 将所有线程计算出的部分评估值 [eval_0, eval_1] 累加起来。
            // 使用 Unreduced 类型进行中间累加延迟取模，提高性能。
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

        // 5. 构造并返回单变量多项式
        // 我们利用 `previous_claim` 作为 hint (Previous Claim = P(0) + P(1))。
        // 并传入计算出的 [P(0), P(1)] (或其他插值点)，还原出当前轮次的多项式。
        UniPoly::from_evals_and_hint(
            previous_claim,
            &[
                F::from_barrett_reduce(evals[0]),
                F::from_barrett_reduce(evals[1]),
            ],
        )
    }

    /// 计算 Phase 3 (高位变量绑定阶段) 的当前轮次 Sumcheck 消息。
    ///
    /// # 作用
    /// Phase 3 负责处理前两个阶段未绑定的剩余变量。逻辑上分为两个子阶段：
    /// 1. **绑定剩余 Cycle 变量**: 当 `inc.len() > 1` 时。此时 $eq$, $ra$, $val$, $inc$ 都依赖于当前变量，
    ///    因此被积项的多项式次数为 3。
    /// 2. **绑定剩余 Address 变量**: 当 Cycle 变量全部绑定完 (`inc.len() <= 1`) 后。
    ///    此时 $eq$ 和 $inc$ 对于当前正在绑定的地址变量而言是常数，仅 $ra$ 和 $val$ 变化，
    ///    因此被积项的多项式次数降为 2。
    ///
    /// # 公式回顾
    /// Sumcheck 该步骤验证的核心项是：
    /// $$ eq(r_{cyc}) \cdot ra(addr, r_{cyc}) \cdot (Val(addr, r_{cyc}) + \gamma \cdot (inc(r_{cyc}) + Val(addr, r_{cyc}))) $$
    fn phase3_compute_message(&self, previous_claim: F) -> UniPoly<F> {
        let Self {
            inc,
            merged_eq,
            ra,
            val,
            params,
            ..
        } = self;
        // 在 Phase 3 开始前，这些多项式应当已经被物化（从稀疏矩阵转换为了密集向量形式）
        let merged_eq = merged_eq.as_ref().unwrap();
        let ra = ra.as_ref().unwrap();
        let val = val.as_ref().unwrap();

        // 分支 1: 仍然在处理 Cycle (时间) 变量
        // 只要 inc 多项式的长度大于 1，说明时间维度还没被完全折叠/绑定。
        if inc.len() > 1 {
            // Cycle variables remaining
            // 被积函数涉及 Eq( cycle ) * Ra( cycle ) * Val( cycle )，所以关于当前 cycle 变量是 3 次的。
            const DEGREE: usize = 3;

            // K_prime: 当前剩余的地址空间大小
            let K_prime = params.K >> params.phase2_num_rounds;
            // T_prime: 当前剩余的时间步数 (即 inc 的长度)
            let T_prime = inc.len();
            debug_assert_eq!(ra.len(), K_prime * inc.len());

            // 并行计算：对外层循环（Cycle 维度的一部分）进行并行化
            let evals = (0..inc.len() / 2)
                .into_par_iter()
                .map(|j| {
                    // 获取 inc 和 eq 在当前变量取值下的评估点（通常是 0, 1, 2... 用于插值）
                    // BindingOrder::LowToHigh 表示我们正在从低位向高位绑定
                    let inc_evals = inc.sumcheck_evals(j, DEGREE, BindingOrder::LowToHigh);
                    let eq_evals = merged_eq.sumcheck_evals(j, DEGREE, BindingOrder::LowToHigh);

                    // 内层循环：遍历所有剩余的 Address (空间维度)
                    let inner = (0..K_prime)
                        .into_par_iter()
                        .map(|k| {
                            // 计算 ra 和 val 在当前变量下的评估点
                            // 索引计算：k * T_prime (跳过前面的地址块) / 2 (当前变量折叠) + j (当前时间偏移)
                            let ra_evals = ra.sumcheck_evals(
                                k * T_prime / 2 + j,
                                DEGREE,
                                BindingOrder::LowToHigh,
                            );
                            let val_evals = val.sumcheck_evals(
                                k * T_prime / 2 + j,
                                DEGREE,
                                BindingOrder::LowToHigh,
                            );
                            // 计算被积项的核心公式 (针对 3 个评估点):
                            // term = ra * (val + gamma * (val + inc))
                            [
                                ra_evals[0]
                                    * (val_evals[0] + params.gamma * (val_evals[0] + inc_evals[0])),
                                ra_evals[1]
                                    * (val_evals[1] + params.gamma * (val_evals[1] + inc_evals[1])),
                                ra_evals[2]
                                    * (val_evals[2] + params.gamma * (val_evals[2] + inc_evals[2])),
                            ]
                        })
                        // 累加所有 Address 的贡献
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

                    // 最后乘以 Eq (Eq 只与 Cycle 有关，因此在内层循环外乘)
                    [
                        eq_evals[0] * F::from_barrett_reduce(inner[0]),
                        eq_evals[1] * F::from_barrett_reduce(inner[1]),
                        eq_evals[2] * F::from_barrett_reduce(inner[2]),
                    ]
                })
                // 全局归约：累加所有 Cycle 块的结果
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

            // 构造 3 次单变量多项式
            UniPoly::from_evals_and_hint(
                previous_claim,
                &[
                    F::from_barrett_reduce(evals[0]),
                    F::from_barrett_reduce(evals[1]),
                    F::from_barrett_reduce(evals[2]),
                ],
            )
        } else {
            // 分支 2: Cycle 变量已绑定完毕，开始处理 Address (空间) 变量
            // inc 和 merged_eq 已经完全固定（变为了标量值），不再随当前的 Address 变量变化。
            const DEGREE: usize = 2;

            // 获取常数项
            let inc_eval = inc.final_sumcheck_claim();
            let eq_eval = merged_eq.final_sumcheck_claim();

            // 并行遍历剩余的 ra 长度 (即剩余的 Address 空间)
            let evals = (0..ra.len() / 2)
                .into_par_iter()
                .map(|k| {
                    // 获取 ra 和 val 的评估值 (Degree 2)
                    let ra_evals = ra.sumcheck_evals_array::<DEGREE>(k, BindingOrder::LowToHigh);
                    let val_evals = val.sumcheck_evals_array::<DEGREE>(k, BindingOrder::LowToHigh);

                    // 计算被积项：
                    // term = ra * (val + gamma * (val + CONST_INC))
                    // 这里 inc_eval 是常数，所以关于 ra, val 是二次的。
                    [
                        ra_evals[0] * (val_evals[0] + params.gamma * (val_evals[0] + inc_eval)),
                        ra_evals[1] * (val_evals[1] + params.gamma * (val_evals[1] + inc_eval)),
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

            // 构造 2 次单变量多项式，并乘上全局常数 Eq
            UniPoly::from_evals_and_hint(
                previous_claim,
                &[
                    eq_eval * F::from_barrett_reduce(evals[0]),
                    eq_eval * F::from_barrett_reduce(evals[1]),
                ],
            )
        }
    }

    /// 处理 Phase 1 (Cycle 变量绑定阶段) 的变量绑定操作。
    ///
    /// # 作用
    /// 在 Sumcheck 的每一轮结束收到 Verifier 的挑战 $r_j$ 后，更新 Prover 内部的数据结构。
    /// Phase 1 专注于绑定时间维度（Cycle）的低位变量。
    ///
    /// # 核心逻辑
    /// 1. **绑定基础多项式**:
    ///    * `sparse_matrix`: 稀疏矩阵进行折叠（Fold），合并相邻的行。
    ///    * `gruen_eq`: 更新 Gruen 优化的 Eq 多项式状态。
    ///    * `inc`: 并行绑定 Inc 多项式。
    /// 2. **阶段转换检查**:
    ///    如果是 Phase 1 的最后一轮，需要为下一阶段做准备：
    ///    * **合并 Eq**: 将分裂的 Gruen Eq 多项式合并为标准的密集多项式 `merged_eq`。
    ///    * **矩阵转换**:
    ///      * 如果接着有 Phase 2，将 Cycle-Major 矩阵转换为 Address-Major 矩阵 (`into()`) 以优化空间维度的计算。
    ///      * 如果跳过 Phase 2 直接进入 Phase 3，则直接将稀疏矩阵“物化”（Materialize）为密集的 `ra` (Read Acess) 和 `val` (Value) 多项式向量。
    fn phase1_bind(&mut self, r_j: F::Challenge, round: usize) {
        let Self {
            sparse_matrix_phase1: sparse_matrix,
            inc,
            gruen_eq,
            params,
            ..
        } = self;
        let gruen_eq = gruen_eq.as_mut().unwrap();

        // 绑定稀疏矩阵（折叠行）
        sparse_matrix.bind(r_j);
        // 绑定 Gruen Eq 多项式
        gruen_eq.bind(r_j);
        // 绑定 Inc 多项式（从低位到高位）
        inc.bind_parallel(r_j, BindingOrder::LowToHigh);

        // 检查是否是 Phase 1 的最后一轮
        if round == params.phase1_num_rounds - 1 {
            // 此时 Cycle 的低位已全部绑定，合并 Eq 多项式供后续阶段使用
            self.merged_eq = Some(MultilinearPolynomial::LargeScalars(gruen_eq.merge()));

            // 获取所有权以转换数据结构
            let sparse_matrix = std::mem::take(sparse_matrix);

            if params.phase2_num_rounds > 0 {
                // 进入 Phase 2：转置矩阵为 Address-Major 格式
                self.sparse_matrix_phase2 = sparse_matrix.into();
            } else {
                // 跳过 Phase 2，直接进入 Phase 3（高位绑定阶段）
                // 此时所有的 Cycle 变量已绑定，Address 变量均未绑定。
                // Materialize 将稀疏表示转换为密集向量，便于 Phase 3 处理。
                let T_prime = params.T >> params.phase1_num_rounds;
                let (ra, val) = sparse_matrix.materialize(params.K, T_prime);
                self.ra = Some(ra);
                self.val = Some(val);
            }
        }
    }

    /// 处理 Phase 2 (Address 变量绑定阶段) 的变量绑定操作。
    ///
    /// # 作用
    /// 此时 Sumcheck 正在绑定空间维度（Address）的低位变量。
    ///
    /// # 核心逻辑
    /// 1. **绑定矩阵**: 调用 Address-Major 矩阵的 `bind` 方法，折叠列。
    ///    注意：在此阶段 `inc` 和 `merged_eq` 不需要绑定，因为它们只与 Cycle 有关，而当前绑定的是 Address 变量。
    /// 2. **阶段转换检查**:
    ///    如果是 Phase 2 的最后一轮，说明所有用于优化的稀疏矩阵阶段结束。
    ///    必须将稀疏矩阵完全展开（Materialize）为密集的 `ra` 和 `val` 多项式，
    ///    以便在 Phase 3 中处理剩余的高位变量。
    fn phase2_bind(&mut self, r_j: F::Challenge, round: usize) {
        let Self {
            params,
            sparse_matrix_phase2: sparse_matrix,
            ..
        } = self;

        // 绑定稀疏矩阵（折叠列）
        sparse_matrix.bind(r_j);

        // 检查是否是 Phase 2 的最后一轮
        // round 是全局轮数索引，所以等于 phase1 + phase2 - 1
        if round == params.phase1_num_rounds + params.phase2_num_rounds - 1 {
            let sparse_matrix = std::mem::take(sparse_matrix);
            // 将剩余的稀疏矩阵转换为密集多项式 ra 和 val
            let (ra, val) = sparse_matrix.materialize(
                params.K >> params.phase2_num_rounds, // 剩余的地址空间大小
                params.T >> params.phase1_num_rounds, // 剩余的时间步数
            );
            self.ra = Some(ra);
            self.val = Some(val);
        }
    }

    /// 处理 Phase 3 (高位变量绑定阶段) 的变量绑定操作。
    ///
    /// # 作用
    /// Phase 3 是“清理”阶段，处理剩余的所有变量。
    /// 逻辑顺序是先绑定剩余的 Cycle 高位，再绑定剩余的 Address 高位。
    ///
    /// # 核心逻辑
    /// 1. **绑定 Cycle 变量（如果存在）**:
    ///    如果 `inc` 的长度大于 1，说明时间维度还没被完全折叠成标量。
    ///    此时需要绑定 `inc` 和 `merged_eq`。
    /// 2. **绑定 Address 变量**:
    ///    `ra` 和 `val` 同时依赖于 Cycle 和 Address，因此无论处于 Phase 3 的哪个子阶段，
    ///    都需要对它们进行绑定。
    fn phase3_bind(&mut self, r_j: F::Challenge) {
        let Self {
            ra,
            val,
            inc,
            merged_eq,
            ..
        } = self;

        let merged_eq = merged_eq.as_mut().unwrap();
        let ra = ra.as_mut().unwrap();
        let val = val.as_mut().unwrap();

        // Check: 是否仍在绑定剩余的 Cycle 变量？
        if inc.len() > 1 {
            // Cycle variables remaining
            // Inc 和 Eq 只与 Cycle 有关，仅在此阶段需要绑定
            inc.bind_parallel(r_j, BindingOrder::LowToHigh);
            merged_eq.bind_parallel(r_j, BindingOrder::LowToHigh);
        }

        // Unconditionally bind ra and val
        // ra(addr, cycle) 和 val(addr, cycle) 既受 cycle 影响也受 address 影响。
        // * 当绑定剩余 Cycle 变量时，它们随之折叠；
        // * 当 Cycle 绑定完开始绑定剩余 Address 变量时，它们继续折叠。
        ra.bind_parallel(r_j, BindingOrder::LowToHigh);
        val.bind_parallel(r_j, BindingOrder::LowToHigh);
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for RamReadWriteCheckingProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(skip_all, name = "RamReadWriteCheckingProver::compute_message")]
    fn compute_message(&mut self, round: usize, previous_claim: F) -> UniPoly<F> {
        if round < self.params.phase1_num_rounds {
            self.phase1_compute_message(previous_claim)
        } else if round < self.params.phase1_num_rounds + self.params.phase2_num_rounds {
            self.phase2_compute_message(previous_claim)
        } else {
            self.phase3_compute_message(previous_claim)
        }
    }

    #[tracing::instrument(skip_all, name = "RamReadWriteCheckingProver::ingest_challenge")]
    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        if round < self.params.phase1_num_rounds {
            self.phase1_bind(r_j, round);
        } else if round < self.params.phase1_num_rounds + self.params.phase2_num_rounds {
            self.phase2_bind(r_j, round);
        } else {
            self.phase3_bind(r_j);
        }
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RamVal,
            SumcheckId::RamReadWriteChecking,
            opening_point.clone(),
            self.val.as_ref().unwrap().final_sumcheck_claim(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RamRa,
            SumcheckId::RamReadWriteChecking,
            opening_point.clone(),
            self.ra.as_ref().unwrap().final_sumcheck_claim(),
        );
        let (_, r_cycle) = opening_point.split_at(self.params.K.log_2());
        accumulator.append_dense(
            transcript,
            CommittedPolynomial::RamInc,
            SumcheckId::RamReadWriteChecking,
            r_cycle.r,
            self.inc.final_sumcheck_claim(),
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

pub struct RamReadWriteCheckingVerifier<F: JoltField> {
    params: RamReadWriteCheckingParams<F>,
}

impl<F: JoltField> RamReadWriteCheckingVerifier<F> {
    pub fn new(
        opening_accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
        one_hot_params: &OneHotParams,
        trace_length: usize,
        config: &ReadWriteConfig,
    ) -> Self {
        let params = RamReadWriteCheckingParams::new(
            opening_accumulator,
            transcript,
            one_hot_params,
            trace_length,
            config,
        );
        RamReadWriteCheckingVerifier { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
for RamReadWriteCheckingVerifier<F>
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
        let (_, r_cycle) = r.split_at(self.params.K.log_2());

        let eq_eval_cycle = EqPolynomial::mle_endian(&self.params.r_cycle, &r_cycle);

        let (_, ra_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamRa,
            SumcheckId::RamReadWriteChecking,
        );
        let (_, val_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamVal,
            SumcheckId::RamReadWriteChecking,
        );
        let (_, inc_claim) = accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::RamInc,
            SumcheckId::RamReadWriteChecking,
        );

        eq_eval_cycle * ra_claim * (val_claim + self.params.gamma * (val_claim + inc_claim))
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RamVal,
            SumcheckId::RamReadWriteChecking,
            opening_point.clone(),
        );

        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RamRa,
            SumcheckId::RamReadWriteChecking,
            opening_point.clone(),
        );
        let (_, r_cycle) = opening_point.split_at(self.params.K.log_2());
        accumulator.append_dense(
            transcript,
            CommittedPolynomial::RamInc,
            SumcheckId::RamReadWriteChecking,
            r_cycle.r,
        );
    }
}
