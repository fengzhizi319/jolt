use std::iter::zip;
use std::sync::Arc;

use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use common::constants::XLEN;
use num_traits::Zero;
use rayon::prelude::*;
use strum::{EnumCount, IntoEnumIterator};
use tracer::instruction::Cycle;

use super::LOG_K;

use crate::{
    field::{JoltField, MulTrunc},
    poly::{
        dense_mlpoly::DensePolynomial,
        eq_poly::EqPolynomial,
        identity_poly::{IdentityPolynomial, OperandPolynomial, OperandSide},
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN,
        },
        prefix_suffix::{Prefix, PrefixRegistry, PrefixSuffixDecomposition},
        split_eq_poly::GruenSplitEqPolynomial,
        unipoly::UniPoly,
    },
    subprotocols::{
        mles_product_sum::{eval_linear_prod_accumulate, finish_mles_product_sum_from_evals},
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
    utils::{
        expanding_table::ExpandingTable,
        lookup_bits::LookupBits,
        math::Math,
        thread::{drop_in_background_thread, unsafe_allocate_zero_vec},
    },
    zkvm::{
        config::{self, OneHotParams},
        instruction::{Flags, InstructionLookup, InterleavedBitsMarker, LookupQuery},
        lookup_table::{
            prefixes::{PrefixCheckpoint, PrefixEval, Prefixes},
            suffixes::Suffixes,
            LookupTables,
        },
        witness::VirtualPolynomial,
    },
};

use rayon::iter::{IndexedParallelIterator, ParallelIterator};

// Instruction lookups: Read + RAF batched sumcheck
//
// Notation:
// - Field F. Let K = 2^{LOG_K}, T = 2^{log_T}.
// - Address index k ∈ {0..K-1}, cycle index j ∈ {0..T-1}.
// - eq(k; r_addr) := multilinear equality polynomial over LOG_K vars.
// - eq(j; r_reduction) := equality polynomials over LOG_T vars.
// - ra(k, j) is the selector arising from prefix/suffix condensation.
//   It is decomposed as the product of virtual sub selectors:
//   ra((k_0, k_1, ..., k_{n-1}), j) := ra_0(k_0, j) * ra_1(k_1, j) * ... * ra_{n-1}(k_{n-1}, j).
//   n is typically 1, 2, 4 or 8.
//   logically ra(k, j) = 1 when the j-th cycle's lookup key equals k, and 0 otherwise.// - Val_j(k) ∈ F is the lookup-table value selected by (j, k); concretely Val_j(k) = table_j(k)
//   if cycle j uses a table and 0 otherwise (materialized via prefix/suffix decomposition).
// - raf_flag(j) ∈ {0,1} is 1 iff the instruction at cycle j is NOT interleaved operands.
// - Let LeftPrefix_j, RightPrefix_j, IdentityPrefix_j ∈ F be the address-only (prefix) factors for
//   the left/right operand and identity polynomials at cycle j (from `PrefixSuffixDecomposition`).
//
// We introduce a batching challenge γ ∈ F. Define
//   RafVal_j(k) := (1 - raf_flag(j)) · (LeftPrefix_j + γ · RightPrefix_j)
//                  + raf_flag(j) · γ · IdentityPrefix_j.
// The overall γ-weights are arranged so that γ multiplies RafVal_j(k) in the final identity.
//
// Claims supplied by the accumulator (LHS), all claimed at `SumcheckId::InstructionClaimReduction`
// and `SumcheckId::SpartanProductVirtualization`:
// - rv         := ⟦LookupOutput⟧
// - left_op    := ⟦LeftLookupOperand⟧
// - right_op   := ⟦RightLookupOperand⟧
//   Combined as: rv + γ·left_op + γ^2·right_op
//
// Statement proved by this sumcheck (RHS), for random challenges
// r_addr ∈ F^{LOG_K}, r_reduction ∈ F^{log_T}:
//
//   rv(r_reduction) + γ·left_op(r_reduction) + γ^2·right_op(r_reduction)
//   = Σ_{j=0}^{T-1} Σ_{k=0}^{K-1} [ eq(j; r_reduction) · ra(k, j) · (Val_j(k) + γ · RafVal_j(k)) ].
//
// Prover structure:
// - First log(K) rounds bind address vars using prefix/suffix decomposition, accumulating:
//   Σ_k ra(k, j)·Val_j(k)  and  Σ_k ra(k, j)·RafVal_j(k)
//   for each j (via u_evals vectors and suffix polynomials).
// - Last log(T) rounds bind cycle vars producing a degree-3 univariate with the required previous-round claim.
// - The published univariate matches the RHS above; the verifier checks it against the LHS claims.

#[derive(Allocative, Clone)]
pub struct InstructionReadRafSumcheckParams<F: JoltField> {
    /// γ and its square (γ^2) used for batching rv/branch/raf components.
    pub gamma: F,
    pub gamma_sqr: F,
    /// log2(T): number of cycle variables (last rounds bind cycles).
    pub log_T: usize,
    /// How many address variables each virtual ra polynomial has.
    pub ra_virtual_log_k_chunk: usize,
    /// Number of phases for instruction lookups.
    pub phases: usize,
    pub r_reduction: OpeningPoint<BIG_ENDIAN, F>,
}

impl<F: JoltField> InstructionReadRafSumcheckParams<F> {
    /// 该函数负责初始化指令查找表 Sumcheck 参数，主要通过生成随机挑战 <span>\gamma</span> 将输出与操作数 Claim 批量合并，
    /// 并从前序阶段提取时间维度归约点 r_reduction。
    ///
    /// # 功能
    /// 该函数负责设置 Sumcheck 实例所需的各项参数，主要完成以下任务：
    /// 1. **生成批处理随机数 ($\gamma$)**: 通过 Transcript 生成随机挑战 $\gamma$，用于将三个独立的查找表 Claim
    ///    (LookupOutput, LeftOperand, RightOperand) 线性组合成一个 Claim 进行批量证明。
    ///    组合形式为: $Claim = output + \gamma \cdot left + \gamma^2 \cdot right$。
    /// 2. **获取归约挑战点**: 从上一阶段 (`InstructionClaimReduction`) 的累加器中提取
    ///    用于时间周期 (Cycle) 维度的随机挑战点 `r_reduction`。
    ///
    /// # 参数
    ///
    /// * `n_cycle_vars` - 周期变量的数量 (即 $log_2(\text{Trace Length})$)。
    /// * `one_hot_params` - One-hot 编码配置，定义了地址分块的大小 (`ra_virtual_log_k_chunk`)。
    /// * `opening_accumulator` - 包含前序阶段生成的 Challenge 和 Claim 的累加器。
    /// * `transcript` - Fiat-Shamir 协议的 Transcript，用于生成不可预测的随机数。
    pub fn new(
        n_cycle_vars: usize,
        one_hot_params: &OneHotParams,
        opening_accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        // 1. 生成随机挑战 gamma。
        //    这个 gamma 用于将三个不同的 claim (Lookup返回值, 左操作数, 右操作数)
        //    压缩成一个单一的多项式 claim，以此通过一次 Sumcheck 完成所有验证。
        let gamma = transcript.challenge_scalar::<F>();
        let gamma_sqr = gamma.square(); // 预计算平方，因为组合形式是 1, gamma, gamma^2 的线性组合

        // 2. 确定 Sumcheck 的执行阶段数 (Phases)。
        //    根据 Trace 的长度决定是否需要分阶段执行，以优化内存或计算效率。
        let phases = config::get_instruction_sumcheck_phases(n_cycle_vars);

        // 3. 获取上一阶段产生的挑战点 r_reduction。
        //    在 `InstructionClaimReduction` (Stage 1/2 部分) 中，我们已经针对时间维度 (Cycle)
        //    进行了归约。这里提取那个归约产生的随机点，作为本阶段 Sumcheck 中
        //    Cycle 变量 (最后 log(T) 轮) 的绑定目标。
        //    注意：虽然这里取的是 LookupOutput 的点，但对于同一阶段的所有多项式，r_reduction 是共享的。
        let (r_reduction, _) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::LookupOutput,
            SumcheckId::InstructionClaimReduction,
        );

        Self {
            gamma,
            gamma_sqr,
            log_T: n_cycle_vars,
            ra_virtual_log_k_chunk: one_hot_params.lookups_ra_virtual_log_k_chunk,
            phases,
            r_reduction,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for InstructionReadRafSumcheckParams<F> {
    fn num_rounds(&self) -> usize {
        LOG_K + self.log_T
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let (_, rv_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::LookupOutput,
            SumcheckId::InstructionClaimReduction,
        );
        let (_, rv_claim_branch) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::LookupOutput,
            SumcheckId::SpartanProductVirtualization,
        );
        // TODO: Make error and move to more appropriate place.
        assert_eq!(rv_claim, rv_claim_branch);
        let (_, left_operand_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::LeftLookupOperand,
            SumcheckId::InstructionClaimReduction,
        );
        let (_, right_operand_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RightLookupOperand,
            SumcheckId::InstructionClaimReduction,
        );
        rv_claim + self.gamma * left_operand_claim + self.gamma_sqr * right_operand_claim
    }

    fn degree(&self) -> usize {
        let n_virtual_ra_polys = LOG_K / self.ra_virtual_log_k_chunk;
        n_virtual_ra_polys + 2
    }

    fn normalize_opening_point(
        &self,
        challenges: &[<F as JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        let (r_address_prime, r_cycle_prime) = challenges.split_at(LOG_K);
        let r_cycle_prime = r_cycle_prime.iter().copied().rev().collect::<Vec<_>>();

        OpeningPoint::new([r_address_prime.to_vec(), r_cycle_prime].concat())
    }
}

/// Sumcheck prover for [`InstructionReadRafSumcheckVerifier`].
///
/// Binds address variables first using prefix/suffix decomposition to aggregate, per cycle j,
///   Σ_k ra(k, j)·Val_j(k) and Σ_k ra(k, j)·RafVal_j(k),
#[derive(Allocative)]
pub struct InstructionReadRafSumcheckProver<F: JoltField> {
    /// The execution trace, shared via Arc for efficient access in cache_openings.
    #[allocative(skip)]
    trace: Arc<Vec<Cycle>>,

    /// Running list of sumcheck challenges r_j (address then cycle) in binding order.
    r: Vec<F::Challenge>,

    /// Precomputed lookup keys k (bit-packed) per cycle j.
    lookup_indices: Vec<LookupBits>,
    /// Indices of cycles grouped by selected lookup table; used to form per-table flags.
    lookup_indices_by_table: Vec<Vec<usize>>,
    /// Per-cycle flag: instruction uses interleaved operands.
    is_interleaved_operands: Vec<bool>,

    /// Prefix checkpoints for each registered `Prefix` variant, updated every two rounds.
    prefix_checkpoints: Vec<PrefixCheckpoint<F>>,
    /// For each lookup table, dense polynomials holding suffix contributions in the current phase.
    suffix_polys: Vec<Vec<DensePolynomial<F>>>,
    /// Expanding tables accumulating address-prefix products per phase.
    v: Vec<ExpandingTable<F>>,
    /// u_evals for read-checking and RAF: eq(r_reduction,j).
    u_evals: Vec<F>,

    /// Registry holding prefix checkpoint values for `PrefixSuffixDecomposition` instances.
    prefix_registry: PrefixRegistry<F>,
    /// Prefix-suffix decomposition for right operand identity polynomial family.
    right_operand_ps: PrefixSuffixDecomposition<F, 2>,
    /// Prefix-suffix decomposition for left operand identity polynomial family.
    left_operand_ps: PrefixSuffixDecomposition<F, 2>,
    /// Prefix-suffix decomposition for the instruction-identity path (RAF flag path).
    identity_ps: PrefixSuffixDecomposition<F, 2>,

    /// Gruen-split equality polynomial over cycle vars. Present only in the last log(T) rounds.
    eq_r_reduction: GruenSplitEqPolynomial<F>,

    /// Materialized `ra_i(k_i, j)` polynomials. Present only in the last log(T) rounds.
    ra_polys: Option<Vec<MultilinearPolynomial<F>>>,

    /// Materialized Val_j(k) + γ · RafVal_j(k) over (address, cycle) for final log T rounds.
    /// Combines lookup table values with γ-weighted RAF operand contributions.
    combined_val_polynomial: Option<MultilinearPolynomial<F>>,

    #[allocative(skip)]
    params: InstructionReadRafSumcheckParams<F>,
}

impl<F: JoltField> InstructionReadRafSumcheckProver<F> {
    /// Creates a prover-side instance for the Read+RAF batched sumcheck.
    ///
    /// Builds prover-side working state:
    /// - Precomputes per-cycle lookup index, interleaving flags, and table choices
    /// - Buckets cycles by table and by path (interleaved vs identity)
    /// - Allocates per-table suffix accumulators and u-evals for rv/raf parts
    /// - Instantiates the three RAF decompositions and Gruen EQs over cycles
    /// 初始化指令读取与 RAF (Read, Arithmetic, Flag) 批量 Sumcheck 的 Prover 实例
    ///
    /// # 功能
    ///
    /// 该函数负责构建 Read+RAF Sumcheck 协议所需的初始状态。该协议用于证明指令查找表操作的正确性，
    /// 即证明：`lookup(Input) == Output` 以及操作数分解的正确性。
    ///
    /// 主要完成以下初始化工作：
    /// 1. **前缀-后缀分解 (Prefix-Suffix Decomposition) 初始化**: 为左操作数、右操作数和 Identity 多项式
    ///    初始化分解结构。这是一种针对稀疏或结构化多项式的优化技术，用于加速 Sumcheck。
    /// 2. **Trace 预处理**: 并行遍历执行轨迹 (Trace)，提取每个 Cycle 的查找键 (Lookup Index)、
    ///    交错操作数标志 (Interleaved Flag) 以及所使用的具体查找表类型。
    /// 3. **数据重组**:
    ///    - 构建全局的 `lookup_indices` 和 `is_interleaved_operands` 向量。
    ///    - 构建 `lookup_indices_by_table`，将 Cycle 按所使用的查找表进行分组，以便后续针对特定表进行批处理。
    /// 4. **多项式准备**:
    ///    - 初始化用于后续阶段的后缀多项式占位符。
    ///    - 计算 `eq(r_reduction, j)` 的评估值 (`u_evals`)，这是上一阶段归约产生的系数。
    ///    - 初始化 `GruenSplitEqPolynomial`，用于最后 $log(T)$ 轮的时间维度绑定。
    /// 5. **阶段启动**: 调用 `init_phase(0)` 启动第一阶段的前缀计算。
    ///
    /// # 参数
    ///
    /// * `params` - Sumcheck 参数，包含批处理随机数 $\gamma$、阶段配置和归约点 $r_{reduction}$。
    /// * `trace` - 完整的指令执行轨迹。
    #[tracing::instrument(skip_all, name = "InstructionReadRafSumcheckProver::initialize")]
    /// 初始化指令 RAF (Read Access Frequency) 证明器
    ///
    /// **函数功能**:
    /// 负责将原始的执行轨迹 (Trace) 转化为 Lasso 查找算法所需的多维索引数据结构。
    /// 它并不直接生成证明，而是准备所有的"原材料"，包括：
    /// 1. **Index 分解**: 将 64位的大索引分解为多个小的 chunks (Prefix/Suffix)。
    /// 2. **Table 路由**: 确定每一步操作归属于哪个查找表 (ADD, MUL 等)。
    /// 3. **Time 权重**: 预计算时间维度的随机线性组合系数。
    ///
    /// **输入**:
    /// - `params`: 包含 Verifier 的随机挑战点 (r_reduction) 和系统配置 (phases, LOG_K)。
    /// - `trace`: CPU 的完整执行记录。
    pub fn initialize(params: InstructionReadRafSumcheckParams<F>, trace: Arc<Vec<Cycle>>) -> Self {
        // 计算 Trace 长度的对数，用于确定时间维度多项式的变量数
        let log_T = trace.len().log_2();

        // ========================================================================
        // 1. 初始化前缀-后缀分解器 (Prefix-Suffix Decomposition)
        // ------------------------------------------------------------------------
        // **背景**: Lasso 算法为了避免对巨大的查找表 (如 2^64) 进行整体承诺，
        // 将大的 lookup index 拆分为多个较小的片段 (Phases/Chunks)。
        // 例如：将 64位拆分为 4 个 16位片段。
        //
        // 这里分别为三种不同的"输入源"建立分解器：
        // - Right Operand: 右操作数
        // - Left Operand: 左操作数
        // - Identity: 电路标志位 (Flags)
        // ========================================================================

        // log_m: 每个 Phase 处理的比特数 (例如 64位 / 4 phases = 16位)
        let log_m = LOG_K / params.phases;

        // 创建基础多项式对象
        let right_operand_poly = OperandPolynomial::new(LOG_K, OperandSide::Right);
        let left_operand_poly = OperandPolynomial::new(LOG_K, OperandSide::Left);
        let identity_poly = IdentityPolynomial::new(LOG_K);

        let span = tracing::span!(tracing::Level::INFO, "Init PrefixSuffixDecomposition");
        let _guard = span.enter();

        // 包装为分解器，支持按 chunk 提取数据
        let right_operand_ps =
            PrefixSuffixDecomposition::new(Box::new(right_operand_poly), log_m, LOG_K);
        let left_operand_ps =
            PrefixSuffixDecomposition::new(Box::new(left_operand_poly), log_m, LOG_K);
        let identity_ps = PrefixSuffixDecomposition::new(Box::new(identity_poly), log_m, LOG_K);
        drop(_guard);
        drop(span);

        let num_tables = LookupTables::<XLEN>::COUNT;

        // ========================================================================
        // 2. 并行提取 Trace 元数据 (Build CycleData)
        // ------------------------------------------------------------------------
        // **目标**: 遍历 Trace，解析出每一条指令具体"查了什么表"、"查了什么索引"。
        // 这是一个计算密集型操作，使用 Rayon 进行并行处理。
        // ========================================================================
        let span = tracing::span!(tracing::Level::INFO, "Build cycle_data");
        let _guard = span.enter();

        // 临时结构体，用于存储单步解析结果
        struct CycleData<const XLEN: usize> {
            idx: usize,                     // Trace 中的绝对索引 (时间步 t)
            lookup_index: LookupBits,       // 解析出的查找键值 (Key)
            is_interleaved: bool,           // 是否使用交错操作数模式 (针对 64位扩展)
            table: Option<LookupTables<XLEN>>, // 该指令归属的查找表 (如 Table_ADD)
        }

        let cycle_data: Vec<CycleData<XLEN>> = trace
            .par_iter()
            .enumerate()
            .map(|(idx, cycle)| {
                // 将 CPU 状态转换为具体的查找索引 (例如: R1=5, R2=3 -> Index=8)
                let bits = LookupBits::new(LookupQuery::<XLEN>::to_lookup_index(cycle), LOG_K);

                // 检查指令标志位，判断操作数解析模式
                let is_interleaved = cycle
                    .instruction()
                    .circuit_flags()
                    .is_interleaved_operands();

                // 获取该指令对应的逻辑表
                let table = cycle.lookup_table();

                CycleData {
                    idx,
                    lookup_index: bits,
                    is_interleaved,
                    table,
                }
            })
            .collect();
        drop(_guard);
        drop(span);

        // ========================================================================
        // 3. 构建核心平面向量 (Extract Vectors)
        // ------------------------------------------------------------------------
        // **优化**: 将 Struct 数组 (`Vec<CycleData>`) 拆解为多个原生数组 (`Vec<u64>`).
        // 这样做可以提高内存局部性 (Cache Locality)，并方便后续 SIMD 加速。
        // ========================================================================
        let span = tracing::span!(tracing::Level::INFO, "Extract vectors");
        let _guard = span.enter();

        // 预分配内存，避免 realloc
        let mut lookup_indices = Vec::with_capacity(cycle_data.len());
        let mut is_interleaved_operands = Vec::with_capacity(cycle_data.len());

        {
            let span = tracing::span!(tracing::Level::INFO, "par_extend basic vectors");
            let _guard = span.enter();
            // 并行填充数据
            lookup_indices.par_extend(cycle_data.par_iter().map(|data| data.lookup_index));
            is_interleaved_operands
                .par_extend(cycle_data.par_iter().map(|data| data.is_interleaved));
        }

        // ========================================================================
        // 4. 按 Table 进行分组/倒排索引 (Binning / Routing)
        // ------------------------------------------------------------------------
        // **目标**: 建立 "Table ID -> [Cycle Indices]" 的映射。
        // 如果 Trace 中第 0, 5, 8 步是 ADD 指令，那么 lookup_indices_by_table[ADD_ID] = [0, 5, 8]。
        // 这允许后续 Prover 针对每个 Table 独立并行地生成证明，而不需要全量扫描 Trace。
        // ========================================================================
        let lookup_indices_by_table: Vec<Vec<usize>> = (0..num_tables)
            .into_par_iter() // 并行遍历所有表类型
            .map(|t_idx| {
                // 对于每种表，扫描 cycle_data 收集属于它的时间步
                cycle_data
                    .par_iter()
                    .filter_map(|data| {
                        data.table.and_then(|t| {
                            // 匹配 Table 枚举索引
                            if LookupTables::<XLEN>::enum_index(&t) == t_idx {
                                Some(data.idx)
                            } else {
                                None
                            }
                        })
                    })
                    .collect()
            })
            .collect();

        // **内存优化**: cycle_data 是一个巨大的中间结构，不再需要。
        // 在后台线程释放它，防止阻塞主线程 (Drop 大内存可能耗时数毫秒)。
        drop_in_background_thread(cycle_data);
        drop(_guard);
        drop(span);

        // ========================================================================
        // 5. 初始化后缀多项式容器 (Suffix Polynomials Init)
        // ------------------------------------------------------------------------
        // 为每个查找表的"值列" (Values Column) 预留多项式空间。
        // Lasso 需要证明 Index 指向的 Value 是正确的。
        // 这里的 DensePolynomial 暂时为空，具体数值会在 `init_phase` 中按需计算填入。
        // ========================================================================
        let suffix_polys: Vec<Vec<DensePolynomial<F>>> = LookupTables::<XLEN>::iter()
            .collect::<Vec<_>>()
            .par_iter()
            .map(|table| {
                table
                    .suffixes()
                    .par_iter()
                    .map(|_| DensePolynomial::default()) // 占位符
                    .collect()
            })
            .collect();

        // ========================================================================
        // 6. 计算时间维度权重 (Time Binding / U_evals)
        // ------------------------------------------------------------------------
        // **数学原理**: Sumcheck 需要证明 \sum_t Eq(r, t) * Count(t)。
        // `r_reduction` 是 Verifier 提供的用于压缩时间维度的随机点。
        // `u_evals` 预计算了 Eq(r_reduction, t) 对所有 t 的值。
        // 这是将 O(N) 的验证压缩为 O(1) 的关键一步。
        // ========================================================================
        let span = tracing::span!(tracing::Level::INFO, "Compute u_evals");
        let _guard = span.enter();

        // 用于后续 Split-Sumcheck 的辅助结构
        let eq_poly_r_reduction =
            GruenSplitEqPolynomial::<F>::new(&params.r_reduction.r, BindingOrder::LowToHigh);

        // 计算 Eq 表: [Eq(r, 0), Eq(r, 1), ..., Eq(r, T-1)]
        let u_evals = EqPolynomial::evals(&params.r_reduction.r);
        drop(_guard);
        drop(span);

        // ========================================================================
        // 7. 构造 Prover 实例并启动计算
        // ------------------------------------------------------------------------
        // ========================================================================
        let mut res = Self {
            trace,
            r: Vec::with_capacity(log_T + LOG_K),
            lookup_indices,

            // --- 地址绑定 (Space Binding) 相关状态 ---
            lookup_indices_by_table,
            is_interleaved_operands,
            prefix_checkpoints: vec![None.into(); Prefixes::COUNT],
            suffix_polys,
            // 动态规划表 (ExpandingTable):
            // 用于在 Sumcheck 每一轮迭代中，累积 Eq(r_high, prefix) 的乘积。
            // 为每个 Phase 分配一个独立的表。
            v: (0..params.phases)
                .map(|_| ExpandingTable::new(1 << log_m, BindingOrder::HighToLow))
                .collect(),
            u_evals,
            right_operand_ps,
            left_operand_ps,
            identity_ps,

            // --- 时间绑定 (Time Binding) 相关状态 ---
            ra_polys: None,
            eq_r_reduction: eq_poly_r_reduction,
            prefix_registry: PrefixRegistry::new(),
            combined_val_polynomial: None,
            params,
        };

        // **立即启动 Phase 0**:
        // Lasso 是分阶段的。初始化完成后，立即开始 Phase 0 (处理地址最高位或最低位 Chunk) 的
        // 多项式预计算，为 Sumcheck 的第一轮交互做准备。
        res.init_phase(0);
        res
    }

    /// To be called in the beginning of each phase, before any binding
    /// Phase initialization for address-binding:
    /// - Condenses prior-phase u-evals through the expanding-table v[phase-1]
    /// - Builds Q for RAF (Left/Right dual and Identity) from cycle buckets
    /// - Refreshes per-table read-checking suffix polynomials for this phase
    /// - Initializes/caches P via the shared `PrefixRegistry`
    /// - Resets the current expanding table accumulator for this phase
    #[tracing::instrument(skip_all, name = "InstructionReadRafProver::init_phase")]
    fn init_phase(&mut self, phase: usize) {
        let log_m = LOG_K / self.params.phases;
        let m = 1 << log_m;
        let m_mask = m - 1;
        // Condensation
        if phase != 0 {
            let span = tracing::span!(tracing::Level::INFO, "Update u_evals");
            let _guard = span.enter();
            self.lookup_indices
                .par_iter()
                .zip(&mut self.u_evals)
                .for_each(|(k, u_eval)| {
                    let (prefix, _) = k.split((self.params.phases - phase) * log_m);
                    let k_bound = prefix & m_mask;
                    *u_eval *= self.v[phase - 1][k_bound];
                });
        }

        PrefixSuffixDecomposition::init_Q_raf(
            &mut self.left_operand_ps,
            &mut self.right_operand_ps,
            &mut self.identity_ps,
            &self.u_evals,
            &self.lookup_indices,
            &self.is_interleaved_operands,
        );

        self.init_suffix_polys(phase);

        self.identity_ps.init_P(&mut self.prefix_registry);
        self.right_operand_ps.init_P(&mut self.prefix_registry);
        self.left_operand_ps.init_P(&mut self.prefix_registry);

        self.v[phase].reset(F::one());
    }

    /// Recomputes per-table suffix accumulators for the current phase of read-checking.
    ///
    /// For each lookup table's suffix family, this function:
    /// 1. Partitions cycles by their current chunk value (the `log_m`-bit segment
    ///    extracted from each cycle's lookup index for this phase).
    /// 2. Aggregates weighted contributions `u_evals[j] * suffix_mle(suffix_bits)`
    ///    into dense MLEs of size `M = 2^{log_m}`.
    ///
    /// # Suffix classification
    ///
    /// Suffixes are classified into three categories for efficient accumulation:
    /// - **`Suffixes::One`**: Always evaluates to 1; we simply accumulate `u_evals[j]`.
    /// - **{0,1}-valued suffixes**: Add `u_evals[j]` only when `suffix_mle == 1`.
    /// - **General suffixes**: Multiply `u_evals[j]` by `suffix_mle` value.
    #[tracing::instrument(skip_all, name = "InstructionReadRafProver::init_suffix_polys")]
    fn init_suffix_polys(&mut self, phase: usize) {
        /// Maximum number of suffixes any lookup table can have.
        /// (Currently `ValidSignedRemainderTable` has the most with 5.)
        const MAX_SUFFIXES: usize = 5;

        let log_m = LOG_K / self.params.phases;
        let m = 1 << log_m;
        let m_mask = m - 1;
        let num_threads = rayon::current_num_threads();
        let chunk_size = self.lookup_indices.len().div_ceil(num_threads).max(1);

        let new_suffix_polys: Vec<_> = {
            LookupTables::<XLEN>::iter()
                .collect::<Vec<_>>()
                .par_iter()
                .zip(self.lookup_indices_by_table.par_iter())
                .map(|(table, lookup_indices)| {
                    let suffixes = table.suffixes();
                    let num_suffixes = suffixes.len();
                    debug_assert!(num_suffixes <= MAX_SUFFIXES);

                    // Early exit: if no cycles use this table, return zero polynomials
                    if lookup_indices.is_empty() {
                        return vec![unsafe_allocate_zero_vec(m); num_suffixes];
                    }

                    // Pre-partition suffixes using fixed-size arrays to avoid heap allocation.
                    // Also track `Suffixes::One` separately to avoid per-cycle match check.
                    let mut suffix_one_idx: Option<usize> = None;
                    let mut suffix_01_indices = [0usize; MAX_SUFFIXES];
                    let mut suffix_01_count = 0usize;
                    let mut suffix_other_indices = [0usize; MAX_SUFFIXES];
                    let mut suffix_other_count = 0usize;

                    for (s_idx, suffix) in suffixes.iter().enumerate() {
                        if matches!(suffix, Suffixes::One) {
                            suffix_one_idx = Some(s_idx);
                        } else if suffix.is_01_valued() {
                            suffix_01_indices[suffix_01_count] = s_idx;
                            suffix_01_count += 1;
                        } else {
                            suffix_other_indices[suffix_other_count] = s_idx;
                            suffix_other_count += 1;
                        }
                    }

                    let unreduced_polys = lookup_indices
                        .par_chunks(chunk_size)
                        .map(|chunk| {
                            // Single allocation for all suffix accumulators:
                            // layout: [suffix_0 | suffix_1 | ... | suffix_{num_suffixes-1}],
                            // each suffix segment has length `m`.
                            let total_len = num_suffixes * m;
                            let mut chunk_result: Vec<F::Unreduced<6>> =
                                unsafe_allocate_zero_vec(total_len);

                            for j in chunk {
                                let k = self.lookup_indices[*j];
                                let (prefix_bits, suffix_bits) =
                                    k.split((self.params.phases - 1 - phase) * log_m);
                                let idx = prefix_bits & m_mask;
                                let u = self.u_evals[*j];

                                // Suffixes::One always evaluates to 1, so just add u directly.
                                if let Some(one_idx) = suffix_one_idx {
                                    chunk_result[one_idx * m + idx] += *u.as_unreduced_ref();
                                }

                                // Other {0,1}-valued suffixes: add u when suffix_mle == 1.
                                for i in 0..suffix_01_count {
                                    let s_idx = suffix_01_indices[i];
                                    let t = suffixes[s_idx].suffix_mle::<XLEN>(suffix_bits);
                                    debug_assert!(t == 0 || t == 1);
                                    if t == 1 {
                                        chunk_result[s_idx * m + idx] += *u.as_unreduced_ref();
                                    }
                                }

                                // General suffixes: multiply by t.
                                for i in 0..suffix_other_count {
                                    let s_idx = suffix_other_indices[i];
                                    let t = suffixes[s_idx].suffix_mle::<XLEN>(suffix_bits);
                                    if t != 0 {
                                        chunk_result[s_idx * m + idx] += u.mul_u64_unreduced(t);
                                    }
                                }
                            }

                            chunk_result
                        })
                        .reduce(
                            || unsafe_allocate_zero_vec(num_suffixes * m),
                            |mut acc, new| {
                                // Merge accumulator vectors (parallelize over the flat buffer)
                                acc.par_iter_mut()
                                    .zip(new.par_iter())
                                    .for_each(|(a, b)| *a += b);
                                acc
                            },
                        );

                    // Reduce the unreduced values to field elements (parallelized over suffixes)
                    (0..num_suffixes)
                        .into_par_iter()
                        .map(|s_idx| {
                            let start = s_idx * m;
                            let end = start + m;
                            unreduced_polys[start..end]
                                .iter()
                                .copied()
                                .map(F::from_barrett_reduce)
                                .collect::<Vec<F>>()
                        })
                        .collect::<Vec<_>>()
                })
                .collect()
        };

        // Replace existing suffix polynomials
        self.suffix_polys
            .iter_mut()
            .zip(new_suffix_polys.into_iter())
            .for_each(|(old, new)| {
                old.iter_mut()
                    .zip(new.into_iter())
                    .for_each(|(poly, mut coeffs)| {
                        *poly = DensePolynomial::new(std::mem::take(&mut coeffs));
                    });
            });
    }

    /// To be called before the last log(T) rounds
    /// Handoff between address and cycle rounds:
    /// - Materializes all virtual ra_i(k_i,j) from expanding tables across all phases
    /// - Commits prefix checkpoints into a fixed `PrefixEval` vector
    /// - Materializes Val_j(k) from table prefixes/suffixes
    /// - Materializes RafVal_j(k) from (Left,Right,Identity) prefixes with γ-weights
    /// - Converts ra/Val/RafVal into MultilinearPolynomial over (addr,cycle)
    #[tracing::instrument(skip_all, name = "InstructionReadRafProver::init_log_t_rounds")]
    fn init_log_t_rounds(&mut self, gamma: F, gamma_sqr: F) {
        let log_m = LOG_K / self.params.phases;
        let m = 1 << log_m;
        let m_mask = m - 1;
        let num_cycles = self.lookup_indices.len();
        // Drop stuff that's no longer needed
        drop_in_background_thread(std::mem::take(&mut self.u_evals));

        let ra_polys: Vec<MultilinearPolynomial<F>> = {
            let span = tracing::span!(tracing::Level::INFO, "Materialize ra polynomials");
            let _guard = span.enter();
            assert!(self.v.len().is_power_of_two());
            let n = LOG_K / self.params.ra_virtual_log_k_chunk;
            let chunk_size = self.v.len() / n;
            self.v
                .chunks(chunk_size)
                .enumerate()
                .map(|(chunk_i, v_chunk)| {
                    let phase_offset = chunk_i * chunk_size;
                    let res = self
                        .lookup_indices
                        .par_iter()
                        .with_min_len(1024)
                        .map(|i| {
                            // Hot path: compute ra_i(k_i, j) as a product of per-phase expanding-table
                            // values. This is performance sensitive, so we:
                            // - Convert `LookupBits` -> `u128` once per cycle
                            // - Use a decrementing shift instead of recomputing `(phases-1-phase)*log_m`
                            // - Avoid an initial multiply-by-one by seeding `acc` with the first term
                            let v: u128 = (*i).into();

                            if v_chunk.is_empty() {
                                return F::one();
                            }

                            // shift(phase) = (phases - 1 - phase) * log_m
                            // For consecutive phases, this decreases by `log_m` each step.
                            let mut shift = (self.params.phases - 1 - phase_offset) * log_m;

                            let mut iter = v_chunk.iter();
                            let first = iter.next().unwrap();
                            let first_idx = ((v >> shift) as usize) & m_mask;
                            let mut acc = first[first_idx];

                            for table in iter {
                                shift -= log_m;
                                let idx = ((v >> shift) as usize) & m_mask;
                                acc *= table[idx];
                            }

                            acc
                        })
                        .collect::<Vec<F>>();
                    res.into()
                })
                .collect()
        };

        drop_in_background_thread(std::mem::take(&mut self.v));

        let prefixes: Vec<PrefixEval<F>> = std::mem::take(&mut self.prefix_checkpoints)
            .into_iter()
            .map(|checkpoint| checkpoint.unwrap())
            .collect();
        // Materialize combined_val_poly = Val_j(k) + γ·RafVal_j(k)
        // combining lookup table values with RAF operand contributions in a single pass.
        let mut combined_val_poly: Vec<F> = unsafe_allocate_zero_vec(num_cycles);
        {
            let span = tracing::span!(tracing::Level::INFO, "Materialize combined_val_poly");
            let _guard = span.enter();
            let left_prefix = self.prefix_registry.checkpoints[Prefix::LeftOperand].unwrap();
            let right_prefix = self.prefix_registry.checkpoints[Prefix::RightOperand].unwrap();
            let identity_prefix = self.prefix_registry.checkpoints[Prefix::Identity].unwrap();
            let raf_interleaved = gamma * left_prefix + gamma_sqr * right_prefix;
            let raf_identity = gamma_sqr * identity_prefix;

            // At this point we've finished all LOG_K address rounds, so the lookup-table suffix
            // variable set is empty. That means every suffix MLE is evaluated on an empty bitstring,
            // and `table.combine(&prefixes, &suffixes)` becomes a per-table constant that can be
            // precomputed once (instead of allocating a suffix Vec per cycle).
            let empty_suffix_bits = LookupBits::new(0, 0);
            let table_values_at_r_addr: Vec<F> = LookupTables::<XLEN>::iter()
                .map(|table| {
                    let suffix_evals: Vec<F> = table
                        .suffixes()
                        .iter()
                        .map(|suffix| {
                            // Suffix MLEs are u64-valued; convert once here.
                            F::from_u64(suffix.suffix_mle::<XLEN>(empty_suffix_bits))
                        })
                        .collect();
                    table.combine(&prefixes, &suffix_evals)
                })
                .collect();

            combined_val_poly
                .par_iter_mut()
                .zip(self.trace.par_iter())
                .zip(std::mem::take(&mut self.is_interleaved_operands))
                .for_each(|((val, cycle), is_interleaved_operands)| {
                    // Add lookup table value (Val_j(k)) - derive table from trace
                    if let Some(table) = cycle.lookup_table() {
                        let t_idx = LookupTables::<XLEN>::enum_index(&table);
                        *val += table_values_at_r_addr[t_idx];
                    }
                    // Add RAF operand contribution (γ·RafVal_j(k))
                    if is_interleaved_operands {
                        *val += raf_interleaved;
                    } else {
                        *val += raf_identity;
                    }
                });
        }

        self.combined_val_polynomial = Some(MultilinearPolynomial::from(combined_val_poly));
        self.ra_polys = Some(ra_polys);

        // After the address rounds are complete and we have materialized `ra_polys` and the
        // `combined_val_polynomial`, the following buffers are no longer needed for the remaining
        // log(T) cycle rounds:
        // - `lookup_indices` (used only to build `ra_polys` and to size `combined_val_poly`)
        // - `suffix_polys` (used only during the first LOG_K address rounds)
        drop_in_background_thread((
            std::mem::take(&mut self.lookup_indices),
            std::mem::take(&mut self.suffix_polys),
        ));
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T>
for InstructionReadRafSumcheckProver<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(skip_all, name = "InstructionReadRafSumcheckProver::compute_message")]
    /// Produces the prover's degree-≤3 univariate for the current round.
    ///
    /// - For the first LOG_K rounds: returns two evaluations combining
    ///   read-checking and RAF prefix–suffix messages (at X∈{0,2}).
    /// - For the last log(T) rounds: uses Gruen-split EQ.
        #[tracing::instrument(skip_all, name = "InstructionReadRafSumcheckProver::compute_message")]
        /// 计算 Prover 在当前轮次需要发送的单变量多项式消息。
        ///
        /// # 逻辑概览
        /// Sumcheck 协议将多变量多项式的求和问题转化为一系列单变量多项式的交互。
        /// 本函数根据当前轮数 `round` 的不同，分别处理两个阶段：
        /// 1. **Phase 1 (Address Binding)**: 前 LOG_K 轮。处理地址变量，使用 Prefix/Suffix 分解逻辑。
        /// 2. **Phase 2 (Cycle Binding)**: 后 log(T) 轮。处理时间变量，使用 Gruen Split 优化进行并行累加。
        ///
        /// # 参数
        /// - `round`: 当前是第几轮 Sumcheck (从 0 开始)。
        /// - `previous_claim`: 上一轮 Verifier 发来的挑战值在多项式上的评估结果 (用于一致性检查或优化)。
    fn compute_message(&mut self, round: usize, previous_claim: F) -> UniPoly<F> {
        if round < LOG_K {
            // =================================================================
            // Phase 1: 地址绑定阶段 (Address Binding)
            // 目标：将查找表的 Index 变量 (x_0...x_63) 绑定为随机数 r_addr。
            // =================================================================

            // 对应 initialize:
            // 这里消耗的是 initialize 中生成的 "PrefixSuffixDecomposition" 数据结构。
            // 因为大表被拆分成了小块 (Chunks)，这里的计算涉及复杂的“进位”和“指纹传递”逻辑，
            // 也就是之前讨论的 "Grand Product" 在拆分场景下的变体。
            self.compute_prefix_suffix_prover_message(round, previous_claim)
        } else {
            // =================================================================
            // Phase 2: 周期/时间绑定阶段 (Cycle/Time Binding)
            // 目标：将 Trace 的时间变量 (t_0...t_n) 绑定为随机数 r_time。
            // =================================================================

            // 此时，地址变量已经被完全绑定。这就好比我们已经确定了 "我们要查什么地址(r_addr)"。
            // 现在要证明的是："在整个 CPU 执行历史中，对 r_addr 的访问总和是对的"。

            // 公式目标： sum_{t} Eq(r_time, t) * [ Combined_Val(t) * RA_Selector(t) ]

            // 1. 获取当前活跃的多项式引用 (消耗 initialize 准备的数据)
            // ---------------------------------------------------------
            // ra_polys (Read Access Polys):
            // 对应 initialize 中的 `lookup_indices_by_table` 分组逻辑。
            // 如果有 3 个表 (ADD, MUL, SUB)，这里就有 3 个多项式。
            // 它们充当 "Selector"：ra_polys[i](t) = 1 表示时刻 t 正在访问表 i。
            let ra_polys = self.ra_polys.as_ref().unwrap();

            // combined_val:
            // 对应 initialize 中提取的 `lookup_indices` (Trace 数据)。
            // 它实际上是 Value(t) 和 RAF_Counter(t) 的线性组合。
            let combined_val = self.combined_val_polynomial.as_ref().unwrap();

            // 计算乘积项的总数
            // 我们要计算 P(x) = Combined(x) * RA_0(x) * RA_1(x) ...
            // 如果有 N 个 RA 多项式，加上 Combined 多项式，乘积的度数最高为 N+1。
            // 因此我们需要计算 N+2 个点的评估值才能确定这个单变量多项式。
            let n_evals = ra_polys.len() + 1;

            // 2. 并行计算和 (Gruen Split Accumulation) - 核心算法
            // ---------------------------------------------------------
            // 这是一个优化算法。朴素做法是遍历所有 2^N 个点，非常慢。
            // Jolt 使用 "Gruen Split" 将巨大的 Eq 多项式切分为 "高位(Out)" 和 "低位(In)" 两部分。
            // 这允许我们将大任务切分为小块，并行分发给 CPU 的不同核心。
            let mut sum_evals = self
                .eq_r_reduction
                .E_out_current() // 外部循环：遍历 Eq 的高位块
                .par_iter()      // <--- 关键：Rayon 并行迭代
                .enumerate()
                .map(|(j_out, e_out)| {
                    // --- 线程内部逻辑 ---

                    // pairs: 存储线性对 (Linear Pairs)。
                    // 对于每一层 Sumcheck，多线性多项式在当前变量 x_j 上都是线性的： L(x) = A(1-x) + Bx
                    // 我们只需要存储 A (x=0时的值) 和 B (x=1时的值)。
                    let mut pairs = vec![(F::zero(), F::zero()); n_evals];

                    // evals_acc: 累加器。
                    // 存储最终乘积多项式在 {0, 1, ..., n_evals} 这些点上的值。
                    // Unreduced 是为了性能：只做整数加法，不频繁做取模运算 (Mod P)。
                    let mut evals_acc = vec![F::Unreduced::<9>::zero(); n_evals];

                    // 内部循环：遍历 Eq 的低位块 (通常大小适合 L1/L2 Cache)
                    for (j_in, e_in) in self.eq_r_reduction.E_in_current().iter().enumerate() {
                        // 计算全局索引 j (对应当前时刻 t 的前半部分)
                        // 这里的 j 对应变量 x_i = 0 的位置，j+1 对应 x_i = 1 的位置 (ML 存储布局)
                        let j = self.eq_r_reduction.group_index(j_out, j_in);

                        // 准备数据容器
                        let Some((val_pair, ra_pairs)) = pairs.split_first_mut() else {
                            unreachable!()
                        };

                        // --- Step A: 加载 Trace 数据 (Combined Val) ---
                        // 从 initialize 准备的大数组中，直接拿出相邻的两个值。
                        let v_at_0 = combined_val.get_bound_coeff(2 * j);     // x_i = 0
                        let v_at_1 = combined_val.get_bound_coeff(2 * j + 1); // x_i = 1

                        // 构造线性项 L_val(x)
                        // 注意：我们将 Eq 的因子 e_in 直接乘到了这里。
                        // 这是 Sumcheck 的标准优化：Eq(r, x) * Poly(x)
                        *val_pair = (*e_in * v_at_0, *e_in * v_at_1);

                        // --- Step B: 加载 Selector 数据 (RA Polys) ---
                        // 遍历所有表的分组多项式，取出它们在当前时刻的值。
                        zip(ra_pairs, ra_polys).for_each(|(pair, ra_poly)| {
                            let eval_at_0 = ra_poly.get_bound_coeff(2 * j);
                            let eval_at_1 = ra_poly.get_bound_coeff(2 * j + 1);
                            *pair = (eval_at_0, eval_at_1);
                        });

                        // --- Step C: 核心数学计算 ---
                        // 输入：一组线性函数 [L_0, L_1, ... L_k] (即 pairs)
                        // 动作：
                        // 1. 构造乘积多项式 P(x) = L_0(x) * ... * L_k(x)
                        // 2. 计算 P(x) 在 x \in {0, 1, ... deg} 处的值
                        // 3. 累加到 evals_acc 中
                        eval_linear_prod_accumulate(&pairs, &mut evals_acc);
                    }

                    // 线程收尾：归约 (Reduce) 并应用外部 Eq 系数
                    evals_acc
                        .into_iter()
                        .map(|v| F::from_montgomery_reduce(v) * e_out)
                        .collect::<Vec<F>>()
                })
                // --- 聚合所有线程的结果 ---
                .reduce(
                    || vec![F::zero(); n_evals],
                    |a, b| zip(a, b).map(|(a, b)| a + b).collect(),
                );

            // 3. 后处理：应用全局缩放因子
            // 由于 Gruen Split 的数学特性，可能还有一个全局系数需要乘上去。
            let current_scalar = self.eq_r_reduction.get_current_scalar();
            sum_evals.iter_mut().for_each(|v| *v *= current_scalar);

            // 4. 插值生成消息 (Univariate Polynomial)
            // 我们现在有了总和多项式在点 0, 1, 2... 处的值。
            // Verifier 需要的是系数形式的多项式 (ax^2 + bx + c)。
            // 这里进行拉格朗日插值 (Lagrange Interpolation) 转换格式。
            finish_mles_product_sum_from_evals(&sum_evals, previous_claim, &self.eq_r_reduction)
        }
    }

    #[tracing::instrument(skip_all, name = "InstructionReadRafSumcheckProver::ingest_challenge")]
    /// Binds the next variable (address or cycle) and advances state.
    ///
    /// Address rounds: bind all active prefix–suffix polynomials and the
    /// expanding-table accumulator; update checkpoints every two rounds;
    /// initialize next phase/handoff when needed. Cycle rounds: bind the ra/Val
    /// polynomials and Gruen EQ.
    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        let log_m = LOG_K / self.params.phases;
        self.r.push(r_j);
        if round < LOG_K {
            let phase = round / log_m;
            rayon::scope(|s| {
                s.spawn(|_| {
                    self.suffix_polys.par_iter_mut().for_each(|polys| {
                        polys
                            .par_iter_mut()
                            .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::HighToLow))
                    });
                });
                s.spawn(|_| self.identity_ps.bind(r_j));
                s.spawn(|_| self.right_operand_ps.bind(r_j));
                s.spawn(|_| self.left_operand_ps.bind(r_j));
                s.spawn(|_| self.v[phase].update(r_j));
            });
            {
                if self.r.len().is_multiple_of(2) {
                    // Calculate suffix_len based on phases, using the same formula as original current_suffix_len
                    let suffix_len = LOG_K - (round / log_m + 1) * log_m;
                    Prefixes::update_checkpoints::<XLEN, F, F::Challenge>(
                        &mut self.prefix_checkpoints,
                        self.r[self.r.len() - 2],
                        self.r[self.r.len() - 1],
                        round,
                        suffix_len,
                    );
                }
            }

            // check if this is the last round in the phase
            if (round + 1).is_multiple_of(log_m) {
                self.prefix_registry.update_checkpoints();
                if phase != self.params.phases - 1 {
                    // if not last phase, init next phase
                    self.init_phase(phase + 1);
                }
            }

            if (round + 1) == LOG_K {
                self.init_log_t_rounds(self.params.gamma, self.params.gamma_sqr);
            }
        } else {
            // log(T) rounds

            self.eq_r_reduction.bind(r_j);
            self.combined_val_polynomial
                .as_mut()
                .unwrap()
                .bind_parallel(r_j, BindingOrder::LowToHigh);

            self.ra_polys
                .as_mut()
                .unwrap()
                .iter_mut()
                .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::LowToHigh));
        }
    }

    #[tracing::instrument(skip_all, name = "ReadRafSumcheckProver::cache_openings")]
    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let r_sumcheck = self.params.normalize_opening_point(sumcheck_challenges);
        // Prover publishes new virtual openings derived by this sumcheck:
        // - Per-table LookupTableFlag(i) at r_cycle
        // - InstructionRa at r_sumcheck (ra MLE's final claim)
        // - InstructionRafFlag at r_cycle
        let (r_address, r_cycle) = r_sumcheck.clone().split_at(LOG_K);

        // Compute flag claims using split-eq + unreduced accumulation for efficiency.
        // This avoids materializing the full eq table (size T) and instead uses
        // E_hi (size √T) and E_lo (size √T), iterating contiguously for cache locality.
        let (flag_claims, raf_flag_claim) = self.compute_flag_claims(&r_cycle);

        for (i, claim) in flag_claims.into_iter().enumerate() {
            accumulator.append_virtual(
                transcript,
                VirtualPolynomial::LookupTableFlag(i),
                SumcheckId::InstructionReadRaf,
                r_cycle.clone(),
                claim,
            );
        }

        let ra_polys = self.ra_polys.as_ref().unwrap();
        let mut r_address_chunks = r_address.r.chunks(LOG_K / ra_polys.len());
        for (i, ra_poly) in self.ra_polys.as_ref().unwrap().iter().enumerate() {
            let r_address = r_address_chunks.next().unwrap();
            let opening_point =
                OpeningPoint::<BIG_ENDIAN, F>::new([r_address, &*r_cycle.r].concat());
            accumulator.append_virtual(
                transcript,
                VirtualPolynomial::InstructionRa(i),
                SumcheckId::InstructionReadRaf,
                opening_point,
                ra_poly.final_sumcheck_claim(),
            );
        }

        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::InstructionRafFlag,
            SumcheckId::InstructionReadRaf,
            r_cycle.clone(),
            raf_flag_claim,
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

impl<F: JoltField> InstructionReadRafSumcheckProver<F> {
    /// Address-round prover message: sum of read-checking and RAF components.
    ///
    /// Each component is a degree-2 univariate evaluated at X∈{0,2} using
    /// prefix–suffix decomposition, then added to form the batched message.
    fn compute_prefix_suffix_prover_message(&self, round: usize, previous_claim: F) -> UniPoly<F> {
        let mut read_checking = [F::zero(), F::zero()];
        let mut raf = [F::zero(), F::zero()];

        rayon::join(
            || {
                read_checking = self.prover_msg_read_checking(round);
            },
            || {
                raf = self.prover_msg_raf();
            },
        );

        let eval_at_0 = read_checking[0] + raf[0];
        let eval_at_2 = read_checking[1] + raf[1];

        UniPoly::from_evals_and_hint(previous_claim, &[eval_at_0, eval_at_2])
    }

    /// RAF part for address rounds.
    ///
    /// Builds two evaluations at X∈{0,2} for the batched
    /// (Left + γ·Right) vs Identity path, folding γ-weights into the result.
    fn prover_msg_raf(&self) -> [F; 2] {
        let len = self.identity_ps.Q_len();
        let [left_0, left_2, right_0, right_2] = (0..len / 2)
            .into_par_iter()
            .map(|b| {
                let (i0, i2) = self.identity_ps.sumcheck_evals(b);
                let (r0, r2) = self.right_operand_ps.sumcheck_evals(b);
                let (l0, l2) = self.left_operand_ps.sumcheck_evals(b);
                [
                    *l0.as_unreduced_ref(),
                    *l2.as_unreduced_ref(),
                    *(i0 + r0).as_unreduced_ref(),
                    *(i2 + r2).as_unreduced_ref(),
                ]
            })
            .fold_with([F::Unreduced::<5>::zero(); 4], |running, new| {
                [
                    running[0] + new[0],
                    running[1] + new[1],
                    running[2] + new[2],
                    running[3] + new[3],
                ]
            })
            .reduce(
                || [F::Unreduced::zero(); 4],
                |running, new| {
                    [
                        running[0] + new[0],
                        running[1] + new[1],
                        running[2] + new[2],
                        running[3] + new[3],
                    ]
                },
            );
        [
            F::from_montgomery_reduce(
                left_0.mul_trunc::<4, 9>(self.params.gamma.as_unreduced_ref())
                    + right_0.mul_trunc::<4, 9>(self.params.gamma_sqr.as_unreduced_ref()),
            ),
            F::from_montgomery_reduce(
                left_2.mul_trunc::<4, 9>(self.params.gamma.as_unreduced_ref())
                    + right_2.mul_trunc::<4, 9>(self.params.gamma_sqr.as_unreduced_ref()),
            ),
        ]
    }

    /// Read-checking part for address rounds.
    ///
    /// For each lookup table, evaluates Σ P(0)·Q^L, Σ P(2)·Q^L, Σ P(2)·Q^R via
    /// table-specific suffix families, then returns [g(0), g(2)] by the standard
    /// quadratic interpolation trick.
    fn prover_msg_read_checking(&self, j: usize) -> [F; 2] {
        let lookup_tables: Vec<_> = LookupTables::<XLEN>::iter().collect();

        let len = self.suffix_polys[0][0].len();
        let log_len = len.log_2();

        let r_x = if j % 2 == 1 {
            self.r.last().copied()
        } else {
            None
        };

        let [eval_0, eval_2_left, eval_2_right] = (0..len / 2)
            .into_par_iter()
            .flat_map_iter(|b| {
                let b = LookupBits::new(b as u128, log_len - 1);
                let prefixes_c0: Vec<_> = Prefixes::iter()
                    .map(|prefix| {
                        prefix.prefix_mle::<XLEN, F, F::Challenge>(
                            &self.prefix_checkpoints,
                            r_x,
                            0,
                            b,
                            j,
                        )
                    })
                    .collect();
                let prefixes_c2: Vec<_> = Prefixes::iter()
                    .map(|prefix| {
                        prefix.prefix_mle::<XLEN, F, F::Challenge>(
                            &self.prefix_checkpoints,
                            r_x,
                            2,
                            b,
                            j,
                        )
                    })
                    .collect();
                lookup_tables
                    .iter()
                    .zip(self.suffix_polys.iter())
                    .map(move |(table, suffixes)| {
                        let suffixes_left: Vec<_> =
                            suffixes.iter().map(|suffix| suffix[b.into()]).collect();
                        let suffixes_right: Vec<_> = suffixes
                            .iter()
                            .map(|suffix| suffix[usize::from(b) + len / 2])
                            .collect();
                        [
                            table.combine(&prefixes_c0, &suffixes_left),
                            table.combine(&prefixes_c2, &suffixes_left),
                            table.combine(&prefixes_c2, &suffixes_right),
                        ]
                    })
            })
            .fold_with([F::Unreduced::<5>::zero(); 3], |running, new| {
                [
                    running[0] + new[0].as_unreduced_ref(),
                    running[1] + new[1].as_unreduced_ref(),
                    running[2] + new[2].as_unreduced_ref(),
                ]
            })
            .reduce(
                || [F::Unreduced::zero(); 3],
                |running, new| {
                    [
                        running[0] + new[0],
                        running[1] + new[1],
                        running[2] + new[2],
                    ]
                },
            )
            .map(F::from_barrett_reduce);
        [eval_0, eval_2_right + eval_2_right - eval_2_left]
    }

    /// Compute per-table flag claims and RAF flag claim using split-eq + unreduced accumulation.
    ///
    /// For each lookup table i, computes: flag_claim[i] = Σ_{j: table[j] == i} eq(r_cycle, j)
    /// For RAF flag: raf_flag_claim = Σ_{j: identity path} eq(r_cycle, j)
    ///
    /// Uses split-eq optimization:
    /// - Split r_cycle into hi/lo halves, compute E_hi and E_lo (each size √T)
    /// - Parallelize over E_hi chunks (c_hi)
    /// - For each c_hi, iterate sequentially over c_lo for cache locality
    /// - Use unreduced 5-limb accumulation within each c_hi block
    #[tracing::instrument(skip_all, name = "ReadRafSumcheckProver::compute_flag_claims")]
    fn compute_flag_claims(&self, r_cycle: &OpeningPoint<BIG_ENDIAN, F>) -> (Vec<F>, F) {
        let T = self.trace.len();
        let num_tables = LookupTables::<XLEN>::COUNT;

        // Split-eq: divide r_cycle into MSB (hi) and LSB (lo) halves
        let log_T = r_cycle.len();
        let lo_bits = log_T / 2;
        let hi_bits = log_T - lo_bits;
        let (r_hi, r_lo) = r_cycle.r.split_at(hi_bits);

        let (E_hi, E_lo) = rayon::join(
            || EqPolynomial::<F>::evals(r_hi),
            || EqPolynomial::<F>::evals(r_lo),
        );

        let in_len = E_lo.len();

        // Parallel over E_hi chunks
        let num_threads = rayon::current_num_threads();
        let out_len = E_hi.len();
        let chunk_size = out_len.div_ceil(num_threads).max(1);

        E_hi.par_chunks(chunk_size)
            .enumerate()
            .map(|(chunk_idx, chunk)| {
                // Partial accumulators for this thread (field elements)
                let mut partial_flags: Vec<F> = vec![F::zero(); num_tables];
                let mut partial_raf: F = F::zero();

                let chunk_start = chunk_idx * chunk_size;
                for (local_idx, &e_hi) in chunk.iter().enumerate() {
                    let c_hi = chunk_start + local_idx;
                    let c_hi_base = c_hi * in_len;

                    // Local unreduced accumulators for this c_hi (5-limb)
                    let mut local_flags: Vec<F::Unreduced<5>> =
                        vec![F::Unreduced::<5>::zero(); num_tables];
                    let mut local_raf: F::Unreduced<5> = F::Unreduced::<5>::zero();

                    // Sequential over c_lo (contiguous cycles for this c_hi)
                    for c_lo in 0..in_len {
                        let j = c_hi_base + c_lo;
                        if j >= T {
                            break;
                        }

                        let cycle = &self.trace[j];
                        let e_lo_unreduced = *E_lo[c_lo].as_unreduced_ref();

                        // Accumulate table flag
                        if let Some(table) = cycle.lookup_table() {
                            let t_idx = LookupTables::<XLEN>::enum_index(&table);
                            local_flags[t_idx] += e_lo_unreduced;
                        }

                        // Accumulate RAF flag (identity = not interleaved)
                        if !cycle
                            .instruction()
                            .circuit_flags()
                            .is_interleaved_operands()
                        {
                            local_raf += e_lo_unreduced;
                        }
                    }

                    // Reduce and scale by e_hi
                    for t_idx in 0..num_tables {
                        let reduced = F::from_barrett_reduce::<5>(local_flags[t_idx]);
                        partial_flags[t_idx] += e_hi * reduced;
                    }
                    let raf_reduced = F::from_barrett_reduce::<5>(local_raf);
                    partial_raf += e_hi * raf_reduced;
                }

                (partial_flags, partial_raf)
            })
            .reduce(
                || (vec![F::zero(); num_tables], F::zero()),
                |(mut a_flags, a_raf), (b_flags, b_raf)| {
                    for (a, b) in a_flags.iter_mut().zip(b_flags.iter()) {
                        *a += *b;
                    }
                    (a_flags, a_raf + b_raf)
                },
            )
    }
}

/// Instruction lookups: batched Read + RAF sumcheck.
///
/// Let K = 2^{LOG_K}, T = 2^{log_T}. For random r_addr ∈ F^{LOG_K}, r_reduction ∈ F^{log_T},
/// this sumcheck proves that the accumulator claims
///   rv + γ·left_op + γ^2·right_op
/// equal the double sum over (j, k):
///   Σ_j Σ_k [ eq(j; r_reduction) · ra(k, j) · (Val_j(k) + γ·RafVal_j(k)) ].
/// It is implemented as: first log(K) address-binding rounds (prefix/suffix condensation), then
/// last log(T) cycle-binding rounds driven by [`GruenSplitEqPolynomial`].
pub struct InstructionReadRafSumcheckVerifier<F: JoltField> {
    params: InstructionReadRafSumcheckParams<F>,
}

impl<F: JoltField> InstructionReadRafSumcheckVerifier<F> {
    pub fn new(
        n_cycle_vars: usize,
        one_hot_params: &OneHotParams,
        opening_accumulator: &VerifierOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let params = InstructionReadRafSumcheckParams::new(
            n_cycle_vars,
            one_hot_params,
            opening_accumulator,
            transcript,
        );
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
for InstructionReadRafSumcheckVerifier<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        // Verifier's RHS reconstruction from virtual claims at r:
        //
        // Computes Val and RafVal contributions at r_address, forms EQ(r_cycle)
        // for InstructionClaimReduction sumcheck, multiplies by ra claim at r_sumcheck,
        // and returns the batched identity RHS to be matched against the LHS input claim.
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        let (r_address_prime, r_cycle_prime) = opening_point.split_at(LOG_K);
        let left_operand_eval =
            OperandPolynomial::<F>::new(LOG_K, OperandSide::Left).evaluate(&r_address_prime.r);
        let right_operand_eval =
            OperandPolynomial::<F>::new(LOG_K, OperandSide::Right).evaluate(&r_address_prime.r);
        let identity_poly_eval = IdentityPolynomial::<F>::new(LOG_K).evaluate(&r_address_prime.r);
        let val_evals: Vec<_> = LookupTables::<XLEN>::iter()
            .map(|table| table.evaluate_mle::<F, F::Challenge>(&r_address_prime.r))
            .collect();

        let r_reduction = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::LookupOutput,
                SumcheckId::InstructionClaimReduction,
            )
            .0
            .r;
        let eq_eval_r_reduction = EqPolynomial::<F>::mle(&r_reduction, &r_cycle_prime.r);

        let n_virtual_ra_polys = LOG_K / self.params.ra_virtual_log_k_chunk;
        let ra_claim = (0..n_virtual_ra_polys)
            .map(|i| {
                accumulator
                    .get_virtual_polynomial_opening(
                        VirtualPolynomial::InstructionRa(i),
                        SumcheckId::InstructionReadRaf,
                    )
                    .1
            })
            .product::<F>();

        let table_flag_claims: Vec<F> = (0..LookupTables::<XLEN>::COUNT)
            .map(|i| {
                accumulator
                    .get_virtual_polynomial_opening(
                        VirtualPolynomial::LookupTableFlag(i),
                        SumcheckId::InstructionReadRaf,
                    )
                    .1
            })
            .collect();

        let raf_flag_claim = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::InstructionRafFlag,
                SumcheckId::InstructionReadRaf,
            )
            .1;

        let val_claim = val_evals
            .into_iter()
            .zip(table_flag_claims)
            .map(|(claim, val)| claim * val)
            .sum::<F>();

        let raf_claim = (F::one() - raf_flag_claim)
            * (left_operand_eval + self.params.gamma * right_operand_eval)
            + raf_flag_claim * self.params.gamma * identity_poly_eval;

        eq_eval_r_reduction * ra_claim * (val_claim + self.params.gamma * raf_claim)
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let r_sumcheck = self.params.normalize_opening_point(sumcheck_challenges);
        // Verifier requests the virtual openings that the prover must provide
        // for this sumcheck (same set as published by the prover-side cache).
        let (r_address, r_cycle) = r_sumcheck.split_at(LOG_K);

        (0..LookupTables::<XLEN>::COUNT).for_each(|i| {
            accumulator.append_virtual(
                transcript,
                VirtualPolynomial::LookupTableFlag(i),
                SumcheckId::InstructionReadRaf,
                r_cycle.clone(),
            );
        });

        for (i, r_address_chunk) in r_address
            .r
            .chunks(self.params.ra_virtual_log_k_chunk)
            .enumerate()
        {
            let opening_point =
                OpeningPoint::<BIG_ENDIAN, F>::new([r_address_chunk, &*r_cycle.r].concat());
            accumulator.append_virtual(
                transcript,
                VirtualPolynomial::InstructionRa(i),
                SumcheckId::InstructionReadRaf,
                opening_point,
            );
        }

        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::InstructionRafFlag,
            SumcheckId::InstructionReadRaf,
            r_cycle.clone(),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::subprotocols::sumcheck::BatchedSumcheck;
    use crate::transcripts::Blake2bTranscript;
    use ark_bn254::Fr;
    use ark_std::Zero;
    use rand::{rngs::StdRng, RngCore, SeedableRng};
    use strum::IntoEnumIterator;
    use tracer::instruction::Cycle;

    const LOG_T: usize = 8;
    const T: usize = 1 << LOG_T;

    fn random_instruction(rng: &mut StdRng, instruction: &Option<Cycle>) -> Cycle {
        let instruction = instruction.unwrap_or_else(|| {
            let index = rng.next_u64() as usize % Cycle::COUNT;
            Cycle::iter()
                .enumerate()
                .filter(|(i, _)| *i == index)
                .map(|(_, x)| x)
                .next()
                .unwrap()
        });

        match instruction {
            Cycle::ADD(cycle) => cycle.random(rng).into(),
            Cycle::ADDI(cycle) => cycle.random(rng).into(),
            Cycle::AND(cycle) => cycle.random(rng).into(),
            Cycle::ANDN(cycle) => cycle.random(rng).into(),
            Cycle::ANDI(cycle) => cycle.random(rng).into(),
            Cycle::AUIPC(cycle) => cycle.random(rng).into(),
            Cycle::BEQ(cycle) => cycle.random(rng).into(),
            Cycle::BGE(cycle) => cycle.random(rng).into(),
            Cycle::BGEU(cycle) => cycle.random(rng).into(),
            Cycle::BLT(cycle) => cycle.random(rng).into(),
            Cycle::BLTU(cycle) => cycle.random(rng).into(),
            Cycle::BNE(cycle) => cycle.random(rng).into(),
            Cycle::FENCE(cycle) => cycle.random(rng).into(),
            Cycle::JAL(cycle) => cycle.random(rng).into(),
            Cycle::JALR(cycle) => cycle.random(rng).into(),
            Cycle::LUI(cycle) => cycle.random(rng).into(),
            Cycle::LD(cycle) => cycle.random(rng).into(),
            Cycle::MUL(cycle) => cycle.random(rng).into(),
            Cycle::MULHU(cycle) => cycle.random(rng).into(),
            Cycle::OR(cycle) => cycle.random(rng).into(),
            Cycle::ORI(cycle) => cycle.random(rng).into(),
            Cycle::SLT(cycle) => cycle.random(rng).into(),
            Cycle::SLTI(cycle) => cycle.random(rng).into(),
            Cycle::SLTIU(cycle) => cycle.random(rng).into(),
            Cycle::SLTU(cycle) => cycle.random(rng).into(),
            Cycle::SUB(cycle) => cycle.random(rng).into(),
            Cycle::SD(cycle) => cycle.random(rng).into(),
            Cycle::XOR(cycle) => cycle.random(rng).into(),
            Cycle::XORI(cycle) => cycle.random(rng).into(),
            Cycle::VirtualAdvice(cycle) => cycle.random(rng).into(),
            Cycle::VirtualAssertEQ(cycle) => cycle.random(rng).into(),
            Cycle::VirtualAssertHalfwordAlignment(cycle) => cycle.random(rng).into(),
            Cycle::VirtualAssertWordAlignment(cycle) => cycle.random(rng).into(),
            Cycle::VirtualAssertLTE(cycle) => cycle.random(rng).into(),
            Cycle::VirtualAssertValidDiv0(cycle) => cycle.random(rng).into(),
            Cycle::VirtualAssertValidUnsignedRemainder(cycle) => cycle.random(rng).into(),
            Cycle::VirtualMovsign(cycle) => cycle.random(rng).into(),
            Cycle::VirtualMULI(cycle) => cycle.random(rng).into(),
            Cycle::VirtualPow2(cycle) => cycle.random(rng).into(),
            Cycle::VirtualPow2I(cycle) => cycle.random(rng).into(),
            Cycle::VirtualPow2W(cycle) => cycle.random(rng).into(),
            Cycle::VirtualPow2IW(cycle) => cycle.random(rng).into(),
            Cycle::VirtualShiftRightBitmask(cycle) => cycle.random(rng).into(),
            Cycle::VirtualShiftRightBitmaskI(cycle) => cycle.random(rng).into(),
            Cycle::VirtualSRA(cycle) => cycle.random(rng).into(),
            Cycle::VirtualRev8W(cycle) => cycle.random(rng).into(),
            Cycle::VirtualSRAI(cycle) => cycle.random(rng).into(),
            Cycle::VirtualSRL(cycle) => cycle.random(rng).into(),
            Cycle::VirtualSRLI(cycle) => cycle.random(rng).into(),
            Cycle::VirtualZeroExtendWord(cycle) => cycle.random(rng).into(),
            Cycle::VirtualSignExtendWord(cycle) => cycle.random(rng).into(),
            Cycle::VirtualROTRI(cycle) => cycle.random(rng).into(),
            Cycle::VirtualROTRIW(cycle) => cycle.random(rng).into(),
            Cycle::VirtualChangeDivisor(cycle) => cycle.random(rng).into(),
            Cycle::VirtualChangeDivisorW(cycle) => cycle.random(rng).into(),
            Cycle::VirtualAssertMulUNoOverflow(cycle) => cycle.random(rng).into(),
            _ => Cycle::NoOp,
        }
    }

        fn test_read_raf_sumcheck(instruction: Option<Cycle>) {
            // 1. 设置随机数生成器种子，确保测试结果可复现
            let mut rng = StdRng::seed_from_u64(12345);

            // 2. 生成模拟的执行痕迹 (Trace)
            // trace 是一个包含 T 个 Cycle 的向量，每个 Cycle 代表一步指令执行。
            // 如果 instruction 参数是 Some，则全部生成该指令；如果是 None，则随机混合不同指令。
            let trace: Arc<Vec<_>> = Arc::new(
                (0..T)
                    .map(|_| random_instruction(&mut rng, &instruction))
                    .collect(),
            );

            // 3. 初始化 Transcript 和 Opening Accumulators
            // Prover 和 Verifier 各自拥有自己的 Transcript 以模拟交互式协议的随机性生成。
            // Accumulator 用于收集 Sumcheck 过程中产生的 Claim（声称值）。
            let prover_transcript = &mut Blake2bTranscript::new(&[]);
            let mut prover_opening_accumulator = ProverOpeningAccumulator::new(trace.len().log_2());
            let verifier_transcript = &mut Blake2bTranscript::new(&[]);
            let mut verifier_opening_accumulator = VerifierOpeningAccumulator::new(trace.len().log_2());

            // 4. 生成时间维度的随机挑战点 r_cycle (长度为 log(T))
            // 在正式协议中，这是上一轮 Sumcheck (Instruction Claim Reduction) 的输出。
            // 为了便于测试，这里直接从 Transcript 中模拟生成。
            // 注意：Prover 和 Verifier 必须使用相同的 Transcript 状态来获取相同的随机数。
            let r_cycle: Vec<<Fr as JoltField>::Challenge> =
                prover_transcript.challenge_vector_optimized::<Fr>(LOG_T);
            let _r_cycle: Vec<<Fr as JoltField>::Challenge> =
                verifier_transcript.challenge_vector_optimized::<Fr>(LOG_T);

            // 计算 Eq 多项式在 r_cycle 处的评估值，即 [eq(r_cycle, 0), ..., eq(r_cycle, T-1)]
            let eq_r_cycle = EqPolynomial::<Fr>::evals(&r_cycle);

            // 5. 手动计算预期的各个 Claim (Ground Truth)
            // 这一步模拟了 "Oracle" 的行为，通过直接遍历 trace 来计算正确的结果，用于后续校验。
            let mut rv_claim = Fr::zero();              // Lookup Output Claim (Sum(Eq * Output))
            let mut left_operand_claim = Fr::zero();    // 左操作数 Claim (Sum(Eq * Left))
            let mut right_operand_claim = Fr::zero();   // 右操作数 Claim (Sum(Eq * Right))

            for (i, cycle) in trace.iter().enumerate() {
                // 获取当前指令对应的查找键、表类型
                let lookup_index = LookupQuery::<XLEN>::to_lookup_index(cycle);
                let table: Option<LookupTables<XLEN>> = cycle.lookup_table();

                // 累加 Output Claim: 如果指令使用了查找表，则加上 Eq[i] * Table[Key]
                if let Some(table) = table {
                    rv_claim +=
                        JoltField::mul_u64(&eq_r_cycle[i], table.materialize_entry(lookup_index));
                }

                // 累加 Operand Claims: 左操作数和右操作数
                let (lo, ro) = LookupQuery::<XLEN>::to_lookup_operands(cycle);
                left_operand_claim += JoltField::mul_u64(&eq_r_cycle[i], lo);
                right_operand_claim += JoltField::mul_u128(&eq_r_cycle[i], ro);
            }

            // 6. 将计算出的 Claim 注入 Prover 的累加器
            // 模拟上一阶段 Sumcheck 输出的 Claim，Prover 本次只需要证明这些 Claim 是一致的。
            prover_opening_accumulator.append_virtual(
                prover_transcript,
                VirtualPolynomial::LookupOutput,
                SumcheckId::InstructionClaimReduction,
                OpeningPoint::new(r_cycle.clone()),
                rv_claim,
            );
            prover_opening_accumulator.append_virtual(
                prover_transcript,
                VirtualPolynomial::LeftLookupOperand,
                SumcheckId::InstructionClaimReduction,
                OpeningPoint::new(r_cycle.clone()),
                left_operand_claim,
            );
            prover_opening_accumulator.append_virtual(
                prover_transcript,
                VirtualPolynomial::RightLookupOperand,
                SumcheckId::InstructionClaimReduction,
                OpeningPoint::new(r_cycle.clone()),
                right_operand_claim,
            );
            // SpartanProductVirtualization 也是产生 LookupOutput 的 Claim 来源之一
            prover_opening_accumulator.append_virtual(
                prover_transcript,
                VirtualPolynomial::LookupOutput,
                SumcheckId::SpartanProductVirtualization,
                OpeningPoint::new(r_cycle.clone()),
                rv_claim,
            );

            // 7. 初始化 Prover
            // 配置 One-Hot 参数（这里仅用于配置结构，具体值在 InstructionReadRafContext 下不太关键）
            let one_hot_params = OneHotParams::new(trace.len().log_2(), 100, 100);

            // 创建 Sumcheck 参数，这会从 accumulator 中读取 r_reduction，并从 transcript 生成 gamma
            let params = InstructionReadRafSumcheckParams::new(
                trace.len().log_2(),
                &one_hot_params,
                &prover_opening_accumulator,
                prover_transcript,
            );

            // 核心初始化：Prover 会在这里进行 Trace 预处理、表聚合、权重计算等繁重工作
            let mut prover_sumcheck =
                InstructionReadRafSumcheckProver::initialize(params, Arc::clone(&trace));

            // 8. 执行 Prover 端的 Sumcheck 协议
            // 生成多项式承诺证明 proof，以及最终的随机挑战点 r_sumcheck
            let (proof, r_sumcheck) = BatchedSumcheck::prove(
                vec![&mut prover_sumcheck],
                &mut prover_opening_accumulator,
                prover_transcript,
            );

            // 9. 准备 Verifier 环境
            // 获取 Prover 生成的 Claim 值，传递给 Verifier 累加器（模拟网络传输）
            // 实际上 Verifier 并不知晓具体值是怎么算的，只知道 Prover 承诺了这些值。
            for (key, (_, value)) in &prover_opening_accumulator.openings {
                let empty_point = OpeningPoint::<BIG_ENDIAN, Fr>::new(vec![]);
                verifier_opening_accumulator
                    .openings
                    .insert(*key, (empty_point, *value));
            }

            // Verifier 需要知道要在哪些点上查询哪些多项式，这里注册这些“意图”
            verifier_opening_accumulator.append_virtual(
                verifier_transcript,
                VirtualPolynomial::LookupOutput,
                SumcheckId::InstructionClaimReduction,
                OpeningPoint::new(r_cycle.clone()),
            );
            verifier_opening_accumulator.append_virtual(
                verifier_transcript,
                VirtualPolynomial::LeftLookupOperand,
                SumcheckId::InstructionClaimReduction,
                OpeningPoint::new(r_cycle.clone()),
            );
            verifier_opening_accumulator.append_virtual(
                verifier_transcript,
                VirtualPolynomial::RightLookupOperand,
                SumcheckId::InstructionClaimReduction,
                OpeningPoint::new(r_cycle.clone()),
            );
            verifier_opening_accumulator.append_virtual(
                verifier_transcript,
                VirtualPolynomial::LookupOutput,
                SumcheckId::SpartanProductVirtualization,
                OpeningPoint::new(r_cycle.clone()),
            );

            // 10. 初始化 Verifier
            let mut verifier_sumcheck = InstructionReadRafSumcheckVerifier::new(
                trace.len().log_2(),
                &one_hot_params,
                &verifier_opening_accumulator,
                verifier_transcript,
            );

            // 11. 执行 Verifier 端的验证逻辑
            // 验证 proof 是否合法，计算最终用于 Open 操作的 challenge 点。
            let r_sumcheck_verif = BatchedSumcheck::verify(
                &proof,
                vec![&mut verifier_sumcheck],
                &mut verifier_opening_accumulator,
                verifier_transcript,
            )
                .unwrap();

            // 12. 最终断言
            // 确保 Prover 和 Verifier 协商出的最终随机点一致，这意味着协议交互流程正确无误。
            assert_eq!(r_sumcheck, r_sumcheck_verif);
        }

    #[test]
    fn test_random_instructions() {
        test_read_raf_sumcheck(None);
    }

    #[test]
    fn test_add() {
        test_read_raf_sumcheck(Some(Cycle::ADD(Default::default())));
    }

    #[test]
    fn test_addi() {
        test_read_raf_sumcheck(Some(Cycle::ADDI(Default::default())));
    }

    #[test]
    fn test_and() {
        test_read_raf_sumcheck(Some(Cycle::AND(Default::default())));
    }

    #[test]
    fn test_andn() {
        test_read_raf_sumcheck(Some(Cycle::ANDN(Default::default())));
    }

    #[test]
    fn test_andi() {
        test_read_raf_sumcheck(Some(Cycle::ANDI(Default::default())));
    }

    #[test]
    fn test_auipc() {
        test_read_raf_sumcheck(Some(Cycle::AUIPC(Default::default())));
    }

    #[test]
    fn test_beq() {
        test_read_raf_sumcheck(Some(Cycle::BEQ(Default::default())));
    }

    #[test]
    fn test_bge() {
        test_read_raf_sumcheck(Some(Cycle::BGE(Default::default())));
    }

    #[test]
    fn test_bgeu() {
        test_read_raf_sumcheck(Some(Cycle::BGEU(Default::default())));
    }

    #[test]
    fn test_blt() {
        test_read_raf_sumcheck(Some(Cycle::BLT(Default::default())));
    }

    #[test]
    fn test_bltu() {
        test_read_raf_sumcheck(Some(Cycle::BLTU(Default::default())));
    }

    #[test]
    fn test_bne() {
        test_read_raf_sumcheck(Some(Cycle::BNE(Default::default())));
    }

    #[test]
    fn test_fence() {
        test_read_raf_sumcheck(Some(Cycle::FENCE(Default::default())));
    }

    #[test]
    fn test_jal() {
        test_read_raf_sumcheck(Some(Cycle::JAL(Default::default())));
    }

    #[test]
    fn test_jalr() {
        test_read_raf_sumcheck(Some(Cycle::JALR(Default::default())));
    }

    #[test]
    fn test_lui() {
        test_read_raf_sumcheck(Some(Cycle::LUI(Default::default())));
    }

    #[test]
    fn test_ld() {
        test_read_raf_sumcheck(Some(Cycle::LD(Default::default())));
    }

    #[test]
    fn test_mul() {
        test_read_raf_sumcheck(Some(Cycle::MUL(Default::default())));
    }

    #[test]
    fn test_mulhu() {
        test_read_raf_sumcheck(Some(Cycle::MULHU(Default::default())));
    }

    #[test]
    fn test_or() {
        test_read_raf_sumcheck(Some(Cycle::OR(Default::default())));
    }

    #[test]
    fn test_ori() {
        test_read_raf_sumcheck(Some(Cycle::ORI(Default::default())));
    }

    #[test]
    fn test_slt() {
        test_read_raf_sumcheck(Some(Cycle::SLT(Default::default())));
    }

    #[test]
    fn test_slti() {
        test_read_raf_sumcheck(Some(Cycle::SLTI(Default::default())));
    }

    #[test]
    fn test_sltiu() {
        test_read_raf_sumcheck(Some(Cycle::SLTIU(Default::default())));
    }

    #[test]
    fn test_sltu() {
        test_read_raf_sumcheck(Some(Cycle::SLTU(Default::default())));
    }

    #[test]
    fn test_sub() {
        test_read_raf_sumcheck(Some(Cycle::SUB(Default::default())));
    }

    #[test]
    fn test_sd() {
        test_read_raf_sumcheck(Some(Cycle::SD(Default::default())));
    }

    #[test]
    fn test_xor() {
        test_read_raf_sumcheck(Some(Cycle::XOR(Default::default())));
    }

    #[test]
    fn test_xori() {
        test_read_raf_sumcheck(Some(Cycle::XORI(Default::default())));
    }

    #[test]
    fn test_advice() {
        test_read_raf_sumcheck(Some(Cycle::VirtualAdvice(Default::default())));
    }

    #[test]
    fn test_asserteq() {
        test_read_raf_sumcheck(Some(Cycle::VirtualAssertEQ(Default::default())));
    }

    #[test]
    fn test_asserthalfwordalignment() {
        test_read_raf_sumcheck(Some(Cycle::VirtualAssertHalfwordAlignment(
            Default::default(),
        )));
    }

    #[test]
    fn test_assertwordalignment() {
        test_read_raf_sumcheck(Some(Cycle::VirtualAssertWordAlignment(Default::default())));
    }

    #[test]
    fn test_assertlte() {
        test_read_raf_sumcheck(Some(Cycle::VirtualAssertLTE(Default::default())));
    }

    #[test]
    fn test_assertvaliddiv0() {
        test_read_raf_sumcheck(Some(Cycle::VirtualAssertValidDiv0(Default::default())));
    }

    #[test]
    fn test_assertvalidunsignedremainder() {
        test_read_raf_sumcheck(Some(Cycle::VirtualAssertValidUnsignedRemainder(
            Default::default(),
        )));
    }

    #[test]
    fn test_movsign() {
        test_read_raf_sumcheck(Some(Cycle::VirtualMovsign(Default::default())));
    }

    #[test]
    fn test_muli() {
        test_read_raf_sumcheck(Some(Cycle::VirtualMULI(Default::default())));
    }

    #[test]
    fn test_pow2() {
        test_read_raf_sumcheck(Some(Cycle::VirtualPow2(Default::default())));
    }

    #[test]
    fn test_pow2i() {
        test_read_raf_sumcheck(Some(Cycle::VirtualPow2I(Default::default())));
    }

    #[test]
    fn test_pow2w() {
        test_read_raf_sumcheck(Some(Cycle::VirtualPow2W(Default::default())));
    }

    #[test]
    fn test_pow2iw() {
        test_read_raf_sumcheck(Some(Cycle::VirtualPow2IW(Default::default())));
    }

    #[test]
    fn test_shiftrightbitmask() {
        test_read_raf_sumcheck(Some(Cycle::VirtualShiftRightBitmask(Default::default())));
    }

    #[test]
    fn test_shiftrightbitmaski() {
        test_read_raf_sumcheck(Some(Cycle::VirtualShiftRightBitmaskI(Default::default())));
    }

    #[test]
    fn test_virtualrotri() {
        test_read_raf_sumcheck(Some(Cycle::VirtualROTRI(Default::default())));
    }

    #[test]
    fn test_virtualrotriw() {
        test_read_raf_sumcheck(Some(Cycle::VirtualROTRIW(Default::default())));
    }

    #[test]
    fn test_virtualsra() {
        test_read_raf_sumcheck(Some(Cycle::VirtualSRA(Default::default())));
    }

    #[test]
    fn test_virtualsrai() {
        test_read_raf_sumcheck(Some(Cycle::VirtualSRAI(Default::default())));
    }

    #[test]
    fn test_virtualrev8w() {
        test_read_raf_sumcheck(Some(Cycle::VirtualRev8W(Default::default())));
    }

    #[test]
    fn test_virtualsrl() {
        test_read_raf_sumcheck(Some(Cycle::VirtualSRL(Default::default())));
    }

    #[test]
    fn test_virtualsrli() {
        test_read_raf_sumcheck(Some(Cycle::VirtualSRLI(Default::default())));
    }

    #[test]
    fn test_virtualextend() {
        test_read_raf_sumcheck(Some(Cycle::VirtualZeroExtendWord(Default::default())));
    }

    #[test]
    fn test_virtualsignextend() {
        test_read_raf_sumcheck(Some(Cycle::VirtualSignExtendWord(Default::default())));
    }

    #[test]
    fn test_virtualchangedivisor() {
        test_read_raf_sumcheck(Some(Cycle::VirtualChangeDivisor(Default::default())));
    }

    #[test]
    fn test_virtualchangedivisorw() {
        test_read_raf_sumcheck(Some(Cycle::VirtualChangeDivisorW(Default::default())));
    }

    #[test]
    fn test_virtualassertmulnooverflow() {
        test_read_raf_sumcheck(Some(Cycle::VirtualAssertMulUNoOverflow(Default::default())));
    }
}
