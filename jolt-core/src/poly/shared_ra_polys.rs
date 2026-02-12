//! Shared utilities for RA (read-address) polynomials across all families.
//!
//! This module provides efficient computation of RA indices and G evaluations
//! that are shared across instruction, bytecode, and RAM polynomial families.
//!
//! ## Design Goals
//!
//! 1. **Single-pass trace iteration**: Compute all indices for all families in one pass
//! 2. **Cache locality**: All RA polynomials share the same eq table structure
//! 3. **Configurable delay binding**: Support delaying materialization for multiple rounds
//!
//! ## Two-Phase Architecture
//!
//! - **Phase 1**: Store shared eq table(s) and RA indices (compact representation)
//! - **Phase 2**: Materialize RA multilinear polynomials when needed
//!
//! ## SharedRaPolynomials
//!
//! Instead of storing N separate `RaPolynomial` each with their own eq table copy,
//! `SharedRaPolynomials` stores:
//! - One (small) eq table per polynomial (size K each; K is 16 or 256 in practice)
//! - A single `Vec<RaIndices>` (size T, non-transposed) shared by all polynomials
//!
//! This saves memory and improves cache locality when iterating through cycles.

use allocative::Allocative;
use ark_std::Zero;
use fixedbitset::FixedBitSet;

use crate::field::JoltField;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding};
use crate::utils::thread::drop_in_background_thread;
use crate::utils::thread::unsafe_allocate_zero_vec;
use crate::zkvm::bytecode::BytecodePreprocessing;
use crate::zkvm::config::OneHotParams;
use crate::zkvm::instruction::LookupQuery;
use crate::zkvm::ram::remap_address;
use common::constants::XLEN;
use common::jolt_device::MemoryLayout;
use rayon::prelude::*;
use tracing::info;
use tracer::instruction::Cycle;

/// Maximum number of instruction RA chunks (lookup index splits into at most 32 chunks)
pub const MAX_INSTRUCTION_D: usize = 32;
/// Maximum number of bytecode RA chunks (PC splits into at most 6 chunks)
pub const MAX_BYTECODE_D: usize = 6;
/// Maximum number of RAM RA chunks (address splits into at most 8 chunks)
pub const MAX_RAM_D: usize = 8;

/// Asserts that the one_hot_params dimensions are within bounds.
/// Call this once at the start of bulk operations to catch issues early.
#[inline]
pub fn assert_ra_bounds(one_hot_params: &OneHotParams) {
    assert!(
        one_hot_params.instruction_d <= MAX_INSTRUCTION_D,
        "instruction_d {} exceeds MAX_INSTRUCTION_D {}",
        one_hot_params.instruction_d,
        MAX_INSTRUCTION_D
    );
    assert!(
        one_hot_params.bytecode_d <= MAX_BYTECODE_D,
        "bytecode_d {} exceeds MAX_BYTECODE_D {}",
        one_hot_params.bytecode_d,
        MAX_BYTECODE_D
    );
    assert!(
        one_hot_params.ram_d <= MAX_RAM_D,
        "ram_d {} exceeds MAX_RAM_D {}",
        one_hot_params.ram_d,
        MAX_RAM_D
    );
}

/// Stores all RA chunk indices for a single cycle.
/// Uses fixed-size arrays to avoid heap allocation in hot loops.
#[derive(Clone, Copy, Default, Allocative)]
pub struct RaIndices {
    /// Instruction RA chunk indices (always present)
    pub instruction: [u8; MAX_INSTRUCTION_D],
    /// Bytecode RA chunk indices (always present)
    pub bytecode: [u8; MAX_BYTECODE_D],
    /// RAM RA chunk indices (None for non-memory cycles)
    pub ram: [Option<u8>; MAX_RAM_D],
}

impl std::ops::Add for RaIndices {
    type Output = Self;

    fn add(self, _rhs: Self) -> Self::Output {
        // This is only implemented to satisfy the Zero trait bound.
        // RaIndices should never actually be added together.
        unimplemented!("RaIndices::add is not meaningful; this impl exists only for Zero trait")
    }
}

/// Implement Zero trait for RaIndices to satisfy the trait bound for `unsafe_allocate_zero_vec`
impl Zero for RaIndices {
    fn zero() -> Self {
        // `unsafe_allocate_zero_vec` relies on the invariant that `Zero::zero()` is represented
        // by all-zero bytes. Constructing `[None; N]` can leave padding / unused enum payload
        // bytes uninitialized, which breaks that invariant (and is UB to inspect as bytes).
        //
        // All-zero is a valid bit-pattern for `RaIndices` (arrays of integers + `Option<u8>`),
        // so this is safe here.
        unsafe { core::mem::zeroed() }
    }

    fn is_zero(&self) -> bool {
        self.instruction.iter().all(|&x| x == 0)
            && self.bytecode.iter().all(|&x| x == 0)
            && self.ram.iter().all(|x| x.is_none())
    }
}

impl RaIndices {
    /// Compute all RA chunk indices for a single cycle.
    #[inline]
    /// 将一个 CPU 执行周期 (Cycle) 的数据转换为用于查表证明的索引切片 (RaIndices)
    pub fn from_cycle(
        cycle: &Cycle,                   // 输入: 当前 CPU 周期的完整状态 (寄存器、内存、指令等)
        bytecode: &BytecodePreprocessing,// 输入: 预处理的字节码信息 (用于快速查找 PC 等)
        memory_layout: &MemoryLayout,    // 输入: 内存布局配置 (用于物理地址到证明地址的重映射)
        one_hot_params: &OneHotParams,   // 输入: 切分参数 (定义了切成几块、每块多大)
    ) -> Self {
        // info!("one_hot_params={:#?}",one_hot_params);
        /*
        one_hot_params=OneHotParams {
            log_k_chunk: 4,
            lookups_ra_virtual_log_k_chunk: 16,
            k_chunk: 16,
            bytecode_k: 2048,
            ram_k: 8192,
            instruction_d: 32,
            bytecode_d: 3,
            ram_d: 4,
            instruction_shifts:[124,120,116,112,108,104,100,96,92,88,84,80,76,72,68,64,60,56,52,48,44,40,36,32,28,24,20,16,12,8,4,0],
            ram_shifts: [12,8,4,0],
            bytecode_shifts: [8,4,0,
        }

         */
        // =========================================================
        // 1. 安全性断言 (Bounds Check)
        // ---------------------------------------------------------
        // 确保参数中要求的切分数量 (d) 没有超过硬编码的数组上限 (MAX_..._D)。
        // 这样做是为了避免在每一步都进行动态内存分配 (Vec)，而是使用栈上的固定大小数组，
        // 从而极大提升性能 (Jolt 对性能极其敏感)。
        // =========================================================
        debug_assert!(
            one_hot_params.instruction_d <= MAX_INSTRUCTION_D,
            "instruction_d {} exceeds MAX_INSTRUCTION_D {}",
            one_hot_params.instruction_d,
            MAX_INSTRUCTION_D
        );
        debug_assert!(
            one_hot_params.bytecode_d <= MAX_BYTECODE_D,
            "bytecode_d {} exceeds MAX_BYTECODE_D {}",
            one_hot_params.bytecode_d,
            MAX_BYTECODE_D
        );
        debug_assert!(
            one_hot_params.ram_d <= MAX_RAM_D,
            "ram_d {} exceeds MAX_RAM_D {}",
            one_hot_params.ram_d,
            MAX_RAM_D
        );

        // =========================================================
        // 2. 提取指令运算索引 (Instruction Indices)
        // ---------------------------------------------------------
        // 目的: 获取用于 ALU 查表的操作数切片。
        // 逻辑:
        // a. to_lookup_index: 将指令的操作数 (如 rs1, rs2, rd) 组合成一个大整数 "Lookup Index"。
        //    例如对于 ADD 指令，Index = rs1 || rs2 (拼接)。
        // b. 循环切分: 按照 one_hot_params 定义的规则，将大整数切成多个 8-bit 小块。
        // =========================================================
        let lookup_index = LookupQuery::<XLEN>::to_lookup_index(cycle);
        // info!("lookup_index={:#?}", lookup_index);
        let mut instruction = [0u8; MAX_INSTRUCTION_D]; // 初始化固定大小数组
        for i in 0..one_hot_params.instruction_d {
            // 提取第 i 个切片 (Chunk i)
            instruction[i] = one_hot_params.lookup_index_chunk(lookup_index, i);
        }

        // =========================================================
        // 3. 提取字节码/ROM 索引 (Bytecode Indices)
        // ---------------------------------------------------------
        // 目的: 获取用于证明 "代码一致性" 的 PC 切片。
        // 逻辑:
        // a. get_pc: 获取当前指令的程序计数器 PC。
        // b. 循环切分: 将 PC (可能还有指令编码) 切分成小块。
        //    这用于后续证明 "我在 PC 处执行的指令确实是 ROM[PC]"。
        // =========================================================
        let pc = bytecode.get_pc(cycle);
        // info!("pc={:#?}", pc);
        let mut bytecode_arr = [0u8; MAX_BYTECODE_D];
        for i in 0..one_hot_params.bytecode_d {
            bytecode_arr[i] = one_hot_params.bytecode_pc_chunk(pc, i);
        }
        // info!("bytecode_arr={:#?}", bytecode_arr);

        // =========================================================
        // 4. 提取 RAM 内存索引 (RAM Indices)
        // ---------------------------------------------------------
        // 目的: 获取用于内存读写证明的地址切片。
        // 逻辑:
        // a. 获取原始物理地址 (address)。
        // b. 重映射 (remap): 将物理地址转换为证明系统内部使用的虚拟地址 (Canonical Address)。
        //    如果指令不访问内存 (如 ADD)，remap 可能会返回 None 或特定值。
        // c. 循环切分: 如果地址存在 (Some)，则切分；否则填 None。
        //    None 在后续 G 表统计中会被忽略。
        // =========================================================
        let address = cycle.ram_access().address() as u64;
        let remapped = remap_address(address, memory_layout);
        let mut ram = [None; MAX_RAM_D];
        for i in 0..one_hot_params.ram_d {
            // map 处理 Option: 如果 remapped 是 Some(addr)，则执行闭包切分；如果是 None，则返回 None。
            ram[i] = remapped.map(|a| one_hot_params.ram_address_chunk(a, i));
        }

        // =========================================================
        // 5. 返回结果
        // ---------------------------------------------------------
        // 将三类切片打包返回。这个结构体将被放入 G 表统计器中。
        // =========================================================
        Self {
            instruction,
            bytecode: bytecode_arr,
            ram,
        }
    }

    /// Extract the index for polynomial `poly_idx` in the unified ordering:
    /// [instruction_0..d, bytecode_0..d, ram_0..d]
    #[inline]
    pub fn get_index(&self, poly_idx: usize, one_hot_params: &OneHotParams) -> Option<u8> {
        let instruction_d = one_hot_params.instruction_d;
        let bytecode_d = one_hot_params.bytecode_d;

        if poly_idx < instruction_d {
            Some(self.instruction[poly_idx])
        } else if poly_idx < instruction_d + bytecode_d {
            Some(self.bytecode[poly_idx - instruction_d])
        } else {
            self.ram[poly_idx - instruction_d - bytecode_d]
        }
    }
}

/// Compute all G evaluations for all families in parallel using split-eq optimization.
///
/// G_i(k) = Σ_j eq(r_cycle, j) · ra_i(k, j)
///
/// For one-hot RA polynomials, this simplifies to:
/// G_i(k) = Σ_{j: chunk_i(j) = k} eq_r_cycle[j]
///
/// Uses a two-table split-eq: split `r_cycle` into MSB/LSB halves, compute `E_hi` and `E_lo`,
/// then `eq(r_cycle, c) = E_hi[c_hi] * E_lo[c_lo]` where `c = (c_hi << lo_bits) | c_lo`.
///
/// Returns G in order: [instruction_0..d, bytecode_0..d, ram_0..d]
/// Each inner Vec has length k_chunk.
#[tracing::instrument(skip_all, name = "shared_ra_polys::compute_all_G")]
pub fn compute_all_G<F: JoltField>(
    trace: &[Cycle],
    bytecode: &BytecodePreprocessing,
    memory_layout: &MemoryLayout,
    one_hot_params: &OneHotParams,
    r_cycle: &[F::Challenge],
) -> Vec<Vec<F>> {
    compute_all_G_impl::<F>(
        trace,
        bytecode,
        memory_layout,
        one_hot_params,
        r_cycle,
        None,
    )
}

/// Compute all G evaluations AND RA indices in a single pass over the trace.
///
/// This avoids traversing the trace twice when both G and ra_indices are needed.
///
/// Returns (G, ra_indices) where:
/// - G[i] = pushforward of ra_i over r_cycle (length k_chunk each)
/// - ra_indices[j] = RA chunk indices for cycle j
#[tracing::instrument(skip_all, name = "shared_ra_polys::compute_all_G_and_ra_indices")]
pub fn compute_all_G_and_ra_indices<F: JoltField>(
    trace: &[Cycle],
    bytecode: &BytecodePreprocessing,
    memory_layout: &MemoryLayout,
    one_hot_params: &OneHotParams,
    r_cycle: &[F::Challenge],
) -> (Vec<Vec<F>>, Vec<RaIndices>) {
    let T = trace.len();
    // 预分配用于存储每一步 Trace 对应的索引数据的向量 (RaIndices)。
    // 使用 unsafe_allocate_zero_vec 避免默认初始化的开销，
    // 因为 RaIndices 的零值位模式是合法的（只要不包含非法的 enum tag）。
    let mut ra_indices: Vec<RaIndices> = unsafe_allocate_zero_vec(T);

    // 调用核心实现函数。
    // 关键点：传入了 Some(&mut ra_indices)。
    // 这指示 impl 函数在遍历 Trace 计算 G 值（加权和）的同时，
    // 将解析出的指令/内存 chunk 索引直接写入到 ra_indices 中。
    // 这种“单次遍历 (Single-pass)”设计避免了为了获取索引而对巨大的 Trace 进行第二次遍历。
    let G = compute_all_G_impl::<F>(
        trace,
        bytecode,
        memory_layout,
        one_hot_params,
        r_cycle,
        Some(&mut ra_indices),
    );

    // 返回计算出的多项式评估值 G 和 填充好的索引数据
    (G, ra_indices)
}

/// Core implementation for computing G evaluations.
///
/// When `ra_indices` is `Some`, also writes RaIndices to the provided slice.
/// This is safe because each cycle index is visited exactly once (disjoint writes).
#[inline(always)]
fn compute_all_G_impl<F: JoltField>(
    trace: &[Cycle],                   // 执行痕迹
    bytecode: &BytecodePreprocessing,  // 预处理字节码
    memory_layout: &MemoryLayout,      // 内存布局
    one_hot_params: &OneHotParams,     // 切片参数 (如 K=256)
    r_cycle: &[F::Challenge],          // Sumcheck 时间维度的随机挑战点
    ra_indices: Option<&mut [RaIndices]>, // (可选) 用于输出提取出的切片索引
) -> Vec<Vec<F>> { // 返回计算好的 G 表: G[poly_idx][value_k]

    // ----------------------------------------------------------------
    // 1. 线程安全准备
    // ----------------------------------------------------------------
    // 将可变引用转换为 usize (原始指针地址)，以便在 Rayon 并行闭包中传递。
    // Rust 的安全机制通常禁止跨线程共享可变引用，但这里我们会手动保证互斥写入。
    let ra_ptr_usize: usize = ra_indices.map(|s| s.as_mut_ptr() as usize).unwrap_or(0);
    // 检查参数边界
    assert_ra_bounds(one_hot_params);

    let K = one_hot_params.k_chunk; // 值域大小 (通常 256)
    // 各类数据的切片数量
    let instruction_d = one_hot_params.instruction_d;
    let bytecode_d = one_hot_params.bytecode_d;
    let ram_d = one_hot_params.ram_d;
    let N = instruction_d + bytecode_d + ram_d; // 总多项式数量
    let T = trace.len(); // Trace 长度

    // ----------------------------------------------------------------
    // 2. Split-Eq 准备 (双重求和优化)
    // ----------------------------------------------------------------
    // 将随机点 r_cycle 拆分为高位和低位，分别计算 Eq 表。
    // 这是为了把 O(T) 次乘法转化为 O(sqrt(T)) 次乘法。
    let log_T = r_cycle.len();
    let lo_bits = log_T / 2;
    let hi_bits = log_T - lo_bits;
    let (r_hi, r_lo) = r_cycle.split_at(hi_bits);

    // 并行计算 E_hi 和 E_lo
    // E_lo 大小约 sqrt(T)，E_hi 大小约 sqrt(T)
    let (E_hi, E_lo) = rayon::join(
        || EqPolynomial::<F>::evals(r_hi),
        || EqPolynomial::<F>::evals(r_lo),
    );

    let in_len = E_lo.len(); // 内层循环长度 (2^lo_bits)

    // ----------------------------------------------------------------
    // 3. 并行 Map-Reduce 核心逻辑
    // ----------------------------------------------------------------
    let num_threads = rayon::current_num_threads();
    let out_len = E_hi.len(); // 外层循环长度
    let chunk_size = out_len.div_ceil(num_threads);

    // 对 E_hi 进行分块，每个线程处理一块 E_hi (即一批高位时间)
    E_hi.par_chunks(chunk_size)
        .enumerate()
        .map(|(chunk_idx, chunk)| {
            // --- 线程局部变量分配 ---

            // partial_*: 存储最终加权结果 (需要乘法)
            // 初始化为 0
            let mut partial_instruction: Vec<Vec<F>> = (0..instruction_d)
                .map(|_| unsafe_allocate_zero_vec(K))
                .collect();
            // ... (bytecode 和 ram 的分配省略，同上)
            let mut partial_bytecode: Vec<Vec<F>> = (0..bytecode_d)
                .map(|_| unsafe_allocate_zero_vec(K))
                .collect();
            let mut partial_ram: Vec<Vec<F>> =
                (0..ram_d).map(|_| unsafe_allocate_zero_vec(K)).collect();

            // local_*: 存储内层循环的累加结果 (只做加法)
            // 使用 Unreduced<5> 类型：这是关键优化，允许累加 5 次才做一次取模，减少开销。
            let mut local_instruction: Vec<Vec<F::Unreduced<5>>> = (0..instruction_d)
                .map(|_| unsafe_allocate_zero_vec(K))
                .collect();
            // ... (同上)
            let mut local_bytecode: Vec<Vec<F::Unreduced<5>>> = (0..bytecode_d)
                .map(|_| unsafe_allocate_zero_vec(K))
                .collect();
            let mut local_ram: Vec<Vec<F::Unreduced<5>>> =
                (0..ram_d).map(|_| unsafe_allocate_zero_vec(K)).collect();

            // touched_*: 稀疏集 (Sparse Set)
            // 记录哪些值 k 在当前循环中被访问过，避免遍历整个大小为 K 的数组来清零。
            let mut touched_instruction: Vec<FixedBitSet> =
                vec![FixedBitSet::with_capacity(K); instruction_d];
            // ...
            let mut touched_bytecode: Vec<FixedBitSet> =
                vec![FixedBitSet::with_capacity(K); bytecode_d];
            let mut touched_ram: Vec<FixedBitSet> = vec![FixedBitSet::with_capacity(K); ram_d];

            let chunk_start = chunk_idx * chunk_size;

            // --- 遍历外层块 (High Bits of Time) ---
            for (local_idx, &e_hi) in chunk.iter().enumerate() {
                let c_hi = chunk_start + local_idx; // 当前高位时间索引
                let c_hi_base = c_hi * in_len;      // 当前块的起始绝对时间

                // 3.1 清理上一轮的脏数据 (Reset accumulators)
                // 利用 touched set 只清空非零项，极快。
                for i in 0..instruction_d {
                    for k in touched_instruction[i].ones() {
                        local_instruction[i][k] = Default::default();
                    }
                    touched_instruction[i].clear();
                }
                // ... (bytecode/ram 清理代码略)
                for i in 0..bytecode_d {
                    for k in touched_bytecode[i].ones() {
                        local_bytecode[i][k] = Default::default();
                    }
                    touched_bytecode[i].clear();
                }
                for i in 0..ram_d {
                    for k in touched_ram[i].ones() {
                        local_ram[i][k] = Default::default();
                    }
                    touched_ram[i].clear();
                }

                // 3.2 内层循环 (Low Bits of Time) - 只有加法！
                for c_lo in 0..in_len {
                    let j = c_hi_base + c_lo; // 绝对时间索引 j
                    if j >= T {
                        break;
                    }

                    // 获取 E_lo[c_lo] 的值 (权重)
                    let add = *E_lo[c_lo].as_unreduced_ref();

                    // 从 Trace 中提取所有切片索引 (关键逻辑)
                    // RaIndices 包含：指令切片、字节码切片、RAM切片
                    let ra_idx =
                        RaIndices::from_cycle(&trace[j], bytecode, memory_layout, one_hot_params);

                    // 如果需要，保存提取出的索引 (Phase 2 用)
                    if ra_ptr_usize != 0 {
                        // SAFETY: j 在所有并行线程中是唯一的，不会发生数据竞争。
                        unsafe {
                            let ra_ptr = ra_ptr_usize as *mut RaIndices;
                            *ra_ptr.add(j) = ra_idx;
                        }
                    }

                    // --- 累加权重到对应的桶 (Binning) ---

                    // 处理 Instruction Chunks
                    for i in 0..instruction_d {
                        let k = ra_idx.instruction[i] as usize; // 切片值 k
                        if !touched_instruction[i].contains(k) {
                            touched_instruction[i].insert(k); // 标记 k 被触碰
                        }
                        local_instruction[i][k] += add; // 累加 E_lo
                    }

                    // 处理 Bytecode Chunks
                    for i in 0..bytecode_d {
                        let k = ra_idx.bytecode[i] as usize;
                        if !touched_bytecode[i].contains(k) {
                            touched_bytecode[i].insert(k);
                        }
                        local_bytecode[i][k] += add;
                    }

                    // 处理 RAM Chunks (可能有 None，表示无 RAM 操作)
                    for i in 0..ram_d {
                        if let Some(k) = ra_idx.ram[i] {
                            let k = k as usize;
                            if !touched_ram[i].contains(k) {
                                touched_ram[i].insert(k);
                            }
                            local_ram[i][k] += add;
                        }
                    }
                }

                // 3.3 外层折叠 (High Bits Contribution) - 只有这里做乘法！
                // 遍历刚才内层循环触碰过的所有 k
                for i in 0..instruction_d {
                    for k in touched_instruction[i].ones() {
                        // Barrett Reduction: 将累加了多次的大数取模归一化
                        let reduced = F::from_barrett_reduce::<5>(local_instruction[i][k]);
                        // 核心公式: G[k] += E_hi * sum(E_lo)
                        partial_instruction[i][k] += e_hi * reduced;
                    }
                }
                // ... (bytecode/ram 折叠代码略，逻辑同上)
                for i in 0..bytecode_d {
                    for k in touched_bytecode[i].ones() {
                        let reduced = F::from_barrett_reduce::<5>(local_bytecode[i][k]);
                        partial_bytecode[i][k] += e_hi * reduced;
                    }
                }
                for i in 0..ram_d {
                    for k in touched_ram[i].ones() {
                        let reduced = F::from_barrett_reduce::<5>(local_ram[i][k]);
                        partial_ram[i][k] += e_hi * reduced;
                    }
                }
            }

            // 合并当前线程的结果
            let mut result: Vec<Vec<F>> = Vec::with_capacity(N);
            result.extend(partial_instruction);
            result.extend(partial_bytecode);
            result.extend(partial_ram);
            result
        })
        // ----------------------------------------------------------------
        // 4. Reduce 阶段
        // ----------------------------------------------------------------
        // 将所有并行线程计算出的 partial_G 向量相加，得到全局唯一的 G 表
        .reduce(
            || (0..N).map(|_| unsafe_allocate_zero_vec::<F>(K)).collect(),
            |mut a, b| {
                for (a_poly, b_poly) in a.iter_mut().zip(b.iter()) {
                    a_poly
                        .par_iter_mut()
                        .zip(b_poly.par_iter())
                        .for_each(|(a_val, b_val)| *a_val += *b_val);
                }
                a
            },
        )
}

/// Shared RA polynomials that use a single eq table for all polynomials.
///
/// Instead of N separate `RaPolynomial` each with their own eq table copy,
/// this stores:
/// - ONE (small) eq table per polynomial (or split tables for later rounds)
/// - `Vec<RaIndices>` (size T, non-transposed)
///
/// This saves memory and improves cache locality.
#[derive(Allocative)]
pub enum SharedRaPolynomials<F: JoltField> {
    /// Round 1: Single shared eq table
    Round1(SharedRaRound1<F>),
    /// Round 2: Split into F_0, F_1
    Round2(SharedRaRound2<F>),
    /// Round 3: Split into F_00, F_01, F_10, F_11
    Round3(SharedRaRound3<F>),
    /// Round N: Fully materialized multilinear polynomials
    RoundN(Vec<MultilinearPolynomial<F>>),
}

/// Round 1 state: single shared eq table
#[derive(Allocative, Default)]
pub struct SharedRaRound1<F: JoltField> {
    /// Per-polynomial eq tables: tables[poly_idx][k] for k in 0..K
    ///
    /// In the booleanity sumcheck, these tables may already be pre-scaled by a per-polynomial
    /// constant (e.g. a batching coefficient).
    tables: Vec<Vec<F>>,
    /// RA indices for all cycles (non-transposed)
    indices: Vec<RaIndices>,
    /// Number of polynomials
    num_polys: usize,
    /// OneHotParams for index extraction
    #[allocative(skip)]
    one_hot_params: OneHotParams,
}

/// Round 2 state: split eq tables
#[derive(Allocative, Default)]
pub struct SharedRaRound2<F: JoltField> {
    /// Per-polynomial tables for the 0-branch: tables_0[poly_idx][k]
    tables_0: Vec<Vec<F>>,
    /// Per-polynomial tables for the 1-branch: tables_1[poly_idx][k]
    tables_1: Vec<Vec<F>>,
    /// RA indices for all cycles
    indices: Vec<RaIndices>,
    num_polys: usize,
    #[allocative(skip)]
    one_hot_params: OneHotParams,
    binding_order: BindingOrder,
}

/// Round 3 state: further split eq tables
#[derive(Allocative, Default)]
pub struct SharedRaRound3<F: JoltField> {
    tables_00: Vec<Vec<F>>,
    tables_01: Vec<Vec<F>>,
    tables_10: Vec<Vec<F>>,
    tables_11: Vec<Vec<F>>,
    indices: Vec<RaIndices>,
    num_polys: usize,
    #[allocative(skip)]
    one_hot_params: OneHotParams,
    binding_order: BindingOrder,
}

impl<F: JoltField> SharedRaPolynomials<F> {
    /// Create new SharedRaPolynomials from eq table and indices.
    pub fn new(tables: Vec<Vec<F>>, indices: Vec<RaIndices>, one_hot_params: OneHotParams) -> Self {
        let num_polys =
            one_hot_params.instruction_d + one_hot_params.bytecode_d + one_hot_params.ram_d;
        debug_assert!(
            tables.len() == num_polys,
            "SharedRaPolynomials::new: tables.len() = {}, expected num_polys = {}",
            tables.len(),
            num_polys
        );
        Self::Round1(SharedRaRound1 {
            tables,
            indices,
            num_polys,
            one_hot_params,
        })
    }

    /// Get the number of polynomials
    pub fn num_polys(&self) -> usize {
        match self {
            Self::Round1(r) => r.num_polys,
            Self::Round2(r) => r.num_polys,
            Self::Round3(r) => r.num_polys,
            Self::RoundN(polys) => polys.len(),
        }
    }

    /// Get the current length (number of cycles / 2^rounds_so_far)
    pub fn len(&self) -> usize {
        match self {
            Self::Round1(r) => r.indices.len(),
            Self::Round2(r) => r.indices.len() / 2,
            Self::Round3(r) => r.indices.len() / 4,
            Self::RoundN(polys) => polys[0].len(),
        }
    }

    /// Get bound coefficient for polynomial `poly_idx` at position `j`
    #[inline]
    pub fn get_bound_coeff(&self, poly_idx: usize, j: usize) -> F {
        match self {
            Self::Round1(r) => r.get_bound_coeff(poly_idx, j),
            Self::Round2(r) => r.get_bound_coeff(poly_idx, j),
            Self::Round3(r) => r.get_bound_coeff(poly_idx, j),
            Self::RoundN(polys) => polys[poly_idx].get_bound_coeff(j),
        }
    }

    /// Get final sumcheck claim for polynomial `poly_idx`
    pub fn final_sumcheck_claim(&self, poly_idx: usize) -> F {
        match self {
            Self::RoundN(polys) => polys[poly_idx].final_sumcheck_claim(),
            _ => panic!("final_sumcheck_claim called before RoundN"),
        }
    }

    /// Bind with a challenge, transitioning to next round state.
    /// Consumes self and returns the new state.
    pub fn bind(self, r: F::Challenge, order: BindingOrder) -> Self {
        match self {
            Self::Round1(r1) => Self::Round2(r1.bind(r, order)),
            Self::Round2(r2) => Self::Round3(r2.bind(r, order)),
            Self::Round3(r3) => Self::RoundN(r3.bind(r, order)),
            Self::RoundN(mut polys) => {
                polys.par_iter_mut().for_each(|p| p.bind_parallel(r, order));
                Self::RoundN(polys)
            }
        }
    }

    /// Bind in place with a challenge, transitioning to next round state.
    pub fn bind_in_place(&mut self, r: F::Challenge, order: BindingOrder) {
        // Use mem::take pattern (same as ra_poly.rs) for efficiency
        match self {
            Self::Round1(r1) => *self = Self::Round2(std::mem::take(r1).bind(r, order)),
            Self::Round2(r2) => *self = Self::Round3(std::mem::take(r2).bind(r, order)),
            Self::Round3(r3) => *self = Self::RoundN(std::mem::take(r3).bind(r, order)),
            Self::RoundN(polys) => {
                polys.par_iter_mut().for_each(|p| p.bind_parallel(r, order));
            }
        }
    }
}

impl<F: JoltField> SharedRaRound1<F> {
    #[inline]
    fn get_bound_coeff(&self, poly_idx: usize, j: usize) -> F {
        self.indices[j]
            .get_index(poly_idx, &self.one_hot_params)
            .map_or(F::zero(), |k| self.tables[poly_idx][k as usize])
    }

    fn bind(self, r0: F::Challenge, order: BindingOrder) -> SharedRaRound2<F> {
        let eq_0_r0 = EqPolynomial::mle(&[F::zero()], &[r0]);
        let eq_1_r0 = EqPolynomial::mle(&[F::one()], &[r0]);
        let (tables_0, tables_1) = rayon::join(
            || {
                self.tables
                    .par_iter()
                    .map(|t| t.iter().map(|v| eq_0_r0 * v).collect::<Vec<F>>())
                    .collect::<Vec<Vec<F>>>()
            },
            || {
                self.tables
                    .par_iter()
                    .map(|t| t.iter().map(|v| eq_1_r0 * v).collect::<Vec<F>>())
                    .collect::<Vec<Vec<F>>>()
            },
        );
        drop_in_background_thread(self.tables);

        SharedRaRound2 {
            tables_0,
            tables_1,
            indices: self.indices,
            num_polys: self.num_polys,
            one_hot_params: self.one_hot_params,
            binding_order: order,
        }
    }
}

impl<F: JoltField> SharedRaRound2<F> {
    #[inline]
    fn get_bound_coeff(&self, poly_idx: usize, j: usize) -> F {
        match self.binding_order {
            BindingOrder::HighToLow => {
                let mid = self.indices.len() / 2;
                let h_0 = self.indices[j]
                    .get_index(poly_idx, &self.one_hot_params)
                    .map_or(F::zero(), |k| self.tables_0[poly_idx][k as usize]);
                let h_1 = self.indices[mid + j]
                    .get_index(poly_idx, &self.one_hot_params)
                    .map_or(F::zero(), |k| self.tables_1[poly_idx][k as usize]);
                h_0 + h_1
            }
            BindingOrder::LowToHigh => {
                let h_0 = self.indices[2 * j]
                    .get_index(poly_idx, &self.one_hot_params)
                    .map_or(F::zero(), |k| self.tables_0[poly_idx][k as usize]);
                let h_1 = self.indices[2 * j + 1]
                    .get_index(poly_idx, &self.one_hot_params)
                    .map_or(F::zero(), |k| self.tables_1[poly_idx][k as usize]);
                h_0 + h_1
            }
        }
    }

    fn bind(self, r1: F::Challenge, order: BindingOrder) -> SharedRaRound3<F> {
        assert_eq!(order, self.binding_order);
        let eq_0_r1 = EqPolynomial::mle(&[F::zero()], &[r1]);
        let eq_1_r1 = EqPolynomial::mle(&[F::one()], &[r1]);

        let mut tables_00 = self.tables_0.clone();
        let mut tables_01 = self.tables_0;
        let mut tables_10 = self.tables_1.clone();
        let mut tables_11 = self.tables_1;

        // Scale all four groups in parallel.
        rayon::join(
            || {
                rayon::join(
                    || {
                        tables_00
                            .par_iter_mut()
                            .for_each(|t| t.par_iter_mut().for_each(|f| *f *= eq_0_r1))
                    },
                    || {
                        tables_01
                            .par_iter_mut()
                            .for_each(|t| t.par_iter_mut().for_each(|f| *f *= eq_1_r1))
                    },
                )
            },
            || {
                rayon::join(
                    || {
                        tables_10
                            .par_iter_mut()
                            .for_each(|t| t.par_iter_mut().for_each(|f| *f *= eq_0_r1))
                    },
                    || {
                        tables_11
                            .par_iter_mut()
                            .for_each(|t| t.par_iter_mut().for_each(|f| *f *= eq_1_r1))
                    },
                )
            },
        );

        SharedRaRound3 {
            tables_00,
            tables_01,
            tables_10,
            tables_11,
            indices: self.indices,
            num_polys: self.num_polys,
            one_hot_params: self.one_hot_params,
            binding_order: order,
        }
    }
}

impl<F: JoltField> SharedRaRound3<F> {
    #[inline]
    fn get_bound_coeff(&self, poly_idx: usize, j: usize) -> F {
        match self.binding_order {
            BindingOrder::HighToLow => {
                let quarter = self.indices.len() / 4;
                let h_00 = self.indices[j]
                    .get_index(poly_idx, &self.one_hot_params)
                    .map_or(F::zero(), |k| self.tables_00[poly_idx][k as usize]);
                let h_01 = self.indices[quarter + j]
                    .get_index(poly_idx, &self.one_hot_params)
                    .map_or(F::zero(), |k| self.tables_01[poly_idx][k as usize]);
                let h_10 = self.indices[2 * quarter + j]
                    .get_index(poly_idx, &self.one_hot_params)
                    .map_or(F::zero(), |k| self.tables_10[poly_idx][k as usize]);
                let h_11 = self.indices[3 * quarter + j]
                    .get_index(poly_idx, &self.one_hot_params)
                    .map_or(F::zero(), |k| self.tables_11[poly_idx][k as usize]);
                h_00 + h_01 + h_10 + h_11
            }
            BindingOrder::LowToHigh => {
                // Bit pattern for offset: (r1, r0), so offset 1 = r0=1,r1=0 → F_10
                let h_00 = self.indices[4 * j]
                    .get_index(poly_idx, &self.one_hot_params)
                    .map_or(F::zero(), |k| self.tables_00[poly_idx][k as usize]);
                let h_10 = self.indices[4 * j + 1]
                    .get_index(poly_idx, &self.one_hot_params)
                    .map_or(F::zero(), |k| self.tables_10[poly_idx][k as usize]);
                let h_01 = self.indices[4 * j + 2]
                    .get_index(poly_idx, &self.one_hot_params)
                    .map_or(F::zero(), |k| self.tables_01[poly_idx][k as usize]);
                let h_11 = self.indices[4 * j + 3]
                    .get_index(poly_idx, &self.one_hot_params)
                    .map_or(F::zero(), |k| self.tables_11[poly_idx][k as usize]);
                h_00 + h_10 + h_01 + h_11
            }
        }
    }

    #[tracing::instrument(skip_all, name = "SharedRaRound3::bind")]
    fn bind(self, r2: F::Challenge, order: BindingOrder) -> Vec<MultilinearPolynomial<F>> {
        assert_eq!(order, self.binding_order);

        // Create 8 F tables: F_ABC where A=r0, B=r1, C=r2
        let eq_0_r2 = EqPolynomial::mle(&[F::zero()], &[r2]);
        let eq_1_r2 = EqPolynomial::mle(&[F::one()], &[r2]);

        let mut tables_000 = self.tables_00.clone();
        let mut tables_001 = self.tables_00;
        let mut tables_010 = self.tables_01.clone();
        let mut tables_011 = self.tables_01;
        let mut tables_100 = self.tables_10.clone();
        let mut tables_101 = self.tables_10;
        let mut tables_110 = self.tables_11.clone();
        let mut tables_111 = self.tables_11;

        // Scale by eq(r2, bit)
        rayon::join(
            || {
                rayon::join(
                    || {
                        rayon::join(
                            || {
                                tables_000
                                    .par_iter_mut()
                                    .for_each(|t| t.par_iter_mut().for_each(|f| *f *= eq_0_r2))
                            },
                            || {
                                tables_001
                                    .par_iter_mut()
                                    .for_each(|t| t.par_iter_mut().for_each(|f| *f *= eq_1_r2))
                            },
                        )
                    },
                    || {
                        rayon::join(
                            || {
                                tables_010
                                    .par_iter_mut()
                                    .for_each(|t| t.par_iter_mut().for_each(|f| *f *= eq_0_r2))
                            },
                            || {
                                tables_011
                                    .par_iter_mut()
                                    .for_each(|t| t.par_iter_mut().for_each(|f| *f *= eq_1_r2))
                            },
                        )
                    },
                )
            },
            || {
                rayon::join(
                    || {
                        rayon::join(
                            || {
                                tables_100
                                    .par_iter_mut()
                                    .for_each(|t| t.par_iter_mut().for_each(|f| *f *= eq_0_r2))
                            },
                            || {
                                tables_101
                                    .par_iter_mut()
                                    .for_each(|t| t.par_iter_mut().for_each(|f| *f *= eq_1_r2))
                            },
                        )
                    },
                    || {
                        rayon::join(
                            || {
                                tables_110
                                    .par_iter_mut()
                                    .for_each(|t| t.par_iter_mut().for_each(|f| *f *= eq_0_r2))
                            },
                            || {
                                tables_111
                                    .par_iter_mut()
                                    .for_each(|t| t.par_iter_mut().for_each(|f| *f *= eq_1_r2))
                            },
                        )
                    },
                )
            },
        );

        // Collect all 8 table groups for indexed access: group[offset][poly_idx][k]
        let table_groups = [
            &tables_000,
            &tables_100,
            &tables_010,
            &tables_110,
            &tables_001,
            &tables_101,
            &tables_011,
            &tables_111,
        ];

        // Materialize all polynomials in parallel
        let num_polys = self.num_polys;
        let indices = &self.indices;
        let one_hot_params = &self.one_hot_params;
        let new_len = indices.len() / 8;

        (0..num_polys)
            .into_par_iter()
            .map(|poly_idx| {
                let coeffs: Vec<F> = match order {
                    BindingOrder::LowToHigh => {
                        (0..new_len)
                            .map(|j| {
                                // Sum over 8 consecutive indices, each using appropriate F table
                                (0..8)
                                    .map(|offset| {
                                        indices[8 * j + offset]
                                            .get_index(poly_idx, one_hot_params)
                                            .map_or(F::zero(), |k| {
                                                table_groups[offset][poly_idx][k as usize]
                                            })
                                    })
                                    .sum()
                            })
                            .collect()
                    }
                    BindingOrder::HighToLow => {
                        let eighth = indices.len() / 8;
                        (0..new_len)
                            .map(|j| {
                                (0..8)
                                    .map(|seg| {
                                        indices[seg * eighth + j]
                                            .get_index(poly_idx, one_hot_params)
                                            .map_or(F::zero(), |k| {
                                                table_groups[seg][poly_idx][k as usize]
                                            })
                                    })
                                    .sum()
                            })
                            .collect()
                    }
                };
                MultilinearPolynomial::from(coeffs)
            })
            .collect()
    }
}

/// Compute all RaIndices in parallel (non-transposed).
///
/// Returns one `RaIndices` per cycle.
#[tracing::instrument(skip_all, name = "shared_ra_polys::compute_ra_indices")]
pub fn compute_ra_indices(
    trace: &[Cycle],
    bytecode: &BytecodePreprocessing,
    memory_layout: &MemoryLayout,
    one_hot_params: &OneHotParams,
) -> Vec<RaIndices> {
    trace
        .par_iter()
        .map(|cycle| RaIndices::from_cycle(cycle, bytecode, memory_layout, one_hot_params))
        .collect()
}
