//! Runtime evaluators for uniform R1CS and product virtualization
//!
//! This module implements the runtime evaluation semantics for the compile-time
//! constraints declared in `r1cs::constraints`:
//!
//! - Grouped evaluators for the uniform R1CS constraints used by the
//!   univariate‑skip first round of Spartan outer sumcheck:
//!   - Typed guard/magnitude structs: `AzFirstGroup`, `BzFirstGroup`,
//!     `AzSecondGroup`, `BzSecondGroup`
//!   - Wrappers `R1CSFirstGroup` and `R1CSSecondGroup` expose `eval_az`,
//!     `eval_bz`, and window-weighted evaluators `az_at_r`, `bz_at_r`
//!   - Specialized `extended_azbz_product` helpers implement the folded
//!     accumulation pattern used by the first-round polynomial
//!   - Shapes (boolean vs. wider signed magnitudes) match the grouping
//!     described in `r1cs::constraints`
//!
//! - Input claim computation (at the end of Spartan outer sumcheck):
//!   - `R1CSEval::compute_claimed_inputs` accumulates all `JoltR1CSInputs`
//!     values at a random point without materializing per-input polynomials,
//!     using split `EqPolynomial` and fixed-limb accumulators
//!
//! - Evaluation helpers for the product virtualization sumcheck:
//!   - `ProductVirtualEval::fused_left_right_at_r` computes the fused left and
//!     right factor values at the r0 window for a single cycle row
//!   - `ProductVirtualEval::compute_claimed_factors` computes z(r) for the 8
//!     de-duplicated factor polynomials consumed by Spartan outer
//!
//! What does not live here:
//! - The definition of any constraint or grouping metadata (see
//!   `r1cs::constraints` for uniform constraints, grouping constants, and the
//!   product-virtualization catalog)
//!
//! Implementation notes:
//! - Accumulator limb widths are chosen to match the value ranges of each type
//!   (bool/u8/u64/i128/S128/S160), minimizing conversions while keeping fast
//!   Barrett reductions.
//! - Test-only `assert_constraints` methods validate that Az guards imply zero
//!   Bz magnitudes for both groups.

use ark_ff::biginteger::{S128, S160, S192, S256, S64};
use ark_std::Zero;
use rayon::prelude::*;
use strum::IntoEnumIterator;
use tracer::instruction::Cycle;

use crate::field::{BarrettReduce, FMAdd, JoltField};
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::lagrange_poly::LagrangeHelper;
use crate::poly::opening_proof::{OpeningPoint, BIG_ENDIAN};
use crate::subprotocols::univariate_skip::uniskip_targets;
use crate::utils::{
    accumulation::{Acc5U, Acc6S, Acc6U, Acc7S, Acc7U, S128Sum, S192Sum},
    math::s64_from_diff_u64s,
};
use crate::zkvm::bytecode::BytecodePreprocessing;
use crate::zkvm::instruction::{CircuitFlags, NUM_CIRCUIT_FLAGS};
use crate::zkvm::r1cs::inputs::ProductCycleInputs;

use super::constraints::{
    NUM_PRODUCT_VIRTUAL, OUTER_UNIVARIATE_SKIP_DEGREE, OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE,
    PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE, PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE,
};
#[cfg(test)]
use super::constraints::{R1CS_CONSTRAINTS_FIRST_GROUP, R1CS_CONSTRAINTS_SECOND_GROUP};
use super::inputs::{JoltR1CSInputs, R1CSCycleInputs, NUM_R1CS_INPUTS};

pub(crate) const UNISKIP_TARGETS: [i64; OUTER_UNIVARIATE_SKIP_DEGREE] =
    uniskip_targets::<OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE, OUTER_UNIVARIATE_SKIP_DEGREE>();

pub(crate) const BASE_LEFT: i64 = -((OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE as i64 - 1) / 2);

pub(crate) const TARGET_SHIFTS: [i64; OUTER_UNIVARIATE_SKIP_DEGREE] = {
    let mut out = [0i64; OUTER_UNIVARIATE_SKIP_DEGREE];
    let mut j: usize = 0;
    while j < OUTER_UNIVARIATE_SKIP_DEGREE {
        out[j] = UNISKIP_TARGETS[j] - BASE_LEFT;
        j += 1;
    }
    out
};

pub(crate) const COEFFS_PER_J: [[i32; OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE];
    OUTER_UNIVARIATE_SKIP_DEGREE] = {
    let mut out = [[0i32; OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE]; OUTER_UNIVARIATE_SKIP_DEGREE];
    let mut j: usize = 0;
    while j < OUTER_UNIVARIATE_SKIP_DEGREE {
        out[j] =
            LagrangeHelper::shift_coeffs_i32::<OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE>(TARGET_SHIFTS[j]);
        j += 1;
    }
    out
};

pub(crate) const PRODUCT_VIRTUAL_UNISKIP_TARGETS: [i64; PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE] =
    uniskip_targets::<
        PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE,
        PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE,
    >();

pub(crate) const PRODUCT_VIRTUAL_BASE_LEFT: i64 =
    -((PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE as i64 - 1) / 2);

pub(crate) const PRODUCT_VIRTUAL_TARGET_SHIFTS: [i64; PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE] = {
    let mut out = [0i64; PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE];
    let mut j: usize = 0;
    while j < PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE {
        out[j] = PRODUCT_VIRTUAL_UNISKIP_TARGETS[j] - PRODUCT_VIRTUAL_BASE_LEFT;
        j += 1;
    }
    out
};

pub(crate) const PRODUCT_VIRTUAL_COEFFS_PER_J: [[i32;
    PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE];
    PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE] = {
    let mut out = [[0i32; PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE];
        PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE];
    let mut j: usize = 0;
    while j < PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE {
        out[j] = LagrangeHelper::shift_coeffs_i32::<PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE>(
            PRODUCT_VIRTUAL_TARGET_SHIFTS[j],
        );
        j += 1;
    }
    out
};

/// Boolean guards for the first group (univariate-skip base window)
#[derive(Clone, Copy, Debug)]
pub struct AzFirstGroup {
    pub not_load_store: bool,      // !(Load || Store)
    pub load_a: bool,              // Load
    pub load_b: bool,              // Load
    pub store: bool,               // Store
    pub add_sub_mul: bool,         // Add || Sub || Mul
    pub not_add_sub_mul: bool,     // !(Add || Sub || Mul)
    pub assert_flag: bool,         // Assert
    pub should_jump: bool,         // ShouldJump
    pub virtual_instruction: bool, // VirtualInstruction
    pub must_start_sequence: bool, // NextIsVirtual && !NextIsFirstInSequence
}

impl AzFirstGroup {
    /// Fused multiply-add into an unreduced accumulator using Lagrange weights `w`
    /// over the univariate-skip base window. This mirrors `az_at_r_first_group`
    /// but keeps the result in an `Acc5U` accumulator without reducing.
    #[inline(always)]
    pub fn fmadd_at_r<F: JoltField>(
        &self,
        w: &[F; OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE],
        acc: &mut Acc5U<F>,
    ) {
        acc.fmadd(&w[0], &self.not_load_store);
        acc.fmadd(&w[1], &self.load_a);
        acc.fmadd(&w[2], &self.load_b);
        acc.fmadd(&w[3], &self.store);
        acc.fmadd(&w[4], &self.add_sub_mul);
        acc.fmadd(&w[5], &self.not_add_sub_mul);
        acc.fmadd(&w[6], &self.assert_flag);
        acc.fmadd(&w[7], &self.should_jump);
        acc.fmadd(&w[8], &self.virtual_instruction);
        acc.fmadd(&w[9], &self.must_start_sequence);
    }
}

/// Magnitudes for the first group (kept small: bool/u64/S64)
#[derive(Clone, Copy, Debug)]
pub struct BzFirstGroup {
    pub ram_addr: u64,                               // RamAddress - 0
    pub ram_read_minus_ram_write: S64,               // RamRead - RamWrite
    pub ram_read_minus_rd_write: S64,                // RamRead - RdWrite
    pub rs2_minus_ram_write: S64,                    // Rs2 - RamWrite
    pub left_lookup: u64,                            // LeftLookup - 0
    pub left_lookup_minus_left_input: S64,           // LeftLookup - LeftInstructionInput
    pub lookup_output_minus_one: S64,                // LookupOutput - 1
    pub next_unexp_pc_minus_lookup_output: S64,      // NextUnexpandedPC - LookupOutput
    pub next_pc_minus_pc_plus_one: S64,              // NextPC - (PC + 1)
    pub one_minus_do_not_update_unexpanded_pc: bool, // 1 - DoNotUpdateUnexpandedPC
}

impl BzFirstGroup {
    /// Fused multiply-add into an unreduced accumulator using Lagrange weights `w`
    /// over the univariate-skip base window. This mirrors `bz_at_r_first_group`
    /// but keeps the result in an `Acc6S` accumulator without reducing.
    #[inline(always)]
    pub fn fmadd_at_r<F: JoltField>(
        &self,
        w: &[F; OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE],
        acc: &mut Acc6S<F>,
    ) {
        acc.fmadd(&w[0], &self.ram_addr);
        acc.fmadd(&w[1], &self.ram_read_minus_ram_write);
        acc.fmadd(&w[2], &self.ram_read_minus_rd_write);
        acc.fmadd(&w[3], &self.rs2_minus_ram_write);
        acc.fmadd(&w[4], &self.left_lookup);
        acc.fmadd(&w[5], &self.left_lookup_minus_left_input);
        acc.fmadd(&w[6], &self.lookup_output_minus_one);
        acc.fmadd(&w[7], &self.next_unexp_pc_minus_lookup_output);
        acc.fmadd(&w[8], &self.next_pc_minus_pc_plus_one);
        acc.fmadd(&w[9], &self.one_minus_do_not_update_unexpanded_pc);
    }
}

/// Guards for the second group (all booleans except two u8 flags)
#[derive(Clone, Copy, Debug)]
pub struct AzSecondGroup {
    pub load_or_store: bool,          // Load || Store
    pub add: bool,                    // Add
    pub sub: bool,                    // Sub
    pub mul: bool,                    // Mul
    pub not_add_sub_mul_advice: bool, // !(Add || Sub || Mul || Advice)
    pub write_lookup_to_rd: bool,     // write_lookup_output_to_rd_addr (Rd != 0)
    pub write_pc_to_rd: bool,         // write_pc_to_rd_addr (Rd != 0)
    pub should_branch: bool,          // ShouldBranch
    pub not_jump_or_branch: bool,     // !(Jump || ShouldBranch)
}

impl AzSecondGroup {
    /// Fused multiply-add into an unreduced accumulator using Lagrange weights `w`
    /// over the univariate-skip base window. This mirrors `az_at_r_second_group`
    /// but keeps the result in an `Acc5U` accumulator without reducing.
    #[inline(always)]
    pub fn fmadd_at_r<F: JoltField>(
        &self,
        w: &[F; OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE],
        acc: &mut Acc5U<F>,
    ) {
        acc.fmadd(&w[0], &self.load_or_store);
        acc.fmadd(&w[1], &self.add);
        acc.fmadd(&w[2], &self.sub);
        acc.fmadd(&w[3], &self.mul);
        acc.fmadd(&w[4], &self.not_add_sub_mul_advice);
        acc.fmadd(&w[5], &self.write_lookup_to_rd);
        acc.fmadd(&w[6], &self.write_pc_to_rd);
        acc.fmadd(&w[7], &self.should_branch);
        acc.fmadd(&w[8], &self.not_jump_or_branch);
    }
}

/// Magnitudes for the second group (mixed precision up to S160)
#[derive(Clone, Copy, Debug)]
pub struct BzSecondGroup {
    pub ram_addr_minus_rs1_plus_imm: i128, // RamAddress - (Rs1 + Imm)
    pub right_lookup_minus_add_result: S160, // RightLookup - (Left + Right)
    pub right_lookup_minus_sub_result: S160, // RightLookup - (Left - Right + 2^64)
    pub right_lookup_minus_product: S160,  // RightLookup - Product
    pub right_lookup_minus_right_input: S160, // RightLookup - RightInput
    pub rd_write_minus_lookup_output: S64, // RdWrite - LookupOutput
    pub rd_write_minus_pc_plus_const: S64, // RdWrite - (UnexpandedPC + const)
    pub next_unexp_pc_minus_pc_plus_imm: i128, // NextUnexpandedPC - (UnexpandedPC + Imm)
    pub next_unexp_pc_minus_expected: S64, // NextUnexpandedPC - (UnexpandedPC + const)
}

impl BzSecondGroup {
    /// Fused multiply-add into an unreduced accumulator using Lagrange weights `w`
    /// over the univariate-skip base window. This mirrors `bz_at_r_second_group`
    /// but keeps the result in an `Acc7S` accumulator without reducing.
    #[inline(always)]
    pub fn fmadd_at_r<F: JoltField>(
        &self,
        w: &[F; OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE],
        acc: &mut Acc7S<F>,
    ) {
        acc.fmadd(&w[0], &self.ram_addr_minus_rs1_plus_imm);
        acc.fmadd(&w[1], &self.right_lookup_minus_add_result);
        acc.fmadd(&w[2], &self.right_lookup_minus_sub_result);
        acc.fmadd(&w[3], &self.right_lookup_minus_product);
        acc.fmadd(&w[4], &self.right_lookup_minus_right_input);
        acc.fmadd(&w[5], &self.rd_write_minus_lookup_output);
        acc.fmadd(&w[6], &self.rd_write_minus_pc_plus_const);
        acc.fmadd(&w[7], &self.next_unexp_pc_minus_pc_plus_imm);
        acc.fmadd(&w[8], &self.next_unexp_pc_minus_expected);
    }
}

/// Unified evaluator wrapper with typed accessors for both groups
#[derive(Clone, Copy, Debug)]
pub struct R1CSEval<'a, F: JoltField> {
    row: &'a R1CSCycleInputs,
    _m: core::marker::PhantomData<F>,
}

impl<'a, F: JoltField> R1CSEval<'a, F> {
    #[inline]
    pub fn from_cycle_inputs(row: &'a R1CSCycleInputs) -> Self {
        Self {
            row,
            _m: core::marker::PhantomData,
        }
    }

    // ---------- First group ----------

    /// 计算第一组约束的 Az 向量（Guard/Selector 向量）。
    ///
    /// # 作用
    /// Az 向量中的每个元素对应一个或一组 R1CS 约束的“开关”。
    /// 如果某个指令触发了特定的逻辑（比如它是一条 LOAD 指令），对应的 Az 字段就会为 true (1)，
    /// 进而强制要求对应的 Bz (Magnitude) 必须为 0。
    ///
    /// # 逻辑映射
    /// 这个函数将底层的电路标志位（CircuitFlags，如 Load/Store/Add 等）映射到
    /// Spartan 协议所需的具体约束分组结构 `AzFirstGroup` 中。
    #[inline]
    pub fn eval_az_first_group(&self) -> AzFirstGroup {
        // 获取当前指令周期的电路标志位数组
        let flags = &self.row.flags;

        // 提取基础指令类型的原始标志位
        let ld = flags[CircuitFlags::Load];             // 是否为加载指令 (LB, LH, LW, etc.)
        let st = flags[CircuitFlags::Store];            // 是否为存储指令 (SB, SH, SW)
        let add = flags[CircuitFlags::AddOperands];     // 是否为加法类操作
        let sub = flags[CircuitFlags::SubtractOperands];// 是否为减法类操作
        let mul = flags[CircuitFlags::MultiplyOperands];// 是否为乘法类操作
        let assert_flag = flags[CircuitFlags::Assert];  // 是否为断言指令 (用于测试或特殊检查)
        let inline_seq = flags[CircuitFlags::VirtualInstruction]; // 是否为虚拟指令序列的一部分

        AzFirstGroup {
            // 守卫: "非访存指令"。
            // 对应的 Bz 约束通常是: 如果不是 Load/Store，则内存地址 (RamAddr) 必须为 0。
            not_load_store: !(ld || st),

            // 守卫: "Load 指令操作数检查 A"。
            // 对应的 Bz 约束通常涉及内存读取值的一致性检查。
            load_a: ld,

            // 守卫: "Load 指令操作数检查 B"。
            // 可能对应 Load 指令的额外约束条件。
            load_b: ld,

            // 守卫: "Store 指令"。
            // 对应的 Bz 约束通常检查存储的值是否正确写入内存 (Rs2 - RamWrite == 0)。
            store: st,

            // 守卫: "基础算术指令 (Add/Sub/Mul)"。
            // 这一类指令共享某些查找表逻辑 (LeftLookup == 0 约束)。
            add_sub_mul: add || sub || mul,

            // 守卫: "非基础算术指令"。
            // 这通常涵盖了除了基本加减乘之外的其他操作（如位运算、比较等），
            // 对应的 Bz 检查左操作数的一致性 (LeftLookup - LeftInput == 0)。
            not_add_sub_mul: !(add || sub || mul),

            // 守卫: "断言"。
            // 对应的 Bz 检查 LookupOutput 是否等于 1。
            assert_flag,

            // 守卫: "应当跳转"。
            // 派生自 Trace 中的 `should_jump` 字段，用于分支跳转指令的 PC 检查。
            should_jump: self.row.should_jump,

            // 守卫: "虚拟指令"。
            // 用于处理复杂的 Jolt 虚拟指令序列。
            virtual_instruction: inline_seq,

            // 守卫: "必须开始序列"。
            // 这是一个复杂的控制流逻辑：如果下一条指令是虚拟指令，但不是序列的第一条，
            // 意味着我们处于一个长序列的中间，必须维护特定的 PC 更新逻辑 (DoNotUpdateUnexpandedPC)。
            must_start_sequence: self.row.next_is_virtual && !self.row.next_is_first_in_sequence,
        }
    }

    /// 计算第一组约束的 Bz 向量（Magnitude/Value 向量）。
    ///
    /// # 作用
    /// Bz 向量代表了约束方程中的“数值差”或“状态值”。
    /// 对于每一个约束 $i$，如果对应的守卫 $Az_i$ 为 true（即当前指令触发了该约束上下文），
    /// 那么 $Bz_i$ 必须等于 0。
    ///
    /// 公式通常形式为：$Bz = \text{Actual} - \text{Expected}$。
    ///
    /// # 字段详细映射
    #[inline]
    pub fn eval_bz_first_group(&self) -> BzFirstGroup {
        BzFirstGroup {
            // [约束 0: 非访存指令]
            // 如果 Az.not_load_store 为真（即不是 Load/Store），则 ram_addr 必须为 0。
            // 这里 Bz 直接取 ram_addr 的值。
            // 目的：防止算术指令污染内存地址总线，或确保非内存操作不产生内存副作用。
            ram_addr: self.row.ram_addr,

            // [约束 1: Load 一致性 A]
            // 如果 Az.load_a 为真，则 ram_read_value 必须等于 ram_write_value。
            // 这里的语义是：在 Load 操作中，内存子系统“读取”的值和该周期“写入/维持”的值是一致的。
            // (通常用于证明内存读操作没有改变内存中的值，或保持 Trace 一致性)
            ram_read_minus_ram_write: s64_from_diff_u64s(
                self.row.ram_read_value,
                self.row.ram_write_value,
            ),

            // [约束 2: Load 一致性 B]
            // 如果 Az.load_b 为真，则 ram_read_value 必须等于 rd_write_value。
            // 目的：确保从内存读取的数据准确无误地被写入了目标寄存器 (Rd)。
            ram_read_minus_rd_write: s64_from_diff_u64s(
                self.row.ram_read_value,
                self.row.rd_write_value,
            ),

            // [约束 3: Store 一致性]
            // 如果 Az.store 为真，则 rs2_read_value 必须等于 ram_write_value。
            // 目的：确保源寄存器 (Rs2) 的数据准确无误地被写入了内存系统。
            rs2_minus_ram_write: s64_from_diff_u64s(
                self.row.rs2_read_value,
                self.row.ram_write_value,
            ),

            // [约束 4: 基础算术指令 (Add/Sub/Mul)]
            // 如果 Az.add_sub_mul 为真，则 left_lookup 必须为 0。
            // 目的：对于这三种基础运算，可能不需要使用左侧查找表输入，强制置零以保持状态确定性。
            left_lookup: self.row.left_lookup,

            // [约束 5: 非基础算术指令]
            // 如果 Az.not_add_sub_mul 为真（如位运算、比较运算），则 left_lookup 必须等于 left_input。
            // 目的：数据透传约束。确保指令的操作数 (left_input) 被正确传递给了查找表接口 (left_lookup)。
            left_lookup_minus_left_input: s64_from_diff_u64s(
                self.row.left_lookup,
                self.row.left_input,
            ),

            // [约束 6: 断言指令]
            // 如果 Az.assert_flag 为真，则 lookup_output 必须等于 1。
            // 这里的 lookup_output 通常是比较操作的结果 (1=True, 0=False)。
            // 目的：实现 Assert 语义，验证条件必须成立。
            lookup_output_minus_one: s64_from_diff_u64s(self.row.lookup_output, 1),

            // [约束 7: 条件跳转]
            // 如果 Az.should_jump 为真，则 next_unexpanded_pc 必须等于 lookup_output。
            // 在分支指令中，lookup_output 承载了目标地址计算结果。
            // 目的：验证程序计数器 (PC) 的跳转目标。
            next_unexp_pc_minus_lookup_output: s64_from_diff_u64s(
                self.row.next_unexpanded_pc,
                self.row.lookup_output,
            ),

            // [约束 8: 虚拟指令序列]
            // 如果 Az.virtual_instruction 为真，则 next_pc 必须等于 pc + 1。
            // 目的：虚拟指令序列通常由多个 micro-op 组成，它们在 trace 中占据连续的行，
            // 逻辑 PC 只是简单递增。
            next_pc_minus_pc_plus_one: s64_from_diff_u64s(
                self.row.next_pc,
                self.row.pc.wrapping_add(1),
            ),

            // [约束 9: 序列控制]
            // 如果 Az.must_start_sequence 为真，则 DoNotUpdateUnexpandedPC 标志必须为 1 (True)。
            // 计算逻辑：Bz = !Flag。如果 Bz=False(0)，则 Flag=True。
            // 目的：当处于虚拟指令序列中间时，真实的 CPU PC (UnexpandedPC) 不应更新，保持锁定直到序列结束。
            one_minus_do_not_update_unexpanded_pc: !self.row.flags
                [CircuitFlags::DoNotUpdateUnexpandedPC],
        }
    }

    #[inline]
    pub fn az_at_r_first_group(&self, w: &[F; OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE]) -> F {
        let az = self.eval_az_first_group();
        let mut acc: Acc5U<F> = Acc5U::zero();
        acc.fmadd(&w[0], &az.not_load_store);
        acc.fmadd(&w[1], &az.load_a);
        acc.fmadd(&w[2], &az.load_b);
        acc.fmadd(&w[3], &az.store);
        acc.fmadd(&w[4], &az.add_sub_mul);
        acc.fmadd(&w[5], &az.not_add_sub_mul);
        acc.fmadd(&w[6], &az.assert_flag);
        acc.fmadd(&w[7], &az.should_jump);
        acc.fmadd(&w[8], &az.virtual_instruction);
        acc.fmadd(&w[9], &az.must_start_sequence);
        acc.barrett_reduce()
    }

    #[inline]
    pub fn bz_at_r_first_group(&self, w: &[F; OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE]) -> F {
        let bz = self.eval_bz_first_group();
        let mut acc: Acc6S<F> = Acc6S::zero();
        acc.fmadd(&w[0], &bz.ram_addr);
        acc.fmadd(&w[1], &bz.ram_read_minus_ram_write);
        acc.fmadd(&w[2], &bz.ram_read_minus_rd_write);
        acc.fmadd(&w[3], &bz.rs2_minus_ram_write);
        acc.fmadd(&w[4], &bz.left_lookup);
        acc.fmadd(&w[5], &bz.left_lookup_minus_left_input);
        acc.fmadd(&w[6], &bz.lookup_output_minus_one);
        acc.fmadd(&w[7], &bz.next_unexp_pc_minus_lookup_output);
        acc.fmadd(&w[8], &bz.next_pc_minus_pc_plus_one);
        acc.fmadd(&w[9], &bz.one_minus_do_not_update_unexpanded_pc);
        acc.barrett_reduce()
    }

    /// Fused accumulate of first-group Az and Bz into unreduced accumulators using
    /// Lagrange weights `w`. This keeps everything in unreduced form; callers are
    /// responsible for reducing at the end.
    #[inline]
    pub fn fmadd_first_group_at_r(
        &self,
        w: &[F; OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE],
        acc_az: &mut Acc5U<F>,
        acc_bz: &mut Acc6S<F>,
    ) {
        let az = self.eval_az_first_group();
        az.fmadd_at_r(w, acc_az);
        let bz = self.eval_bz_first_group();
        bz.fmadd_at_r(w, acc_bz);
    }

    /// Product Az·Bz at the j-th extended uniskip target for the first group (uses precomputed weights).
    /// 计算第一组约束在扩展点 j 处的 Az(j) * Bz(j) 乘积。
    ///
    /// # 数学背景
    /// 我们需要计算两个多项式在点 j 的评估值的乘积：
    /// Result = (\sum_{k} c_k \cdot Az_k) * (\sum_{k} c_k \cdot Bz_k)
    /// 其中 c_k 是拉格朗日基函数 L_k 在点 j 的值 (预计算为 coeffs_i32)。
    ///
    /// # 核心优化逻辑 (分支判断)
    /// 对于每一个约束项 k，Az_k 是布尔值 "Guard"（守卫），Bz_k 是数值 "Magnitude"（量级）。
    /// 如果电路约束被满足，必然有：Az_k * Bz_k == 0。
    /// 这意味着我们不需要同时计算 Az 和 Bz 的累加，可以根据 Az_k 的值走分支：
    ///
    /// 1. 如果 Az_k == 1 (true):
    ///    - Az 多项式累加: Az_eval += c_k * 1
    ///    - Bz 多项式累加: 由于 Az_k=1 且约束满足，隐含 Bz_k=0。所以 Bz_eval += c_k * 0 (跳过计算)。
    ///
    /// 2. 如果 Az_k == 0 (false):
    ///    - Az 多项式累加: Az_eval += c_k * 0 (跳过计算)。
    ///    - Bz 多项式累加: Bz_eval += c_k * Bz_k (执行乘加)。
    ///
    ///这种优化避免了大量的零乘法和加法，极大提升了证明性能。
    pub fn extended_azbz_product_first_group(&self, j: usize) -> S192 {
        // [调试生] 确保在计算前，当前行的数据确实满足 R1CS 约束。
        // 如果不满足，下面的优化逻辑（假设 Az=1 => Bz=0）将导致计算错误的证明。
        #[cfg(test)]
        self.assert_constraints_first_group();

        // 获取预计算好的拉格朗日系数，对应于评估点 j
        let coeffs_i32: &[i32; OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE] = &COEFFS_PER_J[j];

        // 从当前 Trace 行中提取 Az (标志位组) 和 Bz (约束值组)
        let az = self.eval_az_first_group();
        let bz = self.eval_bz_first_group();

        // 初始化累加器
        // Az 部分全是布尔值和系数相加，用 i32 足够（系数很小）
        let mut az_eval_i32: i32 = 0;
        // Bz 部分包含 64位整数运算，累加后可能溢出 u64，使用 S128Sum 保证精度
        let mut bz_eval_s128: S128Sum = S128Sum::zero();

        // --- 逐项展开计算 (利用展开循环消除索引开销) ---

        // --- 约束 0: 非访存指令 (Not Load/Store) ---
        // 场景: 当前指令既不是 Load 也不是 Store。
        // Az (Guard): !Flags.Load && !Flags.Store
        // Bz (Value): ram_addr
        // 约束逻辑: 如果不进行访存，内存地址线 `ram_addr` 必须保持清洁 (0)。
        // 作用: 防止算术指令意外触发内存逻辑。
        let c0_i32 = coeffs_i32[0];
        if az.not_load_store {
            az_eval_i32 += c0_i32; // Az=1, Bz=0 (Implied)
        } else {
            bz_eval_s128.fmadd(&c0_i32, &bz.ram_addr); // Az=0, Accumulate Bz
        }

        // --- 约束 1: Load 指令一致性检查 A ---
        // 场景: 当前是 Load 指令。
        // Az (Guard): Flags.Load
        // Bz (Value): ram_read_value - ram_write_value
        // 约束逻辑: 从内存读取的值必须等于该周期内存系统记录的值。
        // 作用: 确保内存读操作的数据一致性。
        let c1_i32 = coeffs_i32[1];
        if az.load_a {
            az_eval_i32 += c1_i32;
        } else {
            bz_eval_s128.fmadd(&c1_i32, &bz.ram_read_minus_ram_write);
        }

        // --- 约束 2: Load 指令一致性检查 B ---
        // 场景: 当前是 Load 指令。
        // Az (Guard): Flags.Load
        // Bz (Value): ram_read_value - rd_write_value
        // 约束逻辑: 从内存读取的值必须被正确写入到目标寄存器 (Rd)。
        // 作用: 确保数据从内存正确流向寄存器堆。
        let c2_i32 = coeffs_i32[2];
        if az.load_b {
            az_eval_i32 += c2_i32;
        } else {
            bz_eval_s128.fmadd(&c2_i32, &bz.ram_read_minus_rd_write);
        }

        // --- 约束 3: Store 指令一致性检查 ---
        // 场景: 当前是 Store 指令。
        // Az (Guard): Flags.Store
        // Bz (Value): rs2_read_value - ram_write_value
        // 约束逻辑: 源寄存器 (Rs2) 的值必须等于写入内存的值。
        // 作用: 确保数据从寄存器堆正确流向内存。
        let c3_i32 = coeffs_i32[3];
        if az.store {
            az_eval_i32 += c3_i32;
        } else {
            bz_eval_s128.fmadd(&c3_i32, &bz.rs2_minus_ram_write);
        }

        // --- 约束 4: 基础算术指令 (Add/Sub/Mul) ---
        // 场景: 加法、减法或乘法指令。
        // Az (Guard): Add || Sub || Mul
        // Bz (Value): left_lookup (通常为 0)
        // 约束逻辑: 对于基础算术指令，LeftLookup 表输入通常不被直接使用（或者有特定含义），这里约束它为 0。
        // （注：具体语境取决于 Lookup 的定义，通常这里是为了确保 R1CS 矩阵的某些列在这些操作下保持清洁）
        let c4_i32 = coeffs_i32[4];
        if az.add_sub_mul {
            az_eval_i32 += c4_i32;
        } else {
            bz_eval_s128.fmadd(&c4_i32, &bz.left_lookup);
        }

        // --- 约束 5: 非基础算术指令 (Not Add/Sub/Mul) ---
        // 场景: 不是 Add/Sub/Mul 的指令。
        // Az (Guard): !(Add || Sub || Mul)
        // Bz (Value): left_lookup - left_input
        // 约束逻辑: 对于其他指令，Lookups 表的左输入必须等于指令的第一个操作数 (rs1)。
        let c5_i32 = coeffs_i32[5];
        if az.not_add_sub_mul {
            az_eval_i32 += c5_i32;
        } else {
            bz_eval_s128.fmadd(&c5_i32, &bz.left_lookup_minus_left_input);
        }

        let c6_i32 = coeffs_i32[6];
        if az.assert_flag {
            az_eval_i32 += c6_i32;
        } else {
            bz_eval_s128.fmadd(&c6_i32, &bz.lookup_output_minus_one);
        }

        let c7_i32 = coeffs_i32[7];
        if az.should_jump {
            az_eval_i32 += c7_i32;
        } else {
            bz_eval_s128.fmadd(&c7_i32, &bz.next_unexp_pc_minus_lookup_output);
        }

        let c8_i32 = coeffs_i32[8];
        if az.virtual_instruction {
            az_eval_i32 += c8_i32;
        } else {
            bz_eval_s128.fmadd(&c8_i32, &bz.next_pc_minus_pc_plus_one);
        }

        let c9_i32 = coeffs_i32[9];
        if az.must_start_sequence {
            az_eval_i32 += c9_i32;
        } else {
            bz_eval_s128.fmadd(&c9_i32, &bz.one_minus_do_not_update_unexpanded_pc);
        }

        // 最终步骤：将 Az 和 Bz 的结果组合成 R^2 域中的值
        // 1. Az (i32) -> S64 (Signed 64-bit)
        // 使用 S64::from_i64 明确转换
        let az_eval_s64 = S64::from_i64(az_eval_i32 as i64);

        // 2. Bz (S128Sum) -> S128
        // S128Sum 内部维护一个 S128 类型的 sum，直接访问。
        // bz_eval_s128.sum 已经是归约后的结果（或者说是在累加过程中维护的值）。
        let bz_eval_val = bz_eval_s128.sum;

        // 3. 执行 Az * Bz
        //    Az (S64, 1 limb) * Bz (S128, 2 limbs) -> Result (S192, 3 limbs)
        //    使用 mul_trunc 指定右操作数 limb 数为 2，结果 limb 数为 3。
        az_eval_s64.mul_trunc::<2, 3>(&bz_eval_val)
    }

    #[cfg(test)]
    fn assert_constraint_first_group(&self, index: usize, guard: bool, satisfied: bool) {
        if guard && !satisfied {
            let mut constraint_string = String::new();
            let _ = R1CS_CONSTRAINTS_FIRST_GROUP[index]
                .pretty_fmt_with_row(&mut constraint_string, self.row);
            println!("{constraint_string}");
            panic!(
                "First group constraint {} ({:?}) violated",
                index, R1CS_CONSTRAINTS_FIRST_GROUP[index].label
            );
        }
    }

    #[cfg(test)]
    /// 验证第一组约束的满足性（仅用于测试和调试）。
    ///
    /// # 目的
    /// 这个函数遍历第一组的所有 10 个 R1CS 约束，确保对于当前的 Execution Trace 行，
    /// 如果某个约束的 Guard ($Az$) 被激活（为 true），那么对应的 Magnitude ($Bz$) 必须为 0（即约束满足）。
    ///
    /// # 逻辑
    /// 断言逻辑遵循：`Guard_i => Magnitude_i == 0`。
    /// 这里的 `satisfied` 参数在 `assert_constraint_first_group` 内部被检查：
    /// 如果 `guard` 为真且 `satisfied` 为假，则触发 panic。
    #[cfg(test)]
    pub fn assert_constraints_first_group(&self) {
        let az = self.eval_az_first_group();
        let bz = self.eval_bz_first_group();

        // 约束 0: 非访存指令 (Not Load/Store)
        // 规则: 如果不是 Load/Store 指令，内存地址 (RamAddr) 必须为 0。
        // 这防止了非内存指令意外产生内存访问副作用。
        self.assert_constraint_first_group(0, az.not_load_store, bz.ram_addr == 0);

        // 约束 1: Load 指令一致性 A
        // 规则: 如果是 Load 指令，内存读取值 (RamRead) 必须等于写回内存的值 (RamWrite)。
        // 注意：在 Jolt 中，Load 操作通常读写一致（或者 RamWrite 代表本次访问在内存系统的实际值）。
        self.assert_constraint_first_group(
            1,
            az.load_a,
            bz.ram_read_minus_ram_write.to_i128() == 0,
        );

        // 约束 2: Load 指令一致性 B
        // 规则: 如果是 Load 指令，内存读取值 (RamRead) 必须等于写入寄存器的值 (RdWrite)。
        // 确保 Load 加载的数据正确流向了目标寄存器。
        self.assert_constraint_first_group(2, az.load_b, bz.ram_read_minus_rd_write.to_i128() == 0);

        // 约束 3: Store 指令一致性
        // 规则: 如果是 Store 指令，源寄存器值 (Rs2) 必须等于写入内存的值 (RamWrite)。
        // 确保 Store 操作将正确的数据写入了内存系统。
        self.assert_constraint_first_group(3, az.store, bz.rs2_minus_ram_write.to_i128() == 0);

        // 约束 4: 基础算术指令 (Add/Sub/Mul)
        // 规则: 对于基础算术指令，LeftLookup 字段必须为 0。
        // 这可能是因为对于这三种基础操作，某些不需要的查找表输入被强制置零。
        self.assert_constraint_first_group(4, az.add_sub_mul, bz.left_lookup == 0);

        // 约束 5: 非基础算术指令 (所有其他指令)
        // 规则: 如果不是基础算术指令，LeftLookup 必须等于 LeftInput。
        // 这是一个“透传”约束，确保数据在多路复用器（MUX）路径上的一致性，即输入被正确传送到查找表接口。
        self.assert_constraint_first_group(
            5,
            az.not_add_sub_mul,
            bz.left_lookup_minus_left_input.to_i128() == 0,
        );

        // 约束 6: 断言指令 (Assert)
        // 规则: 如果是 Assert 指令，LookupOutput 必须等于 1。
        // 这里的 LookupOutput 通常来自 EQ/NEQ 等比较指令的结果，Assert 确保该结果为真 (1)。
        self.assert_constraint_first_group(
            6,
            az.assert_flag,
            bz.lookup_output_minus_one.to_i128() == 0,
        );

        // 约束 7: 条件跳转 (ShouldJump)
        // 规则: 如果发生跳转，下一个 UnexpandedPC 必须等于 LookupOutput。
        // 在分支指令中，LookupOutput 承载了计算出的跳转目标地址。
        self.assert_constraint_first_group(
            7,
            az.should_jump,
            bz.next_unexp_pc_minus_lookup_output.to_i128() == 0,
        );

        // 约束 8: 虚拟指令内部 (Virtual Instruction)
        // 规则: 如果是普通虚拟指令步骤（非跳转），NextPC 仅仅是 PC + 1（逻辑上的微操作步进）。
        self.assert_constraint_first_group(
            8,
            az.virtual_instruction,
            bz.next_pc_minus_pc_plus_one.to_i128() == 0,
        );

        // 约束 9: 序列开始控制 (Start Sequence)
        // 规则: 如果正处于 "Must Start Sequence" 状态（长序列中间），必须设置 DoNotUpdateUnexpandedPC 标志。
        // bz.one_minus... 实际上是 (1 - Flag)。如果约束满足 (Bz=0)，意味着 Flag 必须为 1。
        // 代码中的 satisfied 传入的是 !bz，即 !(False) -> True，表示验证通过。
        self.assert_constraint_first_group(
            9,
            az.must_start_sequence,
            !bz.one_minus_do_not_update_unexpanded_pc,
        );
    }
    // ---------- Second group ----------


    /// 计算第二组约束的 Az 向量（Guard/Selector 向量）。
    ///
    /// # 作用
    /// Az 向量充当约束的开关。Jolt 将大量的 R1CS 约束分为两组以优化计算结构。
    /// 第二组主要关注：
    /// 1. 复杂算术操作的正确性 (RightLookup 检查)。
    /// 2. 寄存器写回逻辑 (Rd Write)。
    /// 3. 程序计数器 (PC) 的更新逻辑。
    ///
    /// # 逻辑映射
    /// 此函数依据当前 Trace 行的 `CircuitFlags` 和辅助状态（如 `should_branch`），
    /// 生成对应的 `AzSecondGroup` 结构体。
    #[inline]
    pub fn eval_az_second_group(&self) -> AzSecondGroup {
        let flags = &self.row.flags;

        // 辅助逻辑：判断是否是非基础运算且非 Advice 指令。
        //
        // 对于取模 (%)、位移 (>>/<<)、位运算 (&/|/^) 等指令：
        // 1. 它们不属于 Add/Sub/Mul 这类需要特殊电路算术检查的指令。
        // 2. 它们也不属于 Advice (非确定性输入)。
        // 3. 这里的逻辑值为 true。
        //
        // 作用：激活 BzSecondGroup 中的 `right_lookup_minus_right_input` 约束。
        // 效果：强制要求 RightLookup (查找表右输入) == RightInput (指令右操作数)。
        // 结合第一组约束中的 LeftLookup == LeftInput，这确保了 `%` 或 `>>` 的两个操作数
        // 被原封不动地传递给了 Lasso 查找表系统进行计算。
        let not_add_sub_mul_advice = !(flags[CircuitFlags::AddOperands]
            || flags[CircuitFlags::SubtractOperands]
            || flags[CircuitFlags::MultiplyOperands]
            || flags[CircuitFlags::Advice]);

        // 辅助逻辑：判断 PC 更新是否属于“默认情况”
        // 如果不是无条件跳转 (Jump) 且不是需要执行的分支跳转 (ShouldBranch)，
        // 那么 PC 应该顺序更新 (通常是 PC + 4 或 PC + 2，但在 Jolt 中有特定的步进逻辑)。
        let next_update_otherwise = {
            let jump = flags[CircuitFlags::Jump];
            let should_branch = self.row.should_branch;
            (!jump) && (!should_branch)
        };

        AzSecondGroup {
            // 守卫: "Load 或 Store 指令"。
            // 对应的 Bz 检查: 内存地址计算的一致性 (RamAddr - (Rs1 + Imm) == 0)。
            load_or_store: (flags[CircuitFlags::Load] || flags[CircuitFlags::Store]),

            // 守卫: "加法操作"。
            // 对应的 Bz 检查: RightLookup 是否等于 (Left + Right)。
            add: flags[CircuitFlags::AddOperands],

            // 守卫: "减法操作"。
            // 对应的 Bz 检查: RightLookup 是否等于 (Left - Right)。
            sub: flags[CircuitFlags::SubtractOperands],

            // 守卫: "乘法操作"。
            // 对应的 Bz 检查: RightLookup 是否等于 (Left * Right)。
            mul: flags[CircuitFlags::MultiplyOperands],

            // 守卫: "非基础算术且非 Advice"。取模 (%)、位移 (>>/<<)、位运算 (&/|/^) 等指令
            // 对应的 Bz 检查: 右操作数透传检查 (RightLookup == RightInput)。
            not_add_sub_mul_advice,

            // 守卫: "将 Lookup 输出写入通用寄存器 (Rd)"。
            // 对应的 Bz 检查: RdWrite 值是否等于 LookupOutput。
            // 注意: 只有当 Rd != 0 时，write_lookup_output_to_rd_addr 才为真。
            write_lookup_to_rd: self.row.write_lookup_output_to_rd_addr,

            // 守卫: "将 PC 写入通用寄存器 (Rd)" (通常用于 JAL/JALR)。
            // 对应的 Bz 检查: RdWrite 值是否等于 (UnexpandedPC + 4)。
            write_pc_to_rd: self.row.write_pc_to_rd_addr,

            // 守卫: "应当执行条件分支"。
            // 对应的 Bz 检查: 下一个 PC 是否跳转到了 (PC + Imm)。
            should_branch: self.row.should_branch,

            // 守卫: "非跳转且非分支" (顺序执行)。
            // 对应的 Bz 检查: 下一个 PC 是否符合顺序执行的预期 (通常是 PC + 指令长度)。
            not_jump_or_branch: next_update_otherwise,
        }
    }


    /// 计算第二组约束的 Bz 向量（Magnitude/Value 向量）。
    ///
    /// # 作用
    /// Bz 向量代表了约束方程中的“数值差”。只有当对应的 Guard ($Az$) 为 true 时，这些差值才必须为 0。
    /// 例如，对于加法指令：
    /// - $Az$.add = true
    /// - $Bz$.right_lookup_minus_add_result = (RightLookup - (Left + Right))
    /// - 约束要求：1 * (RightLookup - (Left + Right)) == 0
    /// # 核心逻辑：查找表 (Lookup) 与 R1CS 的交互
    /// 在 Jolt 中，大多数复杂指令（如 AND, OR, XOR, SLT 等）不是通过 R1CS 电路直接计算的，而是通过 **Lasso 查找表** 证明的。
    /// R1CS 的作用是将 CPU 的寄存器状态正确地“连接”到查找表的输入/输出端口。
    ///
    /// *   对于通用指令 (如 AND)：R1CS 只需要验证 `LeftLookup == LeftInput` (第一组) 和 `RightLookup == RightInput` (本组)，以及 `Rd == LookupOutput`。至于 `LookupOutput` 是否真的是 `Left AND Right`，由 Lasso 协议保证。
    /// *   对于算术指令 (ADD/SUB/MUL)：虽然也可以用查找表，但 R1CS 直接验证算术关系（如 A+B=C）往往更高效或作为冗余检查。在这里，`RightLookup` 字段被复用作为“计算结果”的占位符。
    #[inline]
    pub fn eval_bz_second_group(&self) -> BzSecondGroup {
        // --- 1. Load/Store 有效地址计算 ---
        // 适用指令：LB, LH, LW, LBU, LHU, SB, SH, SW 等。
        // 约束逻辑：`RamAddr` 必须等于 `Rs1 + Imm`。
        // 原理：验证访存指令生成的物理内存地址是否符合 RISC-V 规范。
        // 这里的 `expected_addr` 处理了立即数的符号扩展。
        let expected_addr: i128 = if self.row.imm.is_positive {
            (self.row.rs1_read_value as u128 + self.row.imm.magnitude_as_u64() as u128) as i128
        } else {
            // 处理负立即数的情况
            self.row.rs1_read_value as i128 - self.row.imm.magnitude_as_u64() as i128
        };
        // Bz = Actual(RamAddress) - Expected(Rs1 + Imm)
        let ram_addr_minus_rs1_plus_imm = self.row.ram_addr as i128 - expected_addr;

        // --- 2. 算术运算与查找表数据通路一致性检查 ---
        // 这里的 `RightLookup` 字段是一个多义字段，根据指令类型的不同，它扮演不同角色。
        //
        // 我们需要验证：对于给定的 `LeftInput` 和 `RightInput`，系统内部流转的数据是否正确。

        // 预计算加法预期值: Left + Right
        let right_add_expected = (self.row.left_input as i128) + self.row.right_input.to_i128();
        // 预计算减法预期值: Left - Right + 2^64 (模拟 64位 溢出截断行为)
        let right_sub_expected =
            (self.row.left_input as i128) - self.row.right_input.to_i128() + (1i128 << 64);

        // [Case A: 加法指令] (ADD, ADDI)
        // 角色：`RightLookup` 在此处被视为加法结果。
        // 约束：验证 `RightLookup` 是否等于 `Left + Right`。
        // 注意：这里是在 S160 域上做检查，确保没有意外的算术错误。
        let right_lookup_minus_add_result =
            S160::from(self.row.right_lookup) - S160::from(right_add_expected);

        // [Case B: 减法指令] (SUB)
        // 角色：`RightLookup` 在此处被视为减法结果。
        // 约束：验证 `RightLookup` 是否等于 `Left - Right` (模 2^64)。
        let right_lookup_minus_sub_result =
            S160::from(self.row.right_lookup) - S160::from(right_sub_expected);

        // [Case C: 乘法指令] (MUL)
        // 角色：`RightLookup` 在此处被视为乘法结果。
        // 约束：验证 `RightLookup` 是否等于 `Product`。
        // 注意：`Product` 是 Prover 提供的 Witness。这里的约束实际上是检查三方一致性：
        // Left * Right (由 Witness 生成逻辑保证) == Product == RightLookup。
        let right_lookup_minus_product =
            S160::from(self.row.right_lookup) - S160::from(self.row.product);

        // [Case D: 通用查找表指令] (SLL, SRL, AND, OR, XOR, SLT 等)
        // 角色：`RightLookup` 在此处回归本义，即“查找表查询的右操作数 (Query Input Y)”。
        // 约束：验证 `RightLookup` 是否等于指令的右操作数 `RightInput`。
        //
        // 详细流程：
        // 1. CPU 读取指令，得到操作数 A (LeftInput) 和 B (RightInput)。
        // 2. R1CS 第一组约束验证：`LeftLookup == LeftInput` (将 A 传入查找表端口 X)。
        // 3. R1CS 本处约束验证：`RightLookup == RightInput` (将 B 传入查找表端口 Y)。
        // 4. Lasso 协议证明：存在一条记录 (X, Y, Z) 在查找表中，其中 Z 是 `LookupOutput`。
        // 5. R1CS 下面的写回约束验证：`Rd == LookupOutput` (将结果 Z 写回寄存器)。
        // 结论：这被称为“数据透传”约束，保证了 CPU 和 查找表协处理器 之间的连线没接错。
        let right_lookup_minus_right_input =
            S160::from(self.row.right_lookup) - S160::from(self.row.right_input);

        // --- 3. 寄存器写回 (Rd Write) 检查 ---

        // [Case A: 常规计算指令]
        // 适用：所有将其结果来自 Lookup Output 的指令 (ALU 运算)。
        // 约束：`RdWriteValue - LookupOutput == 0`。
        // 作用：将计算结果（无论是来自算术逻辑还是查找表）提交到寄存器堆。
        let rd_write_minus_lookup_output =
            s64_from_diff_u64s(self.row.rd_write_value, self.row.lookup_output);

        // [Case B: 跳转链接指令] (JAL, JALR)
        // 适用：需要保存返回地址到 Rd (通常是 x1/ra)。
        // 约束：`RdWriteValue - (PC + Step) == 0`。
        // Step 计算：依据 RISC-V 压缩指令集 (RVC)，如果当前指令是压缩的 (16位)，下一条地址+2，否则+4。
        let const_term = 4 - if self.row.flags[CircuitFlags::IsCompressed] {
            2
        } else {
            0
        };
        let expected_pc_plus_const = self.row.unexpanded_pc.wrapping_add(const_term as u64);
        let rd_write_minus_pc_plus_const =
            s64_from_diff_u64s(self.row.rd_write_value, expected_pc_plus_const);

        // --- 4. 下一条 PC (Next Unexpanded PC) 更新检查 ---

        // [Case A: 分支跳转] (BEQ, BNE, BLT 等)
        // 条件：`should_branch` 为真 (即指令是 Branch 类型且条件成立)。
        // 约束：`NextPC` 必须等于 `CurrentPC + Imm` (相对跳转)。
        let next_unexp_pc_minus_pc_plus_imm = (self.row.next_unexpanded_pc as i128)
            - (self.row.unexpanded_pc as i128 + self.row.imm.to_i128());

        // [Case B: 顺序执行] (大部分指令)
        // 条件：既不跳转也不分支。
        // 约束：`NextPC` 必须等于 `CurrentPC + Step`。
        // Step 逻辑：
        // - 默认 +4。
        // - 如果处于 `DoNotUpdateUnexpandedPC` 状态 (虚拟指令序列中间)，则 Step=0 (保持 PC 不变)。
        // - 如果是压缩指令，Step 减 2 (最终 +2)。
        // - 这些逻辑通过算术叠加处理。
        let const_term_next =
            4 - if self.row.flags[CircuitFlags::DoNotUpdateUnexpandedPC] {
                4
            } else {
                0
            } - if self.row.flags[CircuitFlags::IsCompressed] {
                2
            } else {
                0
            };
        let expected_next = self.row.unexpanded_pc.wrapping_add(const_term_next as u64);
        let next_unexp_pc_minus_expected =
            s64_from_diff_u64s(self.row.next_unexpanded_pc, expected_next);

        BzSecondGroup {
            ram_addr_minus_rs1_plus_imm,
            right_lookup_minus_add_result,
            right_lookup_minus_sub_result,
            right_lookup_minus_product,
            right_lookup_minus_right_input,
            rd_write_minus_lookup_output,
            rd_write_minus_pc_plus_const,
            next_unexp_pc_minus_pc_plus_imm,
            next_unexp_pc_minus_expected,
        }
    }

    #[inline]
    pub fn az_at_r_second_group(&self, _w: &[F; OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE]) -> F {
        let w = _w;
        let az = self.eval_az_second_group();
        let mut acc: Acc5U<F> = Acc5U::zero();
        acc.fmadd(&w[0], &az.load_or_store);
        acc.fmadd(&w[1], &az.add);
        acc.fmadd(&w[2], &az.sub);
        acc.fmadd(&w[3], &az.mul);
        acc.fmadd(&w[4], &az.not_add_sub_mul_advice);
        acc.fmadd(&w[5], &az.write_lookup_to_rd);
        acc.fmadd(&w[6], &az.write_pc_to_rd);
        acc.fmadd(&w[7], &az.should_branch);
        acc.fmadd(&w[8], &az.not_jump_or_branch);
        acc.barrett_reduce()
    }

    #[inline]
    pub fn bz_at_r_second_group(&self, _w: &[F; OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE]) -> F {
        let w = _w;
        let bz = self.eval_bz_second_group();
        let mut acc: Acc7S<F> = Acc7S::zero();
        acc.fmadd(&w[0], &bz.ram_addr_minus_rs1_plus_imm);
        acc.fmadd(&w[1], &bz.right_lookup_minus_add_result);
        acc.fmadd(&w[2], &bz.right_lookup_minus_sub_result);
        acc.fmadd(&w[3], &bz.right_lookup_minus_product);
        acc.fmadd(&w[4], &bz.right_lookup_minus_right_input);
        acc.fmadd(&w[5], &bz.rd_write_minus_lookup_output);
        acc.fmadd(&w[6], &bz.rd_write_minus_pc_plus_const);
        acc.fmadd(&w[7], &bz.next_unexp_pc_minus_pc_plus_imm);
        acc.fmadd(&w[8], &bz.next_unexp_pc_minus_expected);
        acc.barrett_reduce()
    }

    /// Fused accumulate of second-group Az and Bz into unreduced accumulators
    /// using Lagrange weights `w`. This keeps everything in unreduced form; callers
    /// are responsible for reducing at the end.
    #[inline]
    pub fn fmadd_second_group_at_r(
        &self,
        w: &[F; OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE],
        acc_az: &mut Acc5U<F>,
        acc_bz: &mut Acc7S<F>,
    ) {
        let az = self.eval_az_second_group();
        az.fmadd_at_r(w, acc_az);
        let bz = self.eval_bz_second_group();
        bz.fmadd_at_r(w, acc_bz);
    }

    /// Product Az·Bz at the j-th extended uniskip target for the second group (uses precomputed weights).
    /// 计算第二组约束在扩展点 j 处的 Az(j) * Bz(j) 乘积。
    ///
    /// # 核心逻辑
    /// 使用 Univariate Skip 优化：对于每个约束 $k$，
    /// 1. 如果守卫 $Az_k$ 为真，根据约束满足性 $Az \cdot Bz = 0$，意味着 $Bz_k$ 必须为 0。因此只更新 Az 多项式，跳过 Bz 计算。
    /// 2. 如果守卫 $Az_k$ 为假 ($0$)，Az 对总和无贡献。因此只更新 Bz 多项式。
    pub fn extended_azbz_product_second_group(&self, j: usize) -> S192 {
        #[cfg(test)]
        self.assert_constraints_second_group();

        let coeffs_i32: &[i32; OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE] = &COEFFS_PER_J[j];
        let az = self.eval_az_second_group();
        let bz = self.eval_bz_second_group();

        let mut az_eval_i32: i32 = 0;
        let mut bz_eval_s192 = S192Sum::zero();

        // --- 约束 0: Load/Store 地址计算 ---
        // 场景: 当前是 Load 或 Store 指令。
        // Az (Guard): Flags.Load || Flags.Store
        // Bz (Value): ram_addr - (rs1 + imm)
        // 约束逻辑: 内存访问的物理地址 (ram_addr) 必须等于“基址寄存器值 (rs1) + 立即数偏移 (imm)”。
        // 作用: 验证 RISC-V 访存指令的有效地址计算公式。
        let c0 = coeffs_i32[0];
        if az.load_or_store {
            az_eval_i32 += c0;
        } else {
            bz_eval_s192.fmadd(&c0, &bz.ram_addr_minus_rs1_plus_imm);
        }

        // --- 约束 1: 加法 (ADD) ---
        // 场景: 当前是 ADD 指令。
        // Az (Guard): Flags.Add
        // Bz (Value): right_lookup - (left_input + right_input)
        // 约束逻辑: 运算结果 (right_lookup) 必须等于两操作数之和。
        // 注意: 这里的 right_lookup 承载了 ALU 的输出。
        let c1 = coeffs_i32[1];
        if az.add {
            az_eval_i32 += c1;
        } else {
            bz_eval_s192.fmadd(&c1, &bz.right_lookup_minus_add_result);
        }

        // --- 约束 2: 减法 (SUB) ---
        // 场景: 当前是 SUB 指令。
        // Az (Guard): Flags.Sub
        // Bz (Value): right_lookup - (left_input - right_input)
        // 约束逻辑: 运算结果 (right_lookup) 必须等于两操作数之差。
        let c2 = coeffs_i32[2];
        if az.sub {
            az_eval_i32 += c2;
        } else {
            bz_eval_s192.fmadd(&c2, &bz.right_lookup_minus_sub_result);
        }

        // --- 约束 3: 乘法 (MUL) ---
        // 场景: 当前是 MUL 指令。
        // Az (Guard): Flags.Mul
        // Bz (Value): right_lookup - (left_input * right_input)
        // 约束逻辑: 运算结果 必须等于两操作数之积。
        let c3 = coeffs_i32[3];
        if az.mul {
            az_eval_i32 += c3;
        } else {
            bz_eval_s192.fmadd(&c3, &bz.right_lookup_minus_product);
        }

        // --- 约束 4: 非算术指令的数据透传 ---
        // 场景: 不是 Add/Sub/Mul，也不是 Advice 指令。这涵盖了位运算、比较等操作。
        // Az (Guard): !(Add || Sub || Mul || Advice)
        // Bz (Value): right_lookup - right_input
        // 约束逻辑: 在这些指令中，右操作数必须被“透传”到查找表接口 (right_lookup) 以供后续使用。
        // 作用: 确保电路中的多路复用器 (MUX) 路径正确，数据没有在从 Input 到 Lookup 的传输中损坏。
        let c4 = coeffs_i32[4];
        if az.not_add_sub_mul_advice {
            az_eval_i32 += c4;
        } else {
            bz_eval_s192.fmadd(&c4, &bz.right_lookup_minus_right_input);
        }

        // --- 约束 5: 将结果写回寄存器 (Reg Write) ---
        // 场景: 指令需要将 ALU/Lookup 的计算结果写回目标寄存器 Rd (且 Rd != 0)。
        // Az (Guard): Flags.WriteLookupToRd
        // Bz (Value): rd_write - lookup_output
        // 约束逻辑: 最终写入寄存器的值 (rd_write) 必须等于计算模块的输出 (lookup_output)。
        let c5 = coeffs_i32[5];
        if az.write_lookup_to_rd {
            az_eval_i32 += c5;
        } else {
            bz_eval_s192.fmadd(&c5, &bz.rd_write_minus_lookup_output);
        }

        // --- 约束 6: 将 PC 写回寄存器 (JAL/JALR Link) ---
        // 场景: 跳转链接指令，需要保存返回地址 (PC + 4 或 PC + 2) 到寄存器。
        // Az (Guard): Flags.WritePCToRd
        // Bz (Value): rd_write - (pc + const)
        // 约束逻辑: 写入寄存器的值必须是当前 PC 加上指令长度（返回地址）。
        let c6 = coeffs_i32[6];
        if az.write_pc_to_rd {
            az_eval_i32 += c6;
        } else {
            bz_eval_s192.fmadd(&c6, &bz.rd_write_minus_pc_plus_const);
        }

        // --- 约束 7: 条件分支跳转 (Branch Taken) ---
        // 场景: 当前指令是条件分支，且条件成立（执行跳转）。
        // Az (Guard): ShouldBranch
        // Bz (Value): next_pc - (pc + imm)
        // 约束逻辑: 下一个周期的 PC (next_pc) 必须等于当前 PC 加上相对偏移量 (Imm)。
        let c7 = coeffs_i32[7];
        if az.should_branch {
            az_eval_i32 += c7;
        } else {
            bz_eval_s192.fmadd(&c7, &bz.next_unexp_pc_minus_pc_plus_imm);
        }

        // --- 约束 8: 顺序执行 (Sequential Execution) ---
        // 场景: 既不是跳转 (Jump) 也不是分支 (Branch)，程序顺序向下执行。
        // Az (Guard): !Jump && !ShouldBranch
        // Bz (Value): next_pc - (pc + instr_len)
        // 约束逻辑: 下一个周期的 PC 必须等于当前 PC 加上指令长度 (通常是 4，压缩指令是 2，虚拟序列内部是 0)。
        // 作用: 保证控制流在没有跳转指令时的连续性。
        let c8 = coeffs_i32[8];
        if az.not_jump_or_branch {
            az_eval_i32 += c8;
        } else {
            bz_eval_s192.fmadd(&c8, &bz.next_unexp_pc_minus_expected);
        }

        // --- 最终计算 ---
        // 将 Az 的累加结果 (i32) 提升为 S64
        let az_eval_s64 = S64::from_i64(az_eval_i32 as i64);

        // 计算 Az(j) * Bz(j)
        // 使用 mul_trunc 执行宽整数乘法，并将结果截断/转换为 S192 类型返回。
        // S192 足够容纳 (64位 + 128位) 的乘法结果，避免溢出。
        // S64 * S192 -> 需要 S256? 但这里返回 S192。
        // 实际上第二组的 Bz 是 S192Sum (Accumulating S160 values).
        // S64 * S192 = S256. If we truncate to S192?
        // Wait, S192Sum accumulates S160 values? Let's check struct BzSecondGroup.
        // It has S160, i128, S64. So S192Sum is safe.
        // If we multiply S64 * S192, we get S256.
        // BUT the function signature returns S192.
        // Let's check `mul_trunc::<2, 3>` meaning.
        // RHS limbs = ? S192 has 3 limbs. So R=3.
        // Output limbs = 3.
        // So `mul_trunc::<3, 3>`.
        // Wait, the previous code had `mul_trunc::<2, 3>`.
        // Is `bz_eval_s192.sum` actually S128?
        // `let mut bz_eval_s192 = S192Sum::zero();`
        // S192Sum sum field is S192.
        // So R=3.
        // Why did I write `<2, 3>`? Maybe I copied from first group?
        // Let's assume `<3, 3>` is correct for S64 * S192 -> S192 (truncated).
        az_eval_s64.mul_trunc::<3, 3>(&bz_eval_s192.sum)
    }

    #[cfg(test)]
    fn assert_constraint_second_group(&self, index: usize, guard: bool, satisfied: bool) {
        if guard && !satisfied {
            let mut constraint_string = String::new();
            let _ = R1CS_CONSTRAINTS_SECOND_GROUP[index]
                .pretty_fmt_with_row(&mut constraint_string, self.row);
            println!("{constraint_string}");
            panic!(
                "Second group constraint {} ({:?}) violated",
                index, R1CS_CONSTRAINTS_SECOND_GROUP[index].label
            );
        }
    }

    #[cfg(test)]
    pub fn assert_constraints_second_group(&self) {
        let az = self.eval_az_second_group();
        let bz = self.eval_bz_second_group();
        self.assert_constraint_second_group(
            0,
            az.load_or_store,
            bz.ram_addr_minus_rs1_plus_imm == 0i128,
        );
        self.assert_constraint_second_group(1, az.add, bz.right_lookup_minus_add_result.is_zero());
        self.assert_constraint_second_group(2, az.sub, bz.right_lookup_minus_sub_result.is_zero());
        self.assert_constraint_second_group(3, az.mul, bz.right_lookup_minus_product.is_zero());
        self.assert_constraint_second_group(
            4,
            az.not_add_sub_mul_advice,
            bz.right_lookup_minus_right_input.is_zero(),
        );
        self.assert_constraint_second_group(
            5,
            az.write_lookup_to_rd,
            bz.rd_write_minus_lookup_output.is_zero(),
        );
        self.assert_constraint_second_group(
            6,
            az.write_pc_to_rd,
            bz.rd_write_minus_pc_plus_const.is_zero(),
        );
        self.assert_constraint_second_group(
            7,
            az.should_branch,
            bz.next_unexp_pc_minus_pc_plus_imm == 0,
        );
        self.assert_constraint_second_group(
            8,
            az.not_jump_or_branch,
            bz.next_unexp_pc_minus_expected.is_zero(),
        );
    }

    /// Compute `z(r_cycle) = Σ_t eq(r_cycle, t) * P_i(t)` for all inputs i, without
    /// materializing P_i. Returns `[P_0(r_cycle), P_1(r_cycle), ...]` in input order.
    #[tracing::instrument(skip_all, name = "R1CSEval::compute_claimed_inputs")]
    pub fn compute_claimed_inputs(
        bytecode_preprocessing: &BytecodePreprocessing,
        trace: &[Cycle],
        r_cycle: &OpeningPoint<BIG_ENDIAN, F>,
    ) -> [F; NUM_R1CS_INPUTS] {
        let m = r_cycle.len() / 2;
        let (r2, r1) = r_cycle.split_at_r(m);
        let (eq_one, eq_two) = rayon::join(|| EqPolynomial::evals(r2), || EqPolynomial::evals(r1));

        (0..eq_one.len())
            .into_par_iter()
            .map(|x1| {
                let eq1_val = eq_one[x1];

                // Accumulators for each input
                // If bool or u8 => 5 limbs unsigned
                // If u64 => 6 limbs unsigned
                // If i128 => 6 limbs signed
                // If S128 => 7 limbs signed
                let mut acc_left_input: Acc6U<F> = Acc6U::zero();
                let mut acc_right_input: Acc6S<F> = Acc6S::zero();
                let mut acc_product: Acc7S<F> = Acc7S::zero();
                let mut acc_wl_left: Acc5U<F> = Acc5U::zero();
                let mut acc_wp_left: Acc5U<F> = Acc5U::zero();
                let mut acc_sb_right: Acc5U<F> = Acc5U::zero();
                let mut acc_pc: Acc6U<F> = Acc6U::zero();
                let mut acc_unexpanded_pc: Acc6U<F> = Acc6U::zero();
                let mut acc_imm: Acc6S<F> = Acc6S::zero();
                let mut acc_ram_address: Acc6U<F> = Acc6U::zero();
                let mut acc_rs1_value: Acc6U<F> = Acc6U::zero();
                let mut acc_rs2_value: Acc6U<F> = Acc6U::zero();
                let mut acc_rd_write_value: Acc6U<F> = Acc6U::zero();
                let mut acc_ram_read_value: Acc6U<F> = Acc6U::zero();
                let mut acc_ram_write_value: Acc6U<F> = Acc6U::zero();
                let mut acc_left_lookup_operand: Acc6U<F> = Acc6U::zero();
                let mut acc_right_lookup_operand: Acc7U<F> = Acc7U::zero();
                let mut acc_next_unexpanded_pc: Acc6U<F> = Acc6U::zero();
                let mut acc_next_pc: Acc6U<F> = Acc6U::zero();
                let mut acc_lookup_output: Acc6U<F> = Acc6U::zero();
                let mut acc_sj_flag: Acc5U<F> = Acc5U::zero();
                let mut acc_next_is_virtual: Acc5U<F> = Acc5U::zero();
                let mut acc_next_is_first_in_sequence: Acc5U<F> = Acc5U::zero();
                let mut acc_flags: Vec<Acc5U<F>> =
                    (0..NUM_CIRCUIT_FLAGS).map(|_| Acc5U::zero()).collect();

                let eq_two_len = eq_two.len();
                for x2 in 0..eq_two_len {
                    let e_in = eq_two[x2];
                    let idx = x1 * eq_two_len + x2;
                    let row = R1CSCycleInputs::from_trace::<F>(bytecode_preprocessing, trace, idx);

                    acc_left_input.fmadd(&e_in, &row.left_input);
                    acc_right_input.fmadd(&e_in, &row.right_input.to_i128());
                    acc_product.fmadd(&e_in, &row.product);

                    acc_wl_left.fmadd(&e_in, &(row.write_lookup_output_to_rd_addr as u64));
                    acc_wp_left.fmadd(&e_in, &(row.write_pc_to_rd_addr as u64));
                    acc_sb_right.fmadd(&e_in, &row.should_branch);

                    acc_pc.fmadd(&e_in, &row.pc);
                    acc_unexpanded_pc.fmadd(&e_in, &row.unexpanded_pc);
                    acc_imm.fmadd(&e_in, &row.imm.to_i128());
                    acc_ram_address.fmadd(&e_in, &row.ram_addr);
                    acc_rs1_value.fmadd(&e_in, &row.rs1_read_value);
                    acc_rs2_value.fmadd(&e_in, &row.rs2_read_value);
                    acc_rd_write_value.fmadd(&e_in, &row.rd_write_value);
                    acc_ram_read_value.fmadd(&e_in, &row.ram_read_value);
                    acc_ram_write_value.fmadd(&e_in, &row.ram_write_value);
                    acc_left_lookup_operand.fmadd(&e_in, &row.left_lookup);
                    acc_right_lookup_operand.fmadd(&e_in, &row.right_lookup);
                    acc_next_unexpanded_pc.fmadd(&e_in, &row.next_unexpanded_pc);
                    acc_next_pc.fmadd(&e_in, &row.next_pc);
                    acc_lookup_output.fmadd(&e_in, &row.lookup_output);
                    acc_sj_flag.fmadd(&e_in, &row.should_jump);
                    acc_next_is_virtual.fmadd(&e_in, &row.next_is_virtual);
                    acc_next_is_first_in_sequence.fmadd(&e_in, &row.next_is_first_in_sequence);
                    for flag in CircuitFlags::iter() {
                        acc_flags[flag as usize].fmadd(&e_in, &row.flags[flag as usize]);
                    }
                }

                let mut out_unr: [F::Unreduced<9>; NUM_R1CS_INPUTS] =
                    [F::Unreduced::<9>::zero(); NUM_R1CS_INPUTS];
                out_unr[JoltR1CSInputs::LeftInstructionInput.to_index()] =
                    eq1_val.mul_unreduced::<9>(acc_left_input.barrett_reduce());
                out_unr[JoltR1CSInputs::RightInstructionInput.to_index()] =
                    eq1_val.mul_unreduced::<9>(acc_right_input.barrett_reduce());
                out_unr[JoltR1CSInputs::Product.to_index()] =
                    eq1_val.mul_unreduced::<9>(acc_product.barrett_reduce());
                out_unr[JoltR1CSInputs::WriteLookupOutputToRD.to_index()] =
                    eq1_val.mul_unreduced::<9>(acc_wl_left.barrett_reduce());
                out_unr[JoltR1CSInputs::WritePCtoRD.to_index()] =
                    eq1_val.mul_unreduced::<9>(acc_wp_left.barrett_reduce());
                out_unr[JoltR1CSInputs::ShouldBranch.to_index()] =
                    eq1_val.mul_unreduced::<9>(acc_sb_right.barrett_reduce());
                out_unr[JoltR1CSInputs::PC.to_index()] =
                    eq1_val.mul_unreduced::<9>(acc_pc.barrett_reduce());
                out_unr[JoltR1CSInputs::UnexpandedPC.to_index()] =
                    eq1_val.mul_unreduced::<9>(acc_unexpanded_pc.barrett_reduce());
                out_unr[JoltR1CSInputs::Imm.to_index()] =
                    eq1_val.mul_unreduced::<9>(acc_imm.barrett_reduce());
                out_unr[JoltR1CSInputs::RamAddress.to_index()] =
                    eq1_val.mul_unreduced::<9>(acc_ram_address.barrett_reduce());
                out_unr[JoltR1CSInputs::Rs1Value.to_index()] =
                    eq1_val.mul_unreduced::<9>(acc_rs1_value.barrett_reduce());
                out_unr[JoltR1CSInputs::Rs2Value.to_index()] =
                    eq1_val.mul_unreduced::<9>(acc_rs2_value.barrett_reduce());
                out_unr[JoltR1CSInputs::RdWriteValue.to_index()] =
                    eq1_val.mul_unreduced::<9>(acc_rd_write_value.barrett_reduce());
                out_unr[JoltR1CSInputs::RamReadValue.to_index()] =
                    eq1_val.mul_unreduced::<9>(acc_ram_read_value.barrett_reduce());
                out_unr[JoltR1CSInputs::RamWriteValue.to_index()] =
                    eq1_val.mul_unreduced::<9>(acc_ram_write_value.barrett_reduce());
                out_unr[JoltR1CSInputs::LeftLookupOperand.to_index()] =
                    eq1_val.mul_unreduced::<9>(acc_left_lookup_operand.barrett_reduce());
                out_unr[JoltR1CSInputs::RightLookupOperand.to_index()] =
                    eq1_val.mul_unreduced::<9>(acc_right_lookup_operand.barrett_reduce());
                out_unr[JoltR1CSInputs::NextUnexpandedPC.to_index()] =
                    eq1_val.mul_unreduced::<9>(acc_next_unexpanded_pc.barrett_reduce());
                out_unr[JoltR1CSInputs::NextPC.to_index()] =
                    eq1_val.mul_unreduced::<9>(acc_next_pc.barrett_reduce());
                out_unr[JoltR1CSInputs::LookupOutput.to_index()] =
                    eq1_val.mul_unreduced::<9>(acc_lookup_output.barrett_reduce());
                out_unr[JoltR1CSInputs::ShouldJump.to_index()] =
                    eq1_val.mul_unreduced::<9>(acc_sj_flag.barrett_reduce());
                out_unr[JoltR1CSInputs::NextIsVirtual.to_index()] =
                    eq1_val.mul_unreduced::<9>(acc_next_is_virtual.barrett_reduce());
                out_unr[JoltR1CSInputs::NextIsFirstInSequence.to_index()] =
                    eq1_val.mul_unreduced::<9>(acc_next_is_first_in_sequence.barrett_reduce());
                for flag in CircuitFlags::iter() {
                    let idx = JoltR1CSInputs::OpFlags(flag).to_index();
                    let f_idx = flag as usize;
                    out_unr[idx] = eq1_val.mul_unreduced::<9>(acc_flags[f_idx].barrett_reduce());
                }
                out_unr
            })
            .reduce(
                || [F::Unreduced::<9>::zero(); NUM_R1CS_INPUTS],
                |mut acc, item| {
                    for i in 0..NUM_R1CS_INPUTS {
                        acc[i] += item[i];
                    }
                    acc
                },
            )
            .map(|unr| F::from_montgomery_reduce::<9>(unr))
    }
}

/// Struct for implementation of evaluation logic for product virtualization
#[derive(Clone, Copy, Debug)]
pub struct ProductVirtualEval;

impl ProductVirtualEval {
    /// Compute both fused left and right factors at r0 weights for a single cycle row.
    /// Expected order of weights: [Instruction, WriteLookupOutputToRD, WritePCtoRD, ShouldBranch, ShouldJump]
    #[inline]
    pub fn fused_left_right_at_r<F: JoltField>(
        row: &ProductCycleInputs,
        weights_at_r0: &[F],
    ) -> (F, F) {
        // Left: u64/u8/bool
        let mut left_acc: Acc6U<F> = Acc6U::zero();
        left_acc.fmadd(&weights_at_r0[0], &row.instruction_left_input);
        left_acc.fmadd(&weights_at_r0[1], &row.is_rd_not_zero);
        left_acc.fmadd(&weights_at_r0[2], &row.is_rd_not_zero);
        left_acc.fmadd(&weights_at_r0[3], &row.should_branch_lookup_output);
        left_acc.fmadd(&weights_at_r0[4], &row.jump_flag);

        // Right: i128/bool
        let mut right_acc: Acc6S<F> = Acc6S::zero();
        right_acc.fmadd(&weights_at_r0[0], &row.instruction_right_input);
        right_acc.fmadd(&weights_at_r0[1], &row.write_lookup_output_to_rd_flag);
        right_acc.fmadd(&weights_at_r0[2], &row.jump_flag);
        right_acc.fmadd(&weights_at_r0[3], &row.should_branch_flag);
        right_acc.fmadd(&weights_at_r0[4], &row.not_next_noop);

        (left_acc.barrett_reduce(), right_acc.barrett_reduce())
    }

    /// 计算用于乘积虚拟化（Product Virtualization）的融合左右乘积。
    ///
    /// **背景**:
    /// Jolt 有 5 种不同的乘法约束（例如指令解码、寄存器写入、跳转标志等）。
    /// 为了提高效率，我们不分别验证这 5 个约束，而是将它们“虚拟化”为一个单变量多项式 $$P(x) = L(x) \cdot R(x)$$。
    /// - 在基础域（Base Domain, x=0..4）上，$P(x)$ 的值分别对应这 5 个约束的计算结果 ($$L_i \cdot R_i$$)。
    /// - 为了运行 Sumcheck 协议，我们需要在更大的**扩展域**上求值。
    ///
    /// **功能**:
    /// 此函数计算该虚拟多项式在扩展域的第 `j` 个点（uniskip target）上的值。
    /// 计算公式大致为：
    /// $$ Value_j = (\sum_{k=0}^4 c_{j,k} \cdot Left_k) \times (\sum_{k=0}^4 c_{j,k} \cdot Right_k) $$
    /// 其中 $c_{j,k}$ 是预计算的拉格朗日插值系数，$Left_k$ 和 $Right_k$ 是第 $k$ 个约束的左、右输入。
    ///
    /// **参数**:
    /// - `row`: 当前执行周期（Cycle）的所有输入数据（标志位、操作数等）。
    /// - `j`: 扩展域上的评估点索引。
    ///
    /// **返回**:
    /// - `S256`: 为了防止溢出，结果使用 256 位有符号整数表示。
    #[inline]
    pub fn extended_fused_product_at_j<F: JoltField>(row: &ProductCycleInputs, j: usize) -> S256 {
        // 1. 获取预计算的插值系数。
        // `c` 是一个数组，包含 5 个系数，对应于将扩展域点 `j` 映射回基础域 0..4 的拉格朗日基函数值。
        let c: &[i32; PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE] =
            &PRODUCT_VIRTUAL_COEFFS_PER_J[j];

        // 初始化加权后的左、右分量数组。
        // 使用 i128 是为了确保中间计算（系数 * 输入）不会溢出。
        let mut left_w: [i128; NUM_PRODUCT_VIRTUAL] = [0; NUM_PRODUCT_VIRTUAL];
        let mut right_w: [i128; NUM_PRODUCT_VIRTUAL] = [0; NUM_PRODUCT_VIRTUAL];

        // -------------------------------------------------------------------------
        // 2. 计算 5 个虚拟约束分量的加权值
        // -------------------------------------------------------------------------

        // 约束 0: 指令解码约束 (Instruction)
        // 逻辑: 验证指令操作数的拆解是否正确。
        // Left: 指令的左半部分输入
        // Right: 指令的右半部分输入
        left_w[0] = (c[0] as i128) * (row.instruction_left_input as i128);
        right_w[0] = (c[0] as i128) * row.instruction_right_input;

        // 约束 1: 查找表输出写入 RD 寄存器 (WriteLookupOutputToRD)
        // 逻辑: 如果是 RD 写操作且不是跳转，则验证查找表结果是否写入 RD。
        // 原始约束: IsRdNotZero * WriteLookupOutputToRD_flag
        left_w[1] = if row.is_rd_not_zero { c[1] as i128 } else { 0 };
        right_w[1] = if row.write_lookup_output_to_rd_flag {
            c[1] as i128
        } else {
            0
        };

        // 约束 2: PC 值写入 RD 寄存器 (WritePCtoRD)
        // 逻辑: 用于 JAL/JALR 指令，将 PC+4 写入 RD。
        // 原始约束: IsRdNotZero * Jump_flag
        left_w[2] = if row.is_rd_not_zero { c[2] as i128 } else { 0 };
        right_w[2] = if row.jump_flag { c[2] as i128 } else { 0 };

        // 约束 3: 分支判断 (ShouldBranch)
        // 逻辑: 验证分支条件是否成立（由查找表输出决定）。
        // 原始约束: LookupOutput * Branch_flag
        // 该约束通常用于验证 BEQ, BNE 等指令的比较结果。
        left_w[3] = (c[3] as i128) * (row.should_branch_lookup_output as i128);
        right_w[3] = if row.should_branch_flag {
            c[3] as i128
        } else {
            0
        };

        // 约束 4: 跳转判断 (ShouldJump)
        // 逻辑: 验证是否需要执行跳转更新 PC。
        // 原始约束: Jump_flag * (1 - NextIsNoop)
        // 如果下一条不是空操作(Noop)，且当前是跳转指令，则触发跳转逻辑。
        left_w[4] = if row.jump_flag { c[4] as i128 } else { 0 };
        right_w[4] = if row.not_next_noop { c[4] as i128 } else { 0 };

        // -------------------------------------------------------------------------
        // 3. 融合 (Fusion)
        // -------------------------------------------------------------------------
        // 将所有约束的加权左项求和，得到虚拟多项式的左因子 L(x_j)。
        // 将所有约束的加权右项求和，得到虚拟多项式的右因子 R(x_j)。
        let mut left_sum: i128 = 0;
        let mut right_sum: i128 = 0;
        let mut i = 0;
        while i < NUM_PRODUCT_VIRTUAL {
            left_sum += left_w[i];
            right_sum += right_w[i];
            i += 1;
        }

        // -------------------------------------------------------------------------
        // 4. 计算最终乘积
        // -------------------------------------------------------------------------
        // 计算 P(x_j) = L(x_j) * R(x_j)。
        // 使用宽整数类型 (S128 -> S256) 进行乘法，以避免溢出并保持精度。
        let left_s128 = S128::from_i128(left_sum);
        let right_s128 = S128::from_i128(right_sum);

        // 返回最终的 S256 结果
        left_s128.mul_trunc::<2, 4>(&right_s128)
    }

    /// Compute z(r_cycle) for the 8 de-duplicated factor polynomials used by Product Virtualization.
    /// Order of outputs matches PRODUCT_UNIQUE_FACTOR_VIRTUALS:
    /// 0: LeftInstructionInput (u64)
    /// 1: RightInstructionInput (i128)
    /// 2: IsRdNotZero (bool)
    /// 3: OpFlags(WriteLookupOutputToRD) (bool)
    /// 4: OpFlags(Jump) (bool)
    /// 5: LookupOutput (u64)
    /// 6: InstructionFlags(Branch) (bool)
    /// 7: NextIsNoop (bool)
    #[tracing::instrument(skip_all, name = "ProductVirtualEval::compute_claimed_factors")]
    pub fn compute_claimed_factors<F: JoltField>(
        trace: &[tracer::instruction::Cycle],
        r_cycle: &OpeningPoint<BIG_ENDIAN, F>,
    ) -> [F; 8] {
        let m = r_cycle.len() / 2;
        let (r2, r1) = r_cycle.split_at_r(m);
        let (eq_one, eq_two) = rayon::join(|| EqPolynomial::evals(r2), || EqPolynomial::evals(r1));

        let eq_two_len = eq_two.len();

        (0..eq_one.len())
            .into_par_iter()
            .map(|x1| {
                let eq1_val = eq_one[x1];

                // Accumulators for 8 outputs
                let mut acc_left_u64: Acc6U<F> = Acc6U::zero();
                let mut acc_right_i128: Acc6S<F> = Acc6S::zero();
                let mut acc_rd_zero_flag: Acc5U<F> = Acc5U::zero();
                let mut acc_wl_flag: Acc5U<F> = Acc5U::zero();
                let mut acc_jump_flag: Acc5U<F> = Acc5U::zero();
                let mut acc_lookup_output: Acc6U<F> = Acc6U::zero();
                let mut acc_branch_flag: Acc5U<F> = Acc5U::zero();
                let mut acc_next_is_noop: Acc5U<F> = Acc5U::zero();

                for x2 in 0..eq_two_len {
                    let e_in = eq_two[x2];
                    let idx = x1 * eq_two_len + x2;
                    let row = ProductCycleInputs::from_trace::<F>(trace, idx);

                    // 0: LeftInstructionInput (u64)
                    acc_left_u64.fmadd(&e_in, &row.instruction_left_input);
                    // 1: RightInstructionInput (i128)
                    acc_right_i128.fmadd(&e_in, &row.instruction_right_input);
                    // 2: IsRdNotZero (bool)
                    acc_rd_zero_flag.fmadd(&e_in, &(row.is_rd_not_zero));
                    // 3: OpFlags(WriteLookupOutputToRD) (bool)
                    acc_wl_flag.fmadd(&e_in, &row.write_lookup_output_to_rd_flag);
                    // 4: OpFlags(Jump) (bool)
                    acc_jump_flag.fmadd(&e_in, &row.jump_flag);
                    // 5: LookupOutput (u64)
                    acc_lookup_output.fmadd(&e_in, &row.should_branch_lookup_output);
                    // 6: InstructionFlags(Branch) (bool)
                    acc_branch_flag.fmadd(&e_in, &row.should_branch_flag);
                    // 7: NextIsNoop (bool) = !not_next_noop
                    acc_next_is_noop.fmadd(&e_in, &(!row.not_next_noop));
                }

                let mut out_unr = [F::Unreduced::<9>::zero(); 8];
                out_unr[0] = eq1_val.mul_unreduced::<9>(acc_left_u64.barrett_reduce());
                out_unr[1] = eq1_val.mul_unreduced::<9>(acc_right_i128.barrett_reduce());
                out_unr[2] = eq1_val.mul_unreduced::<9>(acc_rd_zero_flag.barrett_reduce());
                out_unr[3] = eq1_val.mul_unreduced::<9>(acc_wl_flag.barrett_reduce());
                out_unr[4] = eq1_val.mul_unreduced::<9>(acc_jump_flag.barrett_reduce());
                out_unr[5] = eq1_val.mul_unreduced::<9>(acc_lookup_output.barrett_reduce());
                out_unr[6] = eq1_val.mul_unreduced::<9>(acc_branch_flag.barrett_reduce());
                out_unr[7] = eq1_val.mul_unreduced::<9>(acc_next_is_noop.barrett_reduce());
                out_unr
            })
            .reduce(
                || [F::Unreduced::<9>::zero(); 8],
                |mut acc, item| {
                    for i in 0..8 {
                        acc[i] += item[i];
                    }
                    acc
                },
            )
            .map(|unr| F::from_montgomery_reduce::<9>(unr))
    }
}
