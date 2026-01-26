//! R1CS input catalog and per-cycle typed views
//!
//! - Canonical enumeration and ordering of all virtual inputs consumed by the
//!   Spartan outer sumcheck: `JoltR1CSInputs` and `ALL_R1CS_INPUTS`. Provides
//!   indices and conversions to `VirtualPolynomial` and to `OpeningId`.
//! - Materialized, single-cycle views sourced from the execution trace:
//!   - `R1CSCycleInputs`: full row used by uniform R1CS and shift constraints;
//!     built via `from_trace` with exact integer semantics.
//!   - `ProductCycleInputs`: minimal tuple for the product virtualization sumcheck;
//!     the de-duplicated factor list is `PRODUCT_UNIQUE_FACTOR_VIRTUALS`.
//!
//! Maintainers: keep the enum order, `ALL_R1CS_INPUTS`, and `to_index` in sync.
//! Changes here affect `r1cs::constraints` (row shapes) and `r1cs::evaluation`
//! (typed evaluators and claim computation).

use crate::poly::opening_proof::{OpeningId, SumcheckId};
use crate::zkvm::bytecode::BytecodePreprocessing;
use crate::zkvm::instruction::{
    CircuitFlags, Flags, InstructionFlags, LookupQuery, NUM_CIRCUIT_FLAGS,
};
use crate::zkvm::witness::VirtualPolynomial;

use crate::field::JoltField;
use ark_ff::biginteger::{S128, S64};
use common::constants::XLEN;
use std::fmt::Debug;
use tracer::instruction::Cycle;

use strum::IntoEnumIterator;

/// Inputs to the Spartan outer sumcheck. All is virtual, each produce a claim for later stages
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum JoltR1CSInputs {
    PC,                    // (bytecode raf)
    UnexpandedPC,          // (bytecode rv)
    Imm,                   // (bytecode rv)
    RamAddress,            // (RAM raf)
    Rs1Value,              // (registers rv)
    Rs2Value,              // (registers rv)
    RdWriteValue,          // (registers wv)
    RamReadValue,          // (RAM rv)
    RamWriteValue,         // (RAM wv)
    LeftInstructionInput,  // (instruction input)
    RightInstructionInput, // (instruction input)
    LeftLookupOperand,     // (instruction raf)
    RightLookupOperand,    // (instruction raf)
    Product,               // (product virtualization)
    WriteLookupOutputToRD, // (product virtualization)
    WritePCtoRD,           // (product virtualization)
    ShouldBranch,          // (product virtualization)
    NextUnexpandedPC,      // (shift sumcheck)
    NextPC,                // (shift sumcheck)
    NextIsVirtual,         // (shift sumcheck)
    NextIsFirstInSequence, // (shift sumcheck)
    LookupOutput,          // (instruction rv)
    ShouldJump,            // (product virtualization)
    OpFlags(CircuitFlags),
}

pub const NUM_R1CS_INPUTS: usize = ALL_R1CS_INPUTS.len();
/// This const serves to define a canonical ordering over inputs (and thus indices
/// for each input). This is needed for sumcheck.
pub const ALL_R1CS_INPUTS: [JoltR1CSInputs; 36] = [
    JoltR1CSInputs::LeftInstructionInput,
    JoltR1CSInputs::RightInstructionInput,
    JoltR1CSInputs::Product,
    JoltR1CSInputs::WriteLookupOutputToRD,
    JoltR1CSInputs::WritePCtoRD,
    JoltR1CSInputs::ShouldBranch,
    JoltR1CSInputs::PC,
    JoltR1CSInputs::UnexpandedPC,
    JoltR1CSInputs::Imm,
    JoltR1CSInputs::RamAddress,
    JoltR1CSInputs::Rs1Value,
    JoltR1CSInputs::Rs2Value,
    JoltR1CSInputs::RdWriteValue,
    JoltR1CSInputs::RamReadValue,
    JoltR1CSInputs::RamWriteValue,
    JoltR1CSInputs::LeftLookupOperand,
    JoltR1CSInputs::RightLookupOperand,
    JoltR1CSInputs::NextUnexpandedPC,
    JoltR1CSInputs::NextPC,
    JoltR1CSInputs::NextIsVirtual,
    JoltR1CSInputs::NextIsFirstInSequence,
    JoltR1CSInputs::LookupOutput,
    JoltR1CSInputs::ShouldJump,
    JoltR1CSInputs::OpFlags(CircuitFlags::AddOperands),
    JoltR1CSInputs::OpFlags(CircuitFlags::SubtractOperands),
    JoltR1CSInputs::OpFlags(CircuitFlags::MultiplyOperands),
    JoltR1CSInputs::OpFlags(CircuitFlags::Load),
    JoltR1CSInputs::OpFlags(CircuitFlags::Store),
    JoltR1CSInputs::OpFlags(CircuitFlags::Jump),
    JoltR1CSInputs::OpFlags(CircuitFlags::WriteLookupOutputToRD),
    JoltR1CSInputs::OpFlags(CircuitFlags::VirtualInstruction),
    JoltR1CSInputs::OpFlags(CircuitFlags::Assert),
    JoltR1CSInputs::OpFlags(CircuitFlags::DoNotUpdateUnexpandedPC),
    JoltR1CSInputs::OpFlags(CircuitFlags::Advice),
    JoltR1CSInputs::OpFlags(CircuitFlags::IsCompressed),
    JoltR1CSInputs::OpFlags(CircuitFlags::IsFirstInSequence),
];

impl JoltR1CSInputs {
    /// The total number of unique constraint inputs
    pub const fn num_inputs() -> usize {
        NUM_R1CS_INPUTS
    }

    /// Converts an index to the corresponding constraint input.
    pub fn from_index(index: usize) -> Self {
        ALL_R1CS_INPUTS[index]
    }

    /// Converts a constraint input to its index in the canonical
    /// ordering over inputs given by `ALL_R1CS_INPUTS`.
    pub const fn to_index(&self) -> usize {
        match self {
            JoltR1CSInputs::LeftInstructionInput => 0,
            JoltR1CSInputs::RightInstructionInput => 1,
            JoltR1CSInputs::Product => 2,
            JoltR1CSInputs::WriteLookupOutputToRD => 3,
            JoltR1CSInputs::WritePCtoRD => 4,
            JoltR1CSInputs::ShouldBranch => 5,
            JoltR1CSInputs::PC => 6,
            JoltR1CSInputs::UnexpandedPC => 7,
            JoltR1CSInputs::Imm => 8,
            JoltR1CSInputs::RamAddress => 9,
            JoltR1CSInputs::Rs1Value => 10,
            JoltR1CSInputs::Rs2Value => 11,
            JoltR1CSInputs::RdWriteValue => 12,
            JoltR1CSInputs::RamReadValue => 13,
            JoltR1CSInputs::RamWriteValue => 14,
            JoltR1CSInputs::LeftLookupOperand => 15,
            JoltR1CSInputs::RightLookupOperand => 16,
            JoltR1CSInputs::NextUnexpandedPC => 17,
            JoltR1CSInputs::NextPC => 18,
            JoltR1CSInputs::NextIsVirtual => 19,
            JoltR1CSInputs::NextIsFirstInSequence => 20,
            JoltR1CSInputs::LookupOutput => 21,
            JoltR1CSInputs::ShouldJump => 22,
            JoltR1CSInputs::OpFlags(CircuitFlags::AddOperands) => 23,
            JoltR1CSInputs::OpFlags(CircuitFlags::SubtractOperands) => 24,
            JoltR1CSInputs::OpFlags(CircuitFlags::MultiplyOperands) => 25,
            JoltR1CSInputs::OpFlags(CircuitFlags::Load) => 26,
            JoltR1CSInputs::OpFlags(CircuitFlags::Store) => 27,
            JoltR1CSInputs::OpFlags(CircuitFlags::Jump) => 28,
            JoltR1CSInputs::OpFlags(CircuitFlags::WriteLookupOutputToRD) => 29,
            JoltR1CSInputs::OpFlags(CircuitFlags::VirtualInstruction) => 30,
            JoltR1CSInputs::OpFlags(CircuitFlags::Assert) => 31,
            JoltR1CSInputs::OpFlags(CircuitFlags::DoNotUpdateUnexpandedPC) => 32,
            JoltR1CSInputs::OpFlags(CircuitFlags::Advice) => 33,
            JoltR1CSInputs::OpFlags(CircuitFlags::IsCompressed) => 34,
            JoltR1CSInputs::OpFlags(CircuitFlags::IsFirstInSequence) => 35,
        }
    }
}

impl From<&JoltR1CSInputs> for VirtualPolynomial {
    fn from(input: &JoltR1CSInputs) -> Self {
        match input {
            JoltR1CSInputs::PC => VirtualPolynomial::PC,
            JoltR1CSInputs::UnexpandedPC => VirtualPolynomial::UnexpandedPC,
            JoltR1CSInputs::Imm => VirtualPolynomial::Imm,
            JoltR1CSInputs::RamAddress => VirtualPolynomial::RamAddress,
            JoltR1CSInputs::Rs1Value => VirtualPolynomial::Rs1Value,
            JoltR1CSInputs::Rs2Value => VirtualPolynomial::Rs2Value,
            JoltR1CSInputs::RdWriteValue => VirtualPolynomial::RdWriteValue,
            JoltR1CSInputs::RamReadValue => VirtualPolynomial::RamReadValue,
            JoltR1CSInputs::RamWriteValue => VirtualPolynomial::RamWriteValue,
            JoltR1CSInputs::LeftLookupOperand => VirtualPolynomial::LeftLookupOperand,
            JoltR1CSInputs::RightLookupOperand => VirtualPolynomial::RightLookupOperand,
            JoltR1CSInputs::Product => VirtualPolynomial::Product,
            JoltR1CSInputs::NextUnexpandedPC => VirtualPolynomial::NextUnexpandedPC,
            JoltR1CSInputs::NextPC => VirtualPolynomial::NextPC,
            JoltR1CSInputs::LookupOutput => VirtualPolynomial::LookupOutput,
            JoltR1CSInputs::ShouldJump => VirtualPolynomial::ShouldJump,
            JoltR1CSInputs::ShouldBranch => VirtualPolynomial::ShouldBranch,
            JoltR1CSInputs::WritePCtoRD => VirtualPolynomial::WritePCtoRD,
            JoltR1CSInputs::WriteLookupOutputToRD => VirtualPolynomial::WriteLookupOutputToRD,
            JoltR1CSInputs::OpFlags(flag) => VirtualPolynomial::OpFlags(*flag),
            JoltR1CSInputs::LeftInstructionInput => VirtualPolynomial::LeftInstructionInput,
            JoltR1CSInputs::RightInstructionInput => VirtualPolynomial::RightInstructionInput,
            JoltR1CSInputs::NextIsVirtual => VirtualPolynomial::NextIsVirtual,
            JoltR1CSInputs::NextIsFirstInSequence => VirtualPolynomial::NextIsFirstInSequence,
        }
    }
}

impl From<&JoltR1CSInputs> for OpeningId {
    fn from(input: &JoltR1CSInputs) -> Self {
        let poly = VirtualPolynomial::from(input);
        OpeningId::Virtual(poly, SumcheckId::SpartanOuter)
    }
}

/// Fully materialized, typed view of all R1CS inputs for a single row (cycle).
/// Filled once and reused to evaluate all constraints without re-reading the trace.
/// Total size: 208 bytes, alignment: 16 bytes
#[derive(Clone, Debug)]
pub struct R1CSCycleInputs {
    /// Left instruction input as a u64 bit-pattern.
    /// Typically `Rs1Value` or the current `UnexpandedPC`, depending on `CircuitFlags`.
    /// 指令的左操作数，作为 u64 位模式存储。
    /// 根据电路标志 (`CircuitFlags`) 的不同，这通常是源寄存器 1 的值 (`Rs1Value`) 或者当前的程序计数器 `UnexpandedPC` (例如在 AUIPC 指令中)。
    pub left_input: u64,

    /// Right instruction input as signed-magnitude `S64`.
    /// Typically `Imm` or `Rs2Value` with exact integer semantics.
    /// 指令的右操作数，使用符号-幅值表示法 (`S64`) 存储。
    /// 通常是立即数 (`Imm`) 或者源寄存器 2 的值 (`Rs2Value`)，保留了精确的整数语义（用于算术运算）。
    pub right_input: S64,

    /// Signed-magnitude `S128` product consistent with the `Product` witness.
    /// Computed from `left_input` × `right_input` using the same truncation semantics as the witness.
    /// 符号-幅值表示的 128 位乘积 (`S128`)，与电路中的 `Product` 见证变量一致。
    /// 它是通过 `left_input` × `right_input` 计算得出的，并且使用了与见证生成阶段完全相同的截断语义（用于处理乘法溢出等情况）。
    pub product: S128,

    /// Left lookup operand (u64) for the instruction lookup query.
    /// Matches `LeftLookupOperand` virtual polynomial semantics.
    /// 用于指令查找查询（Lookup Query）的左操作数 (u64)。
    /// 对应于 `LeftLookupOperand` 虚拟多项式的语义。在位运算等操作中，这是查找表的输入之一。
    pub left_lookup: u64,

    /// Right lookup operand (u128) for the instruction lookup query.
    /// Full-width integer encoding used by add/sub/mul/advice cases.
    /// 用于指令查找查询的右操作数 (u128)。
    /// 在加法/减法/乘法/Advice 等情况中使用的全宽整数编码。
    pub right_lookup: u128,

    /// Instruction lookup output (u64) for this cycle.
    /// 当前周期的指令查找输出 (u64)。
    /// 这是查找表返回的结果，通常也是指令的计算结果（如 ADD 的结果，或 EQ 的布尔值）。
    pub lookup_output: u64,

    /// Value read from Rs1 in this cycle.
    /// 本周期从源寄存器 Rs1 读取的值。
    pub rs1_read_value: u64,

    /// Value read from Rs2 in this cycle.
    /// 本周期从源寄存器 Rs2 读取的值。
    pub rs2_read_value: u64,

    /// Value written to Rd in this cycle.
    /// 本周期写入目标寄存器 Rd 的值。
    pub rd_write_value: u64,

    /// RAM address accessed this cycle.
    /// 本周期访问的 RAM 地址。
    pub ram_addr: u64,

    /// RAM read value for `Read`, pre-write value for `Write`, or 0 for `NoOp`.
    /// 对于 `Read` 操作，这是读取到的值；对于 `Write` 操作，这是写入前的旧值（用于内存一致性检查）；对于 `NoOp` (无访存)，为 0。
    pub ram_read_value: u64,

    /// RAM write value: equals read value for `Read`, post-write value for `Write`, or 0 for `NoOp`.
    /// 内存写入值：对于 `Read` 操作，它等于读取值（表示值未变）；对于 `Write` 操作，它是写入后的新值；对于 `NoOp`，为 0。
    pub ram_write_value: u64,

    /// Expanded PC used by bytecode instance.
    /// 字节码实例使用的扩展 PC（物理 PC）。包含程序段前缀等信息，用于从只读内存中取指。
    pub pc: u64,

    /// Expanded PC for next cycle, or 0 if this is the last cycle in the domain.
    /// 下一个周期的扩展 PC。如果这是执行域的最后一个周期，则为 0。
    pub next_pc: u64,

    /// Unexpanded PC (normalized instruction address) for this cycle.
    /// 本周期的未扩展 PC（标准化的指令地址）。這是 CPU 视角的虚拟地址，例如 0x1000, 0x1004。
    pub unexpanded_pc: u64,

    /// Unexpanded PC for next cycle, or 0 if this is the last cycle in the domain.
    /// 下一个周期的未扩展 PC。用于验证跳转逻辑是否正确。
    pub next_unexpanded_pc: u64,

    /// Immediate operand as signed-magnitude `S64`.
    /// 立即数操作数，使用符号-幅值 `S64` 存储。
    pub imm: S64,

    /// Per-instruction circuit flags indexed by `CircuitFlags`.
    /// 每个指令的电路标志位数组，通过 `CircuitFlags`枚举索引。
    /// 包含了如 `IsAdd`, `IsLoad`, `IsJump` 等布尔标志，用于控制约束系统的选择逻辑。
    pub flags: [bool; NUM_CIRCUIT_FLAGS],

    /// `IsNoop` flag for the next cycle (false for last cycle).
    /// 下一个周期是否为 `NoOp`（空操作）的标志。用于处理 Padding 行或程序结束的情况。
    pub next_is_noop: bool,

    /// Derived: `Jump && !NextIsNoop`.
    /// 派生字段：是否应该发生跳转。逻辑为 `Jump` 指令且下一条指令不是填充的 `NoOp`。
    pub should_jump: bool,

    /// Derived: `Branch && (LookupOutput == 1)`.
    /// 派生字段：是否应该执行分支跳转。逻辑为当前是分支指令 (`Branch`) 且比较结果为真 (`LookupOutput == 1`)。
    pub should_branch: bool,

    /// `IsRdNotZero` && ` `WriteLookupOutputToRD`
    /// 派生字段：是否将 Lookup 的结果写入 RD 寄存器。
    /// 仅当指令需要写回结果 (`WriteLookupOutputToRD`) 且目标寄存器索引非零 (`IsRdNotZero`, 即不是 x0) 时为真。
    pub write_lookup_output_to_rd_addr: bool,

    /// `IsRdNotZero` && `Jump`
    /// 派生字段：是否将 PC 值（通常是返回地址）写入 RD 寄存器。
    /// 用于 JAL/JALR 指令，且仅当 RD != x0 时为真。
    pub write_pc_to_rd_addr: bool,

    /// `VirtualInstruction` flag for the next cycle (false for last cycle).
    /// 下一个周期是否属于虚拟指令及其序列的标志。
    /// Jolt 将复杂指令拆解为微指令序列，此标志用于维护序列内部 PC 的连续性。
    pub next_is_virtual: bool,

    /// `FirstInSequence` flag for the next cycle (false for last cycle).
    /// 下一个周期是否是新指令序列的第一条指令。用于界定微指令序列的边界。
    pub next_is_first_in_sequence: bool,
}

impl R1CSCycleInputs {
    /// Build directly from the execution trace and preprocessing,
    /// mirroring the optimized semantics used in `compute_claimed_r1cs_input_evals`.
    /// 直接从执行轨迹（Trace）和预处理数据中构建单步的 R1CS 输入数据。
    ///
    /// 尽管这个函数在宿主机的运行时(Runtime)执行，但它生成的结构体必须严格对应
    /// 电路中的 Witness (见证数据)。这里的每一个字段稍后都会变成多项式的一部分。
    pub fn from_trace<F>(
        bytecode_preprocessing: &BytecodePreprocessing,
        trace: &[Cycle],
        t: usize,
    ) -> Self
    where
        F: JoltField,
    {
        let len = trace.len();
        // 获取当前时钟周期 t 的完整状态（指令、寄存器、RAM等）
        let cycle = &trace[t];
        // 获取当前解码后的指令信息
        let instr = cycle.instruction();
        // 获取电路标志位视图 (如 IsAdd, IsLoad 等)
        let flags_view = instr.circuit_flags();
        // 获取指令通用属性 (如 IsBranch, IsNoop 等)
        let instruction_flags = instr.instruction_flags();
        // 获取标准化后的指令信息 (包含操作码、操作数等)
        let norm = instr.normalize();

        // 获取下一个 Cycle 的引用，用于计算状态转移约束 (如 NextPC)
        // 如果当前是最后一个 Cycle，则 next_cycle 为 None
        let next_cycle = if t + 1 < len {
            Some(&trace[t + 1])
        } else {
            None
        };

        // --- 1. 指令输入与算术验证 ---

        // 根据指令类型提取左/右操作数。
        // 例如：对于 ADD, left=rs1, right=rs2；对于 ADDI, left=rs1, right=imm
        let (left_input, right_i128) = LookupQuery::<XLEN>::to_instruction_inputs(cycle);

        // 将输入转换为电路使用的有符号 64 位整数格式 (S64)
        let left_s64: S64 = S64::from_u64(left_input);

        // 检查右操作数范围并转换
        let right_mag = right_i128.unsigned_abs();
        debug_assert!(
            right_mag <= u64::MAX as u128,
            "RightInstructionInput overflow at row {t}: |{right_i128}| > 2^64-1"
        );
        let right_input = S64::from_u64_with_sign(right_mag as u64, right_i128 >= 0);

        // 计算乘积 witness (Product)。
        // 即使当前指令不是乘法，我们也会计算这个值，以保持电路结构的统一性 (Uniformity)。
        // 如果当前是 ADD 指令，R1CS 约束会忽略这个 product 字段；但如果是 MUL 指令，则会验证它是否正确。
        let right_s128: S128 = S128::from_i128(right_i128);
        let product: S128 = left_s64.mul_trunc::<2, 2>(&right_s128);

        // --- 2. 查找表 (Lookup) ---

        // 提取用于查表的操作数。对于位运算等复杂操作，Jolt 使用查表法。
        let (left_lookup, right_lookup) = LookupQuery::<XLEN>::to_lookup_operands(cycle);
        // 提取查表的结果（即运算结果）
        let lookup_output = LookupQuery::<XLEN>::to_lookup_output(cycle);

        // --- 3. 寄存器状态 ---

        // 获取读写的寄存器值。unwrap_or_default 处理那些不访问寄存器的指令情况。
        let rs1_read_value = cycle.rs1_read().unwrap_or_default().1;
        let rs2_read_value = cycle.rs2_read().unwrap_or_default().1;
        let rd_write_value = cycle.rd_write().unwrap_or_default().2;

        // --- 4. 内存 (RAM) 状态 ---

        let ram_addr = cycle.ram_access().address() as u64;
        // 分别提取内存读写前后的值：
        // - Read: 读取到的值既是 read_value 也是 write_value (状态不变)
        // - Write: pre_value 是旧值 (read), post_value 是新值 (write)
        let (ram_read_value, ram_write_value) = match cycle.ram_access() {
            tracer::instruction::RAMAccess::Read(r) => (r.value, r.value),
            tracer::instruction::RAMAccess::Write(w) => (w.pre_value, w.post_value),
            tracer::instruction::RAMAccess::NoOp => (0u64, 0u64),
        };

        // --- 5. 程序计数器 (PC) ---

        // 从预处理数据中获取当前指令的真实 PC
        let pc = bytecode_preprocessing.get_pc(cycle) as u64;
        // 确定 NextPC：如果有下一条指令则取之，否则为 0
        let next_pc = if let Some(nc) = next_cycle {
            bytecode_preprocessing.get_pc(nc) as u64
        } else {
            0u64
        };
        // UnexpandedPC 是未展开的原始指令地址
        let unexpanded_pc = norm.address as u64;
        let next_unexpanded_pc = if let Some(nc) = next_cycle {
            nc.instruction().normalize().address as u64
        } else {
            0u64
        };

        // --- 6. 立即数 (Immediate) ---

        let imm_i128 = norm.operands.imm;
        let imm_mag = imm_i128.unsigned_abs();
        debug_assert!(
            imm_mag <= u64::MAX as u128,
            "Imm overflow at row {t}: |{imm_i128}| > 2^64-1"
        );
        let imm = S64::from_u64_with_sign(imm_mag as u64, imm_i128 >= 0);

        // --- 7. 标志位与逻辑计算 ---

        // 将 Enum Map 形式的标志位展平为布尔数组。这是 R1CS 矩阵中 "One-Hot" 编码的基础。
        let mut flags = [false; NUM_CIRCUIT_FLAGS];
        for flag in CircuitFlags::iter() {
            flags[flag] = flags_view[flag];
        }

        // 判断下一条指令是否是 NoOp
        let next_is_noop = if let Some(nc) = next_cycle {
            nc.instruction().instruction_flags()[InstructionFlags::IsNoop]
        } else {
            false // 没有下一条指令，自然不是 NoOp
        };

        // 派生字段：ShouldJump
        // 逻辑：当前是 Jump 指令 且 下一条指令是有效的（非 NoOp）
        let should_jump = flags_view[CircuitFlags::Jump] && !next_is_noop;

        // 派生字段：ShouldBranch
        // 逻辑：当前是 Branch 指令 且 查表结果为 1 (条件成立)
        let should_branch = instruction_flags[InstructionFlags::Branch] && (lookup_output == 1);

        // 派生字段：写回 RD 的控制位
        // 只有当目标寄存器 RD 不是 x0 (零寄存器) 时，才允许写入。

        // 情况 A: 将 Lookup 结果写回 RD (用于普通算术指令)
        let write_lookup_output_to_rd_addr = flags_view[CircuitFlags::WriteLookupOutputToRD]
            && instruction_flags[InstructionFlags::IsRdNotZero];

        // 情况 B: 将 PC 写回 RD (用于 JAL/JALR 跳转链接)
        let write_pc_to_rd_addr =
            flags_view[CircuitFlags::Jump] && instruction_flags[InstructionFlags::IsRdNotZero];

        // 检查下一条指令是否属于虚拟指令序列 (Virtual Instruction Sequence)
        // Jolt 会把某些复杂指令拆解为多个 Micro-ops
        let (next_is_virtual, next_is_first_in_sequence) = if let Some(nc) = next_cycle {
            let flags = nc.instruction().circuit_flags();
            (
                flags[CircuitFlags::VirtualInstruction],
                flags[CircuitFlags::IsFirstInSequence],
            )
        } else {
            (false, false)
        };

        Self {
            left_input,
            right_input,
            product,
            left_lookup,
            right_lookup,
            lookup_output,
            rs1_read_value,
            rs2_read_value,
            rd_write_value,
            ram_addr,
            ram_read_value,
            ram_write_value,
            pc,
            next_pc,
            unexpanded_pc,
            next_unexpanded_pc,
            imm,
            flags,
            next_is_noop,
            should_jump,
            should_branch,
            write_lookup_output_to_rd_addr,
            write_pc_to_rd_addr,
            next_is_virtual,
            next_is_first_in_sequence,
        }
    }


    #[cfg(test)]
    pub fn get_input_value(&self, input: JoltR1CSInputs) -> i128 {
        match input {
            JoltR1CSInputs::PC => self.pc as i128,
            JoltR1CSInputs::UnexpandedPC => self.unexpanded_pc as i128,
            JoltR1CSInputs::Imm => self.imm.to_i128(),
            JoltR1CSInputs::RamAddress => self.ram_addr as i128,
            JoltR1CSInputs::Rs1Value => self.rs1_read_value as i128,
            JoltR1CSInputs::Rs2Value => self.rs2_read_value as i128,
            JoltR1CSInputs::RdWriteValue => self.rd_write_value as i128,
            JoltR1CSInputs::RamReadValue => self.ram_read_value as i128,
            JoltR1CSInputs::RamWriteValue => self.ram_write_value as i128,
            JoltR1CSInputs::LeftInstructionInput => self.left_input as i128,
            JoltR1CSInputs::RightInstructionInput => self.right_input.to_i128(),
            JoltR1CSInputs::LeftLookupOperand => self.left_lookup as i128,
            JoltR1CSInputs::RightLookupOperand => self.right_lookup as i128,
            JoltR1CSInputs::Product => self.product.to_i128().expect("product too large for i128"),
            JoltR1CSInputs::WriteLookupOutputToRD => self.write_lookup_output_to_rd_addr as i128,
            JoltR1CSInputs::WritePCtoRD => self.write_pc_to_rd_addr as i128,
            JoltR1CSInputs::ShouldBranch => self.should_branch as i128,
            JoltR1CSInputs::NextUnexpandedPC => self.next_unexpanded_pc as i128,
            JoltR1CSInputs::NextPC => self.next_pc as i128,
            JoltR1CSInputs::NextIsVirtual => self.next_is_virtual as i128,
            JoltR1CSInputs::NextIsFirstInSequence => self.next_is_first_in_sequence as i128,
            JoltR1CSInputs::LookupOutput => self.lookup_output as i128,
            JoltR1CSInputs::ShouldJump => self.should_jump as i128,
            JoltR1CSInputs::OpFlags(flag) => self.flags[flag] as i128,
        }
    }
}

/// Canonical, de-duplicated list of product-virtual factor polynomials used by
/// the Product Virtualization stage.
/// Order:
/// 0: LeftInstructionInput
/// 1: RightInstructionInput
/// 2: InstructionFlags(IsRdNotZero)
/// 3: OpFlags(WriteLookupOutputToRD)
/// 4: OpFlags(Jump)
/// 5: LookupOutput
/// 6: InstructionFlags(Branch)
/// 7: NextIsNoop
pub const PRODUCT_UNIQUE_FACTOR_VIRTUALS: [VirtualPolynomial; 8] = [
    VirtualPolynomial::LeftInstructionInput,
    VirtualPolynomial::RightInstructionInput,
    VirtualPolynomial::InstructionFlags(InstructionFlags::IsRdNotZero),
    VirtualPolynomial::OpFlags(CircuitFlags::WriteLookupOutputToRD),
    VirtualPolynomial::OpFlags(CircuitFlags::Jump),
    VirtualPolynomial::LookupOutput,
    VirtualPolynomial::InstructionFlags(InstructionFlags::Branch),
    VirtualPolynomial::NextIsNoop,
];

/// Minimal, unified view for the Product-virtualization round: the 5 product pairs
/// (left, right) materialized from the trace for a single cycle.
/// Total size is small; we keep primitive representations that match witness generation.
#[derive(Clone, Debug)]
pub struct ProductCycleInputs {
    // 16-byte aligned
    /// Instruction: LeftInstructionInput × RightInstructionInput (right input as i128)
    pub instruction_right_input: i128,

    // 8-byte aligned
    pub instruction_left_input: u64,
    /// ShouldBranch: LookupOutput × Branch_flag (left side)
    pub should_branch_lookup_output: u64,

    // 1-byte fields
    /// WriteLookupOutputToRD right flag (boolean)
    pub write_lookup_output_to_rd_flag: bool,
    /// Jump flag used by both WritePCtoRD (right) and ShouldJump (left)
    pub jump_flag: bool,
    /// ShouldBranch right flag (boolean)
    pub should_branch_flag: bool,
    /// ShouldJump right flag (1 - NextIsNoop)
    pub not_next_noop: bool,
    /// IsRdNotZero instruction flag (boolean)
    pub is_rd_not_zero: bool,
}

impl ProductCycleInputs {
    /// 从执行痕迹 (Trace) 构建电路输入。
    /// 这个函数镜像了 "Product-Virtualization" 见证生成过程中的语义。
    /// 即：Prover 在生成证明时，需要用同样的方式看待数据。
    pub fn from_trace<F>(trace: &[Cycle], t: usize) -> Self
    where
        F: JoltField,
    {
        // 1. 获取上下文信息
        let len = trace.len();      // 总步数
        let cycle = &trace[t];      // 当前时间步 t 的 CPU 状态
        let instr = cycle.instruction(); // 获取当前执行的指令信息

        // 2. 提取指令标志位 (Flags)
        // flags_view: 对应电路层面的标志 (如 Jump, WriteToRD)
        let flags_view = instr.circuit_flags();
        // instruction_flags: 对应指令语义的标志 (如 Branch, IsNoop, IsRdNotZero)
        let instruction_flags = instr.instruction_flags();

        // 3. 提取指令操作数 (Inputs)
        // 对应数学公式中的 input_x, input_y (即 RS1, RS2 的值)
        let (left_input, right_input) = LookupQuery::<XLEN>::to_instruction_inputs(cycle);

        // 4. 提取 Lookup 输出
        // 如果是分支指令，这可能是分支偏移量；如果是计算指令，这是计算结果
        let lookup_output = LookupQuery::<XLEN>::to_lookup_output(cycle);

        // 5. 提取控制流标志
        // Jump: 无条件跳转标志 (JAL, JALR)
        let jump_flag = flags_view[CircuitFlags::Jump];
        // Branch: 条件分支标志 (BEQ, BNE 等)
        let branch_flag = instruction_flags[InstructionFlags::Branch];

        // 6. 计算 "下一条不是空指令" (Not Next Noop)
        // 这是为了处理 Shift Sum-check (移位校验) 的边界情况。
        // 我们需要知道 t+1 时刻是否是有效的，以决定是否检查 t 到 t+1 的状态转换。
        let not_next_noop = {
            if t + 1 < len {
                // 如果还有下一条指令，检查它是否为 Noop，并取反
                !trace[t + 1].instruction().instruction_flags()[InstructionFlags::IsNoop]
            } else {
                // 边界条件：如果是最后一步，强制视为 false。
                // 这防止了 EqPlusOne (t 与 t+1 比较) 在溢出 trace 长度时出错。
                false
            }
        };

        // 7. 提取目标寄存器非零检查
        // 用于防止写入 x0 寄存器 (RISC-V 中 x0 恒为 0)
        let is_rd_not_zero = instruction_flags[InstructionFlags::IsRdNotZero];

        // 8. 提取写入使能标志
        // 决定 lookup_output 是否应该被写回 RD 寄存器
        let write_lookup_output_to_rd_flag = flags_view[CircuitFlags::WriteLookupOutputToRD];

        // 9. 构造结构体返回
        Self {
            instruction_left_input: left_input,
            instruction_right_input: right_input,
            write_lookup_output_to_rd_flag,
            should_branch_lookup_output: lookup_output, // 分支判断依据或结果
            should_branch_flag: branch_flag,
            jump_flag,
            not_next_noop,
            is_rd_not_zero,
        }
    }
}


/// State extracted from a cycle for use in shift sumcheck
pub struct ShiftSumcheckCycleState {
    pub unexpanded_pc: u64,
    pub pc: u64,
    pub is_virtual: bool,
    pub is_first_in_sequence: bool,
    pub is_noop: bool,
}

impl ShiftSumcheckCycleState {
    pub fn new(cycle: &Cycle, bytecode_preprocessing: &BytecodePreprocessing) -> Self {
        let instruction = cycle.instruction();
        let circuit_flags = instruction.circuit_flags();
        Self {
            unexpanded_pc: instruction.normalize().address as u64,
            pc: bytecode_preprocessing.get_pc(cycle) as u64,
            is_virtual: circuit_flags[CircuitFlags::VirtualInstruction],
            is_first_in_sequence: circuit_flags[CircuitFlags::IsFirstInSequence],
            is_noop: instruction.instruction_flags()[InstructionFlags::IsNoop],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    impl JoltR1CSInputs {
        /// Alternative const implementation that searches through ALL_R1CS_INPUTS array.
        /// This is used for testing to ensure the simple pattern matching to_index()
        /// returns the same results as searching through the array.
        const fn find_index_via_array_search(&self) -> usize {
            let mut i = 0;
            while i < ALL_R1CS_INPUTS.len() {
                if self.const_eq(&ALL_R1CS_INPUTS[i]) {
                    return i;
                }
                i += 1;
            }
            panic!("Invalid variant")
        }

        /// Const-compatible equality check for JoltR1CSInputs
        const fn const_eq(&self, other: &JoltR1CSInputs) -> bool {
            match (self, other) {
                (JoltR1CSInputs::PC, JoltR1CSInputs::PC) => true,
                (JoltR1CSInputs::UnexpandedPC, JoltR1CSInputs::UnexpandedPC) => true,
                (JoltR1CSInputs::Imm, JoltR1CSInputs::Imm) => true,
                (JoltR1CSInputs::RamAddress, JoltR1CSInputs::RamAddress) => true,
                (JoltR1CSInputs::Rs1Value, JoltR1CSInputs::Rs1Value) => true,
                (JoltR1CSInputs::Rs2Value, JoltR1CSInputs::Rs2Value) => true,
                (JoltR1CSInputs::RdWriteValue, JoltR1CSInputs::RdWriteValue) => true,
                (JoltR1CSInputs::RamReadValue, JoltR1CSInputs::RamReadValue) => true,
                (JoltR1CSInputs::RamWriteValue, JoltR1CSInputs::RamWriteValue) => true,
                (JoltR1CSInputs::LeftInstructionInput, JoltR1CSInputs::LeftInstructionInput) => {
                    true
                }
                (JoltR1CSInputs::RightInstructionInput, JoltR1CSInputs::RightInstructionInput) => {
                    true
                }
                (JoltR1CSInputs::LeftLookupOperand, JoltR1CSInputs::LeftLookupOperand) => true,
                (JoltR1CSInputs::RightLookupOperand, JoltR1CSInputs::RightLookupOperand) => true,
                (JoltR1CSInputs::Product, JoltR1CSInputs::Product) => true,
                (JoltR1CSInputs::WriteLookupOutputToRD, JoltR1CSInputs::WriteLookupOutputToRD) => {
                    true
                }
                (JoltR1CSInputs::WritePCtoRD, JoltR1CSInputs::WritePCtoRD) => true,
                (JoltR1CSInputs::ShouldBranch, JoltR1CSInputs::ShouldBranch) => true,
                (JoltR1CSInputs::NextUnexpandedPC, JoltR1CSInputs::NextUnexpandedPC) => true,
                (JoltR1CSInputs::NextPC, JoltR1CSInputs::NextPC) => true,
                (JoltR1CSInputs::NextIsVirtual, JoltR1CSInputs::NextIsVirtual) => true,
                (JoltR1CSInputs::NextIsFirstInSequence, JoltR1CSInputs::NextIsFirstInSequence) => {
                    true
                }
                (JoltR1CSInputs::LookupOutput, JoltR1CSInputs::LookupOutput) => true,
                (JoltR1CSInputs::ShouldJump, JoltR1CSInputs::ShouldJump) => true,
                (JoltR1CSInputs::OpFlags(flag1), JoltR1CSInputs::OpFlags(flag2)) => {
                    self.const_eq_circuit_flags(*flag1, *flag2)
                }
                _ => false,
            }
        }

        /// Const-compatible equality check for CircuitFlags
        const fn const_eq_circuit_flags(&self, flag1: CircuitFlags, flag2: CircuitFlags) -> bool {
            matches!(
                (flag1, flag2),
                (CircuitFlags::AddOperands, CircuitFlags::AddOperands)
                    | (
                        CircuitFlags::SubtractOperands,
                        CircuitFlags::SubtractOperands
                    )
                    | (
                        CircuitFlags::MultiplyOperands,
                        CircuitFlags::MultiplyOperands
                    )
                    | (CircuitFlags::Load, CircuitFlags::Load)
                    | (CircuitFlags::Store, CircuitFlags::Store)
                    | (CircuitFlags::Jump, CircuitFlags::Jump)
                    | (
                        CircuitFlags::WriteLookupOutputToRD,
                        CircuitFlags::WriteLookupOutputToRD
                    )
                    | (
                        CircuitFlags::VirtualInstruction,
                        CircuitFlags::VirtualInstruction
                    )
                    | (CircuitFlags::Assert, CircuitFlags::Assert)
                    | (
                        CircuitFlags::DoNotUpdateUnexpandedPC,
                        CircuitFlags::DoNotUpdateUnexpandedPC
                    )
                    | (CircuitFlags::Advice, CircuitFlags::Advice)
                    | (CircuitFlags::IsCompressed, CircuitFlags::IsCompressed)
                    | (
                        CircuitFlags::IsFirstInSequence,
                        CircuitFlags::IsFirstInSequence
                    )
            )
        }
    }

    #[test]
    fn to_index_consistency() {
        // Ensure to_index() and find_index_via_array_search() return the same values.
        // This validates that the simple pattern matching in to_index() correctly
        // aligns with the ordering in ALL_R1CS_INPUTS.
        for var in ALL_R1CS_INPUTS {
            assert_eq!(
                var.to_index(),
                var.find_index_via_array_search(),
                "Index mismatch for variant {:?}: pattern_match={}, array_search={}",
                var,
                var.to_index(),
                var.find_index_via_array_search()
            );
        }
    }
}
