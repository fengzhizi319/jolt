use common::constants::RAM_START_ADDRESS;
use common::jolt_device::{JoltDevice, MemoryConfig};
use std::path::PathBuf;
use tracer::emulator::memory::Memory;
use tracer::instruction::{Cycle, Instruction};
use tracer::utils::virtual_registers::VirtualRegisterAllocator;
use tracer::LazyTraceIterator;

/// Configuration for program runtime
/// 运行时配置结构体。
/// 用于保存 Guest 程序运行时的输入输出限制。
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    pub max_input_size: u64,  // 最大输入字节数
    pub max_output_size: u64, // 最大输出字节数
}

/// Guest program that handles decoding and tracing
/// Guest 程序封装结构体。
/// 不同于 host::Program (负责编译)，这里的 Program 这里主要处理已编译好的 ELF 二进制数据。
/// 它负责协调解码、内存配置和执行追踪。
pub struct Program {
    pub elf_contents: Vec<u8>,      // ELF 文件的原始字节数据
    pub memory_config: MemoryConfig, // 内存布局和大小配置
    pub elf: Option<PathBuf>,       // ELF 文件的路径（可选，用于调试或日志）
}

impl Program {
    /// 创建一个新的 Guest 程序实例。
    /// 这里的入参是已经读取到内存中的 ELF 字节数据。
    pub fn new(elf_contents: &[u8], memory_config: &MemoryConfig) -> Self {
        Self {
            elf_contents: elf_contents.to_vec(),
            memory_config: *memory_config,
            elf: None,
        }
    }
    /// Decode the ELF file into instructions and memory initialization
    /// 解码 ELF 数据。
    /// 将二进制指令转换为 Jolt 内部的指令结构体列表，并返回内存初始化数据。
    /// 返回值: (指令列表, 内存初始化数据[(地址, 字节)], 程序段大小)
    pub fn decode(&self) -> (Vec<Instruction>, Vec<(u64, u8)>, u64) {
        decode(&self.elf_contents)
    }

    /// Trace the program execution with given inputs
    /// 执行程序并生成追踪（Trace）。
    /// 这是生成证明（Prove）前的关键步骤，它模拟程序的每一步执行。
    ///
    /// 参数:
    /// - inputs: 程序的公共输入 (stdin)
    /// - untrusted_advice: 非受信任的建议输入 (auxiliary input)
    /// - trusted_advice: 受信任的建议输入 (setup/verifier input)
    ///
    /// 返回值:
    /// - LazyTraceIterator: 惰性迭代器，逐步生成执行踪迹
    /// - Vec<Cycle>: 周期统计信息
    /// - Memory: 执行结束后的内存状态
    /// - JoltDevice: IO 设备状态
    pub fn trace(
        &self,
        inputs: &[u8],
        untrusted_advice: &[u8],
        trusted_advice: &[u8],
    ) -> (LazyTraceIterator, Vec<Cycle>, Memory, JoltDevice) {
        trace(
            &self.elf_contents,
            self.elf.as_ref(),
            inputs,
            untrusted_advice,
            trusted_advice,
            &self.memory_config,
        )
    }

    /// 与 `trace` 类似，但将详细的追踪日志输出到指定文件。
    /// 通常用于离线调试或持久化长运行的执行记录。
    pub fn trace_to_file(
        &self,
        inputs: &[u8],
        untrusted_advice: &[u8],
        trusted_advice: &[u8],
        trace_file: &PathBuf,
    ) -> (Memory, JoltDevice) {
        trace_to_file(
            &self.elf_contents,
            self.elf.as_ref(),
            inputs,
            untrusted_advice,
            trusted_advice,
            &self.memory_config,
            trace_file,
        )
    }
}

/// 独立的解码函数。
/// 主要逻辑：
/// 1. 调用底层 tracer::decode 解析 ELF。
/// 2. 使用虚拟寄存器分配器处理伪指令。
/// 3. 将复杂的宏指令/虚拟序列展开为具体的物理指令流。
pub fn decode(elf: &[u8]) -> (Vec<Instruction>, Vec<(u64, u8)>, u64) {
    // raw_bytes: 内存初始化数据
    // program_end: 程序代码段结束地址
    // xlen: 架构位宽 (32 或 64)
    let (mut instructions, raw_bytes, program_end, xlen) = tracer::decode(elf);

    // 计算程序实际占用的内存大小
    let program_size = program_end - RAM_START_ADDRESS;
    // 创建虚拟寄存器分配器，用于处理需要临时寄存器的指令展开
    let allocator = VirtualRegisterAllocator::default();

    // Expand virtual sequences
    // 展开虚拟指令序列 (Virtual Sequences)
    // 某些高级操作无法直接映射到单条 CPU 指令，需要展开为多条基础指令。
    instructions = instructions
        .into_iter()
        .flat_map(|instr| instr.inline_sequence(&allocator, xlen))
        .collect();

    (instructions, raw_bytes, program_size)
}

/// 独立的追踪函数封装。
/// 将高层调用转发到底层的 tracer::trace 实现。
pub fn trace(
    elf_contents: &[u8],
    elf_path: Option<&PathBuf>,
    inputs: &[u8],
    untrusted_advice: &[u8],
    trusted_advice: &[u8],
    memory_config: &MemoryConfig,
) -> (LazyTraceIterator, Vec<Cycle>, Memory, JoltDevice) {
    let (lazy_trace, trace, memory, io_device) = tracer::trace(
        elf_contents,
        elf_path,
        inputs,
        untrusted_advice,
        trusted_advice,
        memory_config,
    );
    (lazy_trace, trace, memory, io_device)
}

/// 独立的文件追踪函数封装。
pub fn trace_to_file(
    elf_contents: &[u8],
    elf_path: Option<&PathBuf>,
    inputs: &[u8],
    untrusted_advice: &[u8],
    trusted_advice: &[u8],
    memory_config: &MemoryConfig,
    trace_file: &PathBuf,
) -> (Memory, JoltDevice) {
    let (memory, io_device) = tracer::trace_to_file(
        elf_contents,
        elf_path,
        inputs,
        untrusted_advice,
        trusted_advice,
        memory_config,
        trace_file,
    );
    (memory, io_device)
}