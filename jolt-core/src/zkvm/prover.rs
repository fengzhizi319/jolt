use crate::{subprotocols::streaming_schedule::LinearOnlySchedule, zkvm::config::OneHotConfig};
use std::{
    collections::HashMap,
    fs::File,
    io::{Read, Write},
    path::Path,
    sync::Arc,
    time::Instant,
};

use crate::poly::commitment::dory::DoryContext;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

use crate::zkvm::config::ReadWriteConfig;
use crate::zkvm::verifier::JoltSharedPreprocessing;
use crate::zkvm::Serializable;

#[cfg(not(target_arch = "wasm32"))]
use crate::utils::profiling::print_current_memory_usage;
#[cfg(feature = "allocative")]
use crate::utils::profiling::{print_data_structure_heap_usage, write_flamegraph_svg};
use crate::{
    field::JoltField,
    guest,
    poly::{
        commitment::{commitment_scheme::StreamingCommitmentScheme, dory::DoryGlobals},
        multilinear_polynomial::MultilinearPolynomial,
        opening_proof::{
            compute_advice_lagrange_factor, DoryOpeningState, OpeningAccumulator,
            ProverOpeningAccumulator, SumcheckId,
        },
        rlc_polynomial::{RLCStreamingData, TraceSource},
    },
    pprof_scope,
    subprotocols::{
        booleanity::{BooleanitySumcheckParams, BooleanitySumcheckProver},
        sumcheck::{BatchedSumcheck, SumcheckInstanceProof},
        sumcheck_prover::SumcheckInstanceProver,
        univariate_skip::{prove_uniskip_round, UniSkipFirstRoundProof},
    },
    transcripts::Transcript,
    utils::{math::Math, thread::drop_in_background_thread},
    zkvm::{
        bytecode::read_raf_checking::BytecodeReadRafSumcheckParams,
        claim_reductions::{
            AdviceClaimReductionPhase1Params, AdviceClaimReductionPhase1Prover,
            AdviceClaimReductionPhase2Params, AdviceClaimReductionPhase2Prover, AdviceKind,
            HammingWeightClaimReductionParams, HammingWeightClaimReductionProver,
            IncClaimReductionSumcheckParams, IncClaimReductionSumcheckProver,
            InstructionLookupsClaimReductionSumcheckParams,
            InstructionLookupsClaimReductionSumcheckProver, RaReductionParams,
            RamRaClaimReductionSumcheckProver, RegistersClaimReductionSumcheckParams,
            RegistersClaimReductionSumcheckProver,
        },
        config::OneHotParams,
        instruction_lookups::{
            ra_virtual::InstructionRaSumcheckParams,
            read_raf_checking::InstructionReadRafSumcheckParams,
        },
        ram::{
            hamming_booleanity::HammingBooleanitySumcheckParams,
            output_check::OutputSumcheckParams,
            populate_memory_states,
            ra_virtual::RamRaVirtualParams,
            raf_evaluation::RafEvaluationSumcheckParams,
            read_write_checking::RamReadWriteCheckingParams,
            val_evaluation::{
                ValEvaluationSumcheckParams,
                ValEvaluationSumcheckProver as RamValEvaluationSumcheckProver,
            },
            val_final::{ValFinalSumcheckParams, ValFinalSumcheckProver},
        },
        registers::{
            read_write_checking::RegistersReadWriteCheckingParams,
            val_evaluation::RegistersValEvaluationSumcheckParams,
        },
        spartan::{
            instruction_input::InstructionInputParams,
            outer::{OuterUniSkipParams, OuterUniSkipProver},
            product::{
                ProductVirtualRemainderParams, ProductVirtualUniSkipParams,
                ProductVirtualUniSkipProver,
            },
            shift::ShiftSumcheckParams,
        },
        witness::all_committed_polynomials,
    },
};
use crate::{
    poly::commitment::commitment_scheme::CommitmentScheme,
    zkvm::{
        bytecode::read_raf_checking::BytecodeReadRafSumcheckProver,
        fiat_shamir_preamble,
        instruction_lookups::{
            ra_virtual::InstructionRaSumcheckProver as LookupsRaSumcheckProver,
            read_raf_checking::InstructionReadRafSumcheckProver,
        },
        proof_serialization::{Claims, JoltProof},
        r1cs::key::UniformSpartanKey,
        ram::{
            gen_ram_memory_states, hamming_booleanity::HammingBooleanitySumcheckProver,
            output_check::OutputSumcheckProver, prover_accumulate_advice,
            ra_virtual::RamRaVirtualSumcheckProver,
            raf_evaluation::RafEvaluationSumcheckProver as RamRafEvaluationSumcheckProver,
            read_write_checking::RamReadWriteCheckingProver,
        },
        registers::{
            read_write_checking::RegistersReadWriteCheckingProver,
            val_evaluation::ValEvaluationSumcheckProver as RegistersValEvaluationSumcheckProver,
        },
        spartan::{
            instruction_input::InstructionInputSumcheckProver,
            outer::{OuterRemainingStreamingSumcheck, OuterSharedState},
            product::ProductVirtualRemainderProver,
            shift::ShiftSumcheckProver,
        },
        witness::CommittedPolynomial,
        ProverDebugInfo,
    },
};

#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use common::jolt_device::MemoryConfig;
use itertools::{zip_eq, Itertools};
use rayon::prelude::*;
use tracing::info;
use tracer::{
    emulator::memory::Memory, instruction::Cycle, ChunksIterator, JoltDevice, LazyTraceIterator,
};

/// Jolt CPU prover for RV64IMAC.
/// Jolt CPU 证明器的主结构体，针对 RV64IMAC 指令集。
///
/// 该结构体由证明者 (Prover) 实例化，负责协调整个零知识证明的生成过程。
/// 它维护了生成证明所需的所有状态，包括执行轨迹 (Trace)、多项式承诺参数、
/// Fiat-Shamir Transcript 状态以及各个阶段产生的中间数据。
///
/// 泛型参数:
/// - `'a`: 预处理数据的生命周期。
/// - `F`: 基础域 (Field)，证明系统运行所在的数学域。
/// - `PCS`: 多项式承诺方案 (Polynomial Commitment Scheme)，例如 Dory，必须支持流式处理。
/// - `ProofTranscript`: 用于 Fiat-Shamir 变换的 Transcript 实现。
pub struct JoltCpuProver<
    'a,
    F: JoltField,
    PCS: StreamingCommitmentScheme<Field = F>,
    ProofTranscript: Transcript,
> {
    /// 证明者的预处理数据。
    ///
    /// 包含多项式承诺方案的公共参数 (Generators/SRS) 以及程序共享的静态信息 (如字节码摘要)。
    /// 这些数据在证明生成之前创建，并且可以被多个证明过程复用。
    pub preprocessing: &'a JoltProverPreprocessing<F, PCS>,

    /// Jolt 设备 IO 状态。
    ///
    /// 包含程序的公共输入 (Inputs)、公共输出 (Outputs) 以及内存布局 (Memory Layout) 配置。
    /// 也用于跟踪 IO 边界，供 Verifier 验证公共输入输出的一致性。
    pub program_io: JoltDevice,

    /// 惰性 Trace 迭代器。
    ///
    /// 允许按需生成或流式访问执行轨迹数据，而不需要一次性将所有复杂的 Trace 对象加载到内存中。
    /// 这对于优化内存使用至关重要，特别是在处理长 Trace 时。
    pub lazy_trace: LazyTraceIterator,

    /// 内存中的完整执行轨迹。
    ///
    /// `Cycle` 结构体包含了单个时钟周期内发生的所有状态变化（指令、操作数、内存访问等）。
    /// 使用 `Arc` 包装以便在多个线程或并行子任务间共享而不进行深拷贝。
    pub trace: Arc<Vec<Cycle>>,

    /// 辅助输入 (Advice) 管理器。
    ///
    /// 负责持有和管理：
    /// 1. **Untrusted Advice**: 私有的 Witness 数据（Verifier 不可见）。
    /// 2. **Trusted Advice**: 预处理阶段已知的辅助数据。
    /// 以及它们对应的多项式形式和承诺值。
    pub advice: JoltAdvice<F, PCS>,

    /// 用于两阶段 Advice 归约 (Phase-Bridge) 的随机挑战值 (Gamma) - Trusted Advice。
    ///
    /// Jolt 对 Advice 的检查分为两个阶段：
    /// 1. **Stage 6 (Phase 1)**: 消除时间维度 (Cycle)。在此阶段生成 Gamma。
    /// 2. **Stage 7 (Phase 2)**: 消除空间维度 (Address)。
    /// 这个随机值 $\gamma$ 存储在这里，以确保在两个独立的 Sumcheck 阶段之间保持一致性。
    advice_reduction_gamma_trusted: Option<F>,

    /// 用于两阶段 Advice 归约的随机挑战值 (Gamma) - Untrusted Advice。
    ///
    /// 作用机制同上，但针对不可信的私有 Witness 数据。
    advice_reduction_gamma_untrusted: Option<F>,

    /// 原始执行轨迹的长度 (Unpadded)。
    ///
    /// 程序实际执行的指令周期数。
    pub unpadded_trace_len: usize,

    /// 填充后的执行轨迹长度 (Padded)。
    ///
    /// 通常向上取整到最近的 2 的幂次。这是为了满足多线性多项式 (Multilinear Polynomials)
    /// 结构的定义，以及适配多项式承诺方案 (如 Dory) 对矩阵维度的要求。
    pub padded_trace_len: usize,

    /// Fiat-Shamir Transcript。
    ///
    /// 用于实现非交互式零知识证明。随着证明过程的推进，Prover 会将生成的承诺和数据
    /// “吸收”到 Transcript 中，并从中“挤出”伪随机挑战 (Challenge)。
    pub transcript: ProofTranscript,

    /// 证明打开累加器 (Opening Accumulator)。
    ///
    /// 在整个 Sumcheck 协议的 8 个阶段中，会产生大量的多项式点值评估声明 (Claims)。
    /// 累加器将这些分散的声明收集起来，最后通过一次批量的 Dory Opening Proof 来统一证明，
    /// 极大地摊销了验证成本。
    pub opening_accumulator: ProverOpeningAccumulator<F>,

    /// Spartan 证明系统的密钥/参数。
    ///
    /// Jolt 使用 Spartan 的变体来证明 R1CS 约束的可满足性。
    /// 这里存储了用于处理 Uniform R1CS 结构的参数（如矩阵维度信息）。
    pub spartan_key: UniformSpartanKey<F>,

    /// 初始内存状态快照。
    ///
    /// 记录了程序开始执行前，所有非零内存地址上的值。
    /// 用于 Stage 4 的内存一致性检查（验证 Initial -> Final 状态转换的正确性）。
    pub initial_ram_state: Vec<u64>,

    /// 最终内存状态快照。
    ///
    /// 记录了程序终止时，所有非零内存地址上的值。
    pub final_ram_state: Vec<u64>,

    /// One-Hot 编码参数。
    ///
    /// 用于配置指令解码、标志位检查等涉及 One-Hot 向量的组件，确保位向量的有效性。
    pub one_hot_params: OneHotParams,

    /// 读写一致性配置 (Read-Write Config)。
    ///
    /// 定义了 RAM 和寄存器文件在进行读写一致性检查时的具体策略和参数。
    pub rw_config: ReadWriteConfig,
}
impl<'a, F: JoltField, PCS: StreamingCommitmentScheme<Field = F>, ProofTranscript: Transcript>
JoltCpuProver<'a, F, PCS, ProofTranscript>
{
    pub fn gen_from_elf(
        preprocessing: &'a JoltProverPreprocessing<F, PCS>,
        elf_contents: &[u8],
        inputs: &[u8],
        untrusted_advice: &[u8],
        trusted_advice: &[u8],
        trusted_advice_commitment: Option<PCS::Commitment>,
        trusted_advice_hint: Option<PCS::OpeningProofHint>,
    ) -> Self {
        // 1. 配置内存参数：
        // 从预处理数据（preprocessing.shared）中提取内存布局信息（如堆栈大小、输入输出区域大小等）。
        // 这些配置必须与电路定义保持一致，以确保追踪器（Tracer）产生的内存访问地址在电路允许的范围内。
        let memory_config = MemoryConfig {
            max_untrusted_advice_size: preprocessing.shared.memory_layout.max_untrusted_advice_size,
            max_trusted_advice_size: preprocessing.shared.memory_layout.max_trusted_advice_size,
            max_input_size: preprocessing.shared.memory_layout.max_input_size,
            max_output_size: preprocessing.shared.memory_layout.max_output_size,
            stack_size: preprocessing.shared.memory_layout.stack_size,
            memory_size: preprocessing.shared.memory_layout.memory_size,
            program_size: Some(preprocessing.shared.memory_layout.program_size),
        };

        // 2. 执行程序生成 Trace（执行轨迹）：
        // 使用 guest 模拟器运行 ELF 二进制文件。这一步是“真实”的执行过程。
        // 输入包括标准输入、不可信 Advice（witness）和可信 Advice。
        // 返回：
        // - lazy_trace: 惰性加载的 Trace 迭代器（用于后续流式处理，节省内存）。
        // - trace: 完整的 Cycle 向量（包含每个时钟周期的详细状态，如寄存器值）。
        // - final_memory_state: 执行结束后的内存状态（用于构建 RAM 的初始/最终一致性检查）。
        // - program_io: 捕获的 IO 设备状态（记录了实际读取了哪些输入，写入了哪些输出）。
        let (lazy_trace, trace, final_memory_state, program_io) = {
            let _pprof_trace = pprof_scope!("trace");
            guest::program::trace(
                elf_contents,
                None, // 这里的 None 表示不使用缓存的 Image，而是从 ELF 重新加载
                inputs,
                untrusted_advice,
                trusted_advice,
                &memory_config,
            )
        };

        // 3. 统计指令周期信息（用于日志和调试）：
        // 区分原始 RISC-V 指令和 Jolt 虚拟指令。
        // Jolt 为了支持复杂的 RISC-V 指令，可能会将其拆分为多个“虚拟微指令序列”（Virtual Sequence）。
        // 这里通过检查 `virtual_sequence_remaining` 字段来统计实际对应的高层 RISC-V 指令数量。
        let num_riscv_cycles: usize = trace
            .par_iter()
            .map(|cycle| {
                // 如果指令不是虚拟序列的一部分（None），或者是序列的第一条指令（Some(0)），则计数为 1。
                // 这样可以过滤掉虚拟序列中的后续微指令，只统计逻辑指令数。
                if let Some(virtual_sequence_remaining) =
                    cycle.instruction().normalize().virtual_sequence_remaining
                {
                    if virtual_sequence_remaining > 0 {
                        return 0;
                    }
                }
                1
            })
            .sum();
        tracing::info!(
            "{num_riscv_cycles} raw RISC-V instructions + {} virtual instructions = {} total cycles",
            trace.len() - num_riscv_cycles,
            trace.len(),
        );
        println!("{num_riscv_cycles} raw RISC-V instructions + {} virtual instructions = {} total cycles",
                 trace.len() - num_riscv_cycles,
                 trace.len());

        // 4. 从 Trace 构建 Prover 实例：
        // 将生成的 Trace 数据和相关上下文传递给 gen_from_trace 方法，
        // 该方法会负责计算填充长度（Padding）、初始化多项式状态等后续步骤。
        Self::gen_from_trace(
            preprocessing,
            lazy_trace,
            trace,
            program_io,
            trusted_advice_commitment,
            trusted_advice_hint,
            final_memory_state,
        )
    }

    /// Adjusts the padded trace length to ensure the main Dory matrix is large enough
    /// to embed advice polynomials as the top-left block.
    ///
    /// Returns the adjusted padded_trace_len that satisfies:
    /// - `sigma_main >= max_sigma_a`
    /// - `nu_main >= max_nu_a`
    ///
    /// Panics if `max_padded_trace_length` is too small for the configured advice sizes.
    fn adjust_trace_length_for_advice(
        mut padded_trace_len: usize,
        max_padded_trace_length: usize,
        max_trusted_advice_size: u64,
        max_untrusted_advice_size: u64,
        has_trusted_advice: bool,
        has_untrusted_advice: bool,
    ) -> usize {
        // Canonical advice shape policy (balanced):
        // - advice_vars = log2(advice_len)
        // - sigma_a = ceil(advice_vars/2)
        // - nu_a    = advice_vars - sigma_a
        let mut max_sigma_a = 0usize;
        let mut max_nu_a = 0usize;

        if has_trusted_advice {
            let (sigma_a, nu_a) =
                DoryGlobals::advice_sigma_nu_from_max_bytes(max_trusted_advice_size as usize);
            max_sigma_a = max_sigma_a.max(sigma_a);
            max_nu_a = max_nu_a.max(nu_a);
        }
        if has_untrusted_advice {
            let (sigma_a, nu_a) =
                DoryGlobals::advice_sigma_nu_from_max_bytes(max_untrusted_advice_size as usize);
            max_sigma_a = max_sigma_a.max(sigma_a);
            max_nu_a = max_nu_a.max(nu_a);
        }

        if max_sigma_a == 0 && max_nu_a == 0 {
            return padded_trace_len;
        }

        // Require main matrix dimensions to be large enough to embed advice as the top-left
        // block: sigma_main >= sigma_a and nu_main >= nu_a.
        //
        // This loop doubles padded_trace_len until the main Dory matrix is large enough.
        // Each doubling increases log_t by 1, which increases total_vars by 1 (since
        // log_k_chunk stays constant for a given log_t range), increasing both sigma_main
        // and nu_main by roughly 0.5 each iteration.
        while {
            let log_t = padded_trace_len.log_2();
            let log_k_chunk = OneHotConfig::new(log_t).log_k_chunk as usize;
            let (sigma_main, nu_main) = DoryGlobals::main_sigma_nu(log_k_chunk, log_t);
            sigma_main < max_sigma_a || nu_main < max_nu_a
        } {
            if padded_trace_len >= max_padded_trace_length {
                // This is a configuration error: the preprocessing was set up with
                // max_padded_trace_length too small for the configured advice sizes.
                // Cannot recover at runtime - user must fix their configuration.
                let log_t = padded_trace_len.log_2();
                let log_k_chunk = OneHotConfig::new(log_t).log_k_chunk as usize;
                let total_vars = log_k_chunk + log_t;
                let (sigma_main, nu_main) = DoryGlobals::main_sigma_nu(log_k_chunk, log_t);
                panic!(
                    "Configuration error: trace too small to embed advice into Dory batch opening.\n\
                    Current: (sigma_main={sigma_main}, nu_main={nu_main}) from total_vars={total_vars} (log_t={log_t}, log_k_chunk={log_k_chunk})\n\
                    Required: (sigma_a={max_sigma_a}, nu_a={max_nu_a}) for advice embedding\n\
                    Solutions:\n\
                    1. Increase max_trace_length in preprocessing (currently {max_padded_trace_length})\n\
                    2. Reduce max_trusted_advice_size or max_untrusted_advice_size\n\
                    3. Run a program with more cycles"
                );
            }
            padded_trace_len = (padded_trace_len * 2).min(max_padded_trace_length);
        }

        padded_trace_len
    }

    ///gen_from_trace 方法是 JoltCpuProver 的核心构造函数。它接收由模拟器生成的原始执行记录（Trace），
    ///对其进行必要的填充（Padding）和配置，并初始化证明生成所需的所有状态（如内存快照、Spartan 密钥、Dory 参数等）。
    pub fn gen_from_trace(
        preprocessing: &'a JoltProverPreprocessing<F, PCS>,
        lazy_trace: LazyTraceIterator,
        mut trace: Vec<Cycle>,       // 原始执行轨迹（未填充）
        mut program_io: JoltDevice,  // 包含输入、输出和 Advice 数据
        trusted_advice_commitment: Option<PCS::Commitment>, // 如果有可信 Advice，这里是其预先计算好的承诺
        trusted_advice_hint: Option<PCS::OpeningProofHint>, // 可信 Advice 的打开证明提示
        final_memory_state: Memory, // 程序执行结束时的内存快照
    ) -> Self {
        // [测试专用] Dory 全局变量是进程级单例。在测试中，通常会在同一个进程中运行多个不同 Trace 长度的
        // 端到端证明。为了避免状态污染，每次构建新 Prover 前重置 Dory 全局状态。
        #[cfg(test)]
        crate::poly::commitment::dory::DoryGlobals::reset();

        // 1. 规范化输出数据：
        // 移除 outputs 向量末尾的所有零字节（truncate trailing zeros）。
        // 这样可以确保输出数据紧凑，去除缓冲区中未使用的部分。
        program_io.outputs.truncate(
            program_io
                .outputs
                .iter()
                .rposition(|&b| b != 0)
                .map_or(0, |pos| pos + 1),
        );

        // 2. 设置 Trace 长度和填充（Padding）：
        let unpadded_trace_len = trace.len();
        // 初始填充逻辑：
        // - 至少填充到 256。这是为了满足 Dory 承诺方案中 T >= k^{1/D} 的数学约束。
        // - 否则，取下一个 2 的幂次（Next Power of Two），便于构建二叉树结构的证明。
        let padded_trace_len = if unpadded_trace_len < 256 {
            256
        } else {
            (trace.len() + 1).next_power_of_two()
        };

        // 检查是否存在 Advice（辅助输入）数据。
        let has_trusted_advice = !program_io.trusted_advice.is_empty();
        let has_untrusted_advice = !program_io.untrusted_advice.is_empty();

        // 调整 Trace 长度以适配 Advice：
        // 如果 Advice 数据很大，Dory 承诺的主矩阵（维度由 Trace 长度决定）可能不够大，无法将 Advice 嵌入其中。
        // 此函数会根据 Advice 的大小，必要时增加 Trace 的长度（例如翻倍），直到矩阵足以容纳 Advice。
        let padded_trace_len = Self::adjust_trace_length_for_advice(
            padded_trace_len,
            preprocessing.shared.max_padded_trace_length,
            preprocessing.shared.memory_layout.max_trusted_advice_size,
            preprocessing.shared.memory_layout.max_untrusted_advice_size,
            has_trusted_advice,
            has_untrusted_advice,
        );

        // 将 Trace 实际 resize 到计算出的填充长度，填充部分使用 NoOp（空指令）。
        trace.resize(padded_trace_len, Cycle::NoOp);

        // 3. 计算 RAM 大小 K (用于 DoryGlobals 初始化和内存检查)：
        // 遍历整个 Trace，找出所有被访问过的内存地址，并结合字节码区域，
        // 计算出一个足以覆盖所有访问地址的最小 2 的幂次大小（ram_K）。
        let ram_K = trace
            .par_iter() // 并行迭代加速处理
            .filter_map(|cycle| {
                // 将物理/虚拟地址重映射到 Jolt 的内部现行 RAM 地址空间
                crate::zkvm::ram::remap_address(
                    cycle.ram_access().address() as u64,
                    &preprocessing.shared.memory_layout,
                )
            })
            .max()
            .unwrap_or(0)
            .max(
                // 确保 RAM 大小至少能包含所有的字节码（Bytecode）
                crate::zkvm::ram::remap_address(
                    preprocessing.shared.ram.min_bytecode_address,
                    &preprocessing.shared.memory_layout,
                )
                    .unwrap_or(0)
                    + preprocessing.shared.ram.bytecode_words.len() as u64
                    + 1,
            )
            .next_power_of_two() as usize;

        // 4. 初始化证明相关的核心组件：
        // 初始化 Transcript，用于 Fiat-Shamir 变换，生成伪随机挑战。
        let transcript = ProofTranscript::new(b"Jolt");
        // 初始化打开累加器（Opening Accumulator），用于在 Stage 8 聚合所有多项式的批处理打开验证。
        let opening_accumulator = ProverOpeningAccumulator::new(trace.len().log_2());

        // 初始化 Spartan Key（基于 Trace 长度构建）。
        let spartan_key = UniformSpartanKey::new(trace.len());

        // 5. 生成 RAM 状态快照：
        // 构建 RAM 的初始状态（包含加载的程序、输入数据）和最终状态。
        // 这对于 RAM 一致性检查（Memory Consistency Check）是必需的。
        let (initial_ram_state, final_ram_state) = gen_ram_memory_states::<F>(
            ram_K,
            &preprocessing.shared.ram,
            &program_io,
            &final_memory_state,
        );

        // 6. 生成配置参数：
        let log_T = trace.len().log_2();
        let ram_log_K = ram_K.log_2();
        // 读写配置：基于时间维度（log_T）和空间维度（ram_log_K）。
        let rw_config = ReadWriteConfig::new(log_T, ram_log_K);
        // One-Hot 参数：用于将指令执行和内存访问转换为电路约束。
        let one_hot_params =
            OneHotParams::new(log_T, preprocessing.shared.bytecode.code_size, ram_K);

        // 7. 构造并返回 Prover 实例
        Self {
            preprocessing,
            program_io,
            lazy_trace,
            trace: trace.into(), // 转换为 Arc 以便共享所有权
            advice: JoltAdvice {
                untrusted_advice_polynomial: None, // 将在 prove() 过程中计算生成
                trusted_advice_commitment,
                trusted_advice_polynomial: None,
                untrusted_advice_hint: None,
                trusted_advice_hint,
            },
            advice_reduction_gamma_trusted: None,
            advice_reduction_gamma_untrusted: None,
            unpadded_trace_len,
            padded_trace_len,
            transcript,
            opening_accumulator,
            spartan_key,
            initial_ram_state,
            final_ram_state,
            one_hot_params,
            rw_config,
        }
    }

    #[allow(clippy::type_complexity)]
    #[tracing::instrument(skip_all)]
    pub fn prove(
        mut self,
    ) -> (
        JoltProof<F, PCS, ProofTranscript>,
        Option<ProverDebugInfo<F, ProofTranscript, PCS>>,
    ) {
        let _pprof_prove = pprof_scope!("prove");

        let start = Instant::now();
        // 初始化 Fiat-Shamir 预处理：将公共输入（程序IO、内存配置、Trace长度等）吸收到 Transcript 中，
        // 以确保后续生成的随机数 challenge 依赖于这些公共参数。
        fiat_shamir_preamble(
            &self.program_io,
            self.one_hot_params.ram_k,
            self.trace.len(),
            &mut self.transcript,
        );

        tracing::info!(
               "bytecode size: {}",
               self.preprocessing.shared.bytecode.code_size
           );

        // 1. 生成并提交 Witness 多项式：
        // 计算执行轨迹（Trace）相关的多项式，进行承诺（Commit），并生成打开证明的提示（Hints）。
        let (commitments, mut opening_proof_hints) = self.generate_and_commit_witness_polynomials();

        // 2. 处理 Advice（辅助输入）：
        // 生成并提交不可信 Advice 的承诺。
        let untrusted_advice_commitment = self.generate_and_commit_untrusted_advice();
        // 准备可信 Advice 的多项式（通常不需要运行时提交，因为是在预处理阶段定义的或可信的）。
        self.generate_and_commit_trusted_advice();

        // 将 Advice 相关的打开提示（Hints）加入集合，以便在 Stage 8 的批量打开中使用。
        if let Some(hint) = self.advice.trusted_advice_hint.take() {
            opening_proof_hints.insert(CommittedPolynomial::TrustedAdvice, hint);
        }
        if let Some(hint) = self.advice.untrusted_advice_hint.take() {
            opening_proof_hints.insert(CommittedPolynomial::UntrustedAdvice, hint);
        }

        // --- 执行各个阶段的 Sumcheck 证明协议 ---

        // Stage 1: 主要处理指令查找（Instruction Lookups）相关的一元跳跃（UniSkip）证明。
        let (stage1_uni_skip_first_round_proof, stage1_sumcheck_proof) = self.prove_stage1();

        // Stage 2: 处理剩余的指令查找及相关的一元跳跃证明。
        let (stage2_uni_skip_first_round_proof, stage2_sumcheck_proof) = self.prove_stage2();

        // Stage 3: Spartan 协议相关的 Sumcheck，包括 trace 的移位（Shift）、指令输入一致性、寄存器 Claim 归约。
        let stage3_sumcheck_proof = self.prove_stage3();

        // Stage 4: 内存（RAM）和寄存器的读写一致性检查（Read/Write Checking），以及值评估（Val Evaluation）。
        let stage4_sumcheck_proof = self.prove_stage4();

        // Stage 5: 寄存器值评估、RAM Read-Access (RA) Claim 归约、指令 Read-RAF 检查。
        let stage5_sumcheck_proof = self.prove_stage5();

        // Stage 6: 字节码读取检查、布尔性校验（Booleanity）、RAM RA 虚拟化检查以及 Advice Claim 归约的第一阶段。
        let stage6_sumcheck_proof = self.prove_stage6();

        // Stage 7: 汉明权重（Hamming Weight）检查以及 Advice Claim 归约的第二阶段。
        let stage7_sumcheck_proof = self.prove_stage7();

        // Stage 8: 联合打开证明 (Joint Opening Proof)。
        // 使用 Dory 协议对之前所有阶段产生的多项式承诺点进行批量验证。
        let joint_opening_proof = self.prove_stage8(opening_proof_hints);

        // 测试环境下，断言所有虚拟打开（Virtual Openings）都已被证明，防止逻辑遗漏。
        #[cfg(test)]
        assert!(
            self.opening_accumulator
                .appended_virtual_openings
                .borrow()
                .is_empty(),
            "Not all virtual openings have been proven, missing: {:?}",
            self.opening_accumulator.appended_virtual_openings.borrow()
        );

        // 测试环境下收集调试信息。
        #[cfg(test)]
        let debug_info = Some(ProverDebugInfo {
            transcript: self.transcript.clone(),
            opening_accumulator: self.opening_accumulator.clone(),
            prover_setup: self.preprocessing.generators.clone(),
        });
        #[cfg(not(test))]
        let debug_info = None;

        // 构建最终的 Jolt 证明结构体，包含所有阶段的 Sumcheck 证明、承诺和公共参数。
        let proof = JoltProof {
            opening_claims: Claims(self.opening_accumulator.openings.clone()),
            commitments,
            untrusted_advice_commitment,
            stage1_uni_skip_first_round_proof,
            stage1_sumcheck_proof,
            stage2_uni_skip_first_round_proof,
            stage2_sumcheck_proof,
            stage3_sumcheck_proof,
            stage4_sumcheck_proof,
            stage5_sumcheck_proof,
            stage6_sumcheck_proof,
            stage7_sumcheck_proof,
            joint_opening_proof,
            trace_length: self.trace.len(),
            ram_K: self.one_hot_params.ram_k,
            bytecode_K: self.one_hot_params.bytecode_k,
            rw_config: self.rw_config.clone(),
            one_hot_config: self.one_hot_params.to_config(),
        };

        let prove_duration = start.elapsed();

        tracing::info!(
               "Proved in {:.1}s ({:.1} kHz / padded {:.1} kHz)",
               prove_duration.as_secs_f64(),
               self.unpadded_trace_len as f64 / prove_duration.as_secs_f64() / 1000.0,
               self.padded_trace_len as f64 / prove_duration.as_secs_f64() / 1000.0,
           );

        (proof, debug_info)
    }


    #[tracing::instrument(skip_all, name = "generate_and_commit_witness_polynomials")]
    fn generate_and_commit_witness_polynomials(
        &mut self,
    ) -> (
        Vec<PCS::Commitment>,
        HashMap<CommittedPolynomial, PCS::OpeningProofHint>,
    ) {
        // 1. 初始化 Dory 上下文：配置全局状态，用于后续的并行 MSM 计算。
        // chunk_k 决定了矩阵的宽度（即一行有多少个元素），padded_trace_len 是矩阵的总大小。
        /*one_hot_params：
         log_k_chunk: 4,
         lookups_ra_virtual_log_k_chunk: 16,
         k_chunk: 16,
         bytecode_k: 2048,
         ram_k: 8192,
         instruction_d: 32,
         bytecode_d: 3,
         ram_d: 4,
         */
        let _guard = DoryGlobals::initialize_context(
            1 << self.one_hot_params.log_k_chunk,
            self.padded_trace_len,
            DoryContext::Main,
        );

        // 2. 准备基本参数
        let T = DoryGlobals::get_T(); // 填充后的 Trace 总长度1820->2048
        // 获取所有需要在这一步提交的多项式定义（Schema）
        let polys = all_committed_polynomials(&self.one_hot_params);//41
        let row_len = DoryGlobals::get_num_columns(); // Dory 矩阵的列数（宽度）256
        let num_rows = T / DoryGlobals::get_max_num_rows(); // Dory 矩阵的行数。16=2048/128

        tracing::debug!(
           "Generating and committing {} witness polynomials with T={}, row_len={}, num_rows={}",
           polys.len(),
           T,
           row_len,
           num_rows
       );

        // ==========================================
        // Tier 1: 流式计算行承诺 (Row Commitments)
        // ==========================================
        // 预分配内存用于存储每一行的中间承诺状态
        let mut row_commitments: Vec<Vec<PCS::ChunkState>> = vec![vec![]; num_rows];

        // 并行流式处理 Trace：
        // self.lazy_trace 是一个懒加载迭代器，不会一次性把所有数据读入内存。
        /*
        self.lazy_trace
            .clone()
            .pad_using(T, |_| Cycle::NoOp) // 如果 Trace 不足 T，用 NoOp 填充
            .iter_chunks(row_len)          // 按照 Dory 的矩阵行宽进行切分
            .zip(row_commitments.iter_mut())
            .par_bridge()                  // 开启并行处理（Rayon）
            .for_each(|(chunk, row_tier1_commitments)| {
                tracing::debug!("polys： {:?}", polys);
                // 在当前线程中，处理这一块 Trace 数据 (Chunk)
                let res: Vec<_> = polys
                    .par_iter() // 对每一种多项式并行处理
                    .map(|poly| {
                        // 核心逻辑：
                        // 1. 根据 chunk 里的 Trace 数据，计算该多项式的具体的点值 (Witness Generation)。
                        // 2. 立即对这些点值进行 MSM 计算，得到一个小的 ChunkState (Commitment)。
                        // 3. 原始的点值数据在这里就被释放了，极大节省内存。
                        poly.stream_witness_and_commit_rows::<_, PCS>(
                            &self.preprocessing.generators,
                            &self.preprocessing.shared,
                            &chunk,
                            &self.one_hot_params,
                        )
                    })
                    .collect();
                // 保存当前这一行所有多项式的中间承诺
                *row_tier1_commitments = res;
            });

         */
        // --------------------------------------------------------------------------
        // [调试修改] 拆分代码与单线程化
        // --------------------------------------------------------------------------

        // 1. 构造迭代器链，但不立即消费
        // 如果 Trace 不足 T，用 NoOp 填充，并按行宽切分,
        let trace_stream = self
            .lazy_trace
            .clone()
            .pad_using(T, |_| Cycle::NoOp)
            .iter_chunks(row_len);

        // 2. 将数据流与结果存储(row_commitments)打包，并添加索引方便调试
        let zipped_iter = trace_stream
            .zip(row_commitments.iter_mut())
            .enumerate();

        tracing::info!("=== 开始 witness 多项式生成与 Commit (单线程调试模式) ===");
        tracing::info!("Trace 长度 T: {}, Dory 行宽: {}, 总行数: {}", T, row_len, num_rows);

        // 3. 外层循环：遍历每一“行”（Trace 的一个 Chunk，每个chunk有很多个trace）
        for (_row_idx, (chunk, row_tier1_commitments)) in zipped_iter {
            //tracing::info!(">>> 正在处理第 {} 行 (Row Index)", row_idx);
            //tracing::info!("    当前 Chunk 大小: {}", chunk.len());
            // tracing::debug!("polys： {:?}", polys); // 保留原有的 log


            // 此时已移除 par_iter，改为串行处理
            let mut row_results = Vec::with_capacity(polys.len());
            // 4. 内层循环：遍历每承诺一个多项式，即把多个trace的对应列提出出来，计算commitment
            for (_poly_idx, poly) in polys.iter().enumerate() {
                // 打印多项式信息
                //tracing::info!("    -> 处理第 {} 个多项式: {:?}", poly_idx, poly);

                // 核心逻辑：计算点值并 Commit
                // 注意：这里没有 par_iter，会在当前线程阻塞执行
                let commitment = poly.stream_witness_and_commit_rows::<_, PCS>(
                    &self.preprocessing.generators,
                    &self.preprocessing.shared,
                    &chunk,
                    &self.one_hot_params,
                );

                row_results.push(commitment);
            }

            // 保存当前这一行所有多项式的中间承诺
            *row_tier1_commitments = row_results;

            //tracing::info!("<<< 第 {} 行处理完毕\n", row_idx);
        }
        tracing::info!("=== Witness Commit 结束 ===");

        // ==========================================
        // 数据转置 (Transpose)
        // ==========================================
        // 目前数据格式是：row_commitments[row_idx][poly_idx]
        // 我们需要按多项式聚合，所以转换为：tier1_per_poly[poly_idx][row_idx]
        let tier1_per_poly: Vec<Vec<PCS::ChunkState>> = (0..polys.len())
            .into_par_iter()
            .map(|poly_idx| {
                row_commitments
                    .iter()
                    .flat_map(|row| row.get(poly_idx).cloned())
                    .collect()
            })
            .collect();

        // ==========================================
        // Tier 2: 最终聚合 (Aggregation)
        // ==========================================
        // 并行地为每个多项式计算最终承诺
        let (commitments, hints): (Vec<_>, Vec<_>) = tier1_per_poly
            .into_par_iter()
            .zip(&polys)
            .map(|(tier1_commitments, poly)| {
                // 获取该多项式的大小参数（One-Hot 编码相关）
                let onehot_k = poly.get_onehot_k(&self.one_hot_params);
                // aggregate_chunks 会将所有行的中间承诺（ChunkState）合并
                // 生成最终的 Commitment 和用于打开证明的 Hint
                PCS::aggregate_chunks(&self.preprocessing.generators, onehot_k, &tier1_commitments)
            })
            .unzip();

        // 将 Hint 放入 HashMap 以便后续检索
        let hint_map = HashMap::from_iter(zip_eq(polys, hints));

        // ==========================================
        // Fiat-Shamir: 写入 Transcript
        // ==========================================
        // 将计算出的承诺吸收到随机预言机中，以冻结当前的计算状态
        for commitment in &commitments {
            self.transcript.append_serializable(commitment);
        }

        (commitments, hint_map)
    }

    /// 生成并提交不可信 Advice（辅助输入）的多项式承诺。
    ///
    /// "不可信 Advice" 指的是 Prover 在运行时提供的私有输入（例如 witness），
    /// Verifier 无法提前获知其内容，因此需要 Prover 对其进行密码学承诺。
    fn generate_and_commit_untrusted_advice(&mut self) -> Option<PCS::Commitment> {
        // 1. 检查是否存在不可信 Advice。如果为空，则无需处理，直接返回 None。
        if self.program_io.untrusted_advice.is_empty() {
            return None;
        }

        // 2. 准备数据容器。
        // 根据内存布局中定义的 `max_untrusted_advice_size`（以位为单位，所以除以 8 转为字节）
        // 分配一个全零向量 `untrusted_advice_vec`。
        // 这里的逻辑是：即便实际提供的 advice 数据较少，也会按照最大允许大小进行填充（Padding），
        // 以保证多项式结构的一致性。
        let mut untrusted_advice_vec =
            vec![0; self.program_io.memory_layout.max_untrusted_advice_size as usize / 8];

        // 3. 填充数据。
        // 将实际的 advice 数据 (`program_io.untrusted_advice`) 复制到刚才创建的向量中。
        // `populate_memory_states` 负责将字节数据按正确的端序（Little-Endian）打包成 u64 格式，
        // 以便后续转换为 Field 元素。
        populate_memory_states(
            0, // 起始偏移量
            &self.program_io.untrusted_advice,
            Some(&mut untrusted_advice_vec), // 目标缓冲区
            None,
        );

        // 4. 转换为多项式。
        // 将填充好的 u64 向量转换为一个多线性多项式（MultilinearPolynomial）。
        // 这一步在数学上将离散的数据点映射为了代数对象。
        let poly = MultilinearPolynomial::from(untrusted_advice_vec);

        // 计算多项式的长度，确保是 2 的幂次（这是 FFT/NTT 或构建承诺所需的结构）。
        let advice_len = poly.len().next_power_of_two().max(1);

        // 5. 初始化 Dory 上下文。
        // Dory 是一种批量多项式承诺方案。这里初始化一个专门针对 "UntrustedAdvice" 的上下文。
        // `1` 表示 num_vars_chunk（或者说矩阵的行数维度参数，advice 通常被视为扁平向量）。
        // `advice_len` 是总长度。
        // `_guard` 利用 RAII 机制，在作用域结束时自动清理上下文全局状念。
        let _guard = DoryGlobals::initialize_context(1, advice_len, DoryContext::UntrustedAdvice);
        // 切换当前线程到 UntrustedAdvice 上下文 (影响后续的点积/MSM 计算参数)。
        let _ctx = DoryGlobals::with_context(DoryContext::UntrustedAdvice);

        // 6. 计算承诺 (Commit)。
        // 使用 PCS (Polynomial Commitment Scheme) 对多项式进行承诺。
        // `commitment`: 这是一个短小的密码学值（椭圆曲线点），相当于数据的指纹。
        // `hint`: 生成 Opening Proof 时需要的辅助信息（通常包含中间计算结果）。
        let (commitment, hint) = PCS::commit(&poly, &self.preprocessing.generators);

        // 7. 更新 Transcript。
        // 将承诺值写入 Fiat-Shamir Transcript。
        // 这至关重要：它将 Advice 的内容“绑定”到当前的证明会话中，
        // 使得后续生成的随机 Challenge 都会依赖于这个承诺值。
        self.transcript.append_serializable(&commitment);

        // 8. 保存状态。
        // 将多项式本体和 hint 保存到 `self.advice` 结构中，
        // 供后续步骤（如 Sumcheck 或生成 Opening Proof）使用。
        self.advice.untrusted_advice_polynomial = Some(poly);
        self.advice.untrusted_advice_hint = Some(hint);

        // 返回承诺值。
        Some(commitment)
    }

    /// 生成可信 Advice（辅助输入）的多项式数据，并不直接计算承诺（Commitment）。
    ///
    /// "可信 Advice" 通常是在预处理阶段已经确定或者由 Verifier 已知的辅助输入。
    /// 与 "不可信 Advice" 不同，它的承诺值通常在 `gen_from_elf` 之前就已经存在，
    /// 或者由预处理步骤提供。这里的主要任务是将原始数据转换为多项式格式，
    /// 并将已有的承诺值吸收到 Transcript 中。
    fn generate_and_commit_trusted_advice(&mut self) {
        // 1. 检查数据是否存在：如果没有可信 Advice 数据，直接返回，不做任何操作。
        if self.program_io.trusted_advice.is_empty() {
            return;
        }

        // 2. 预分配内存缓冲区：
        // 根据内存布局中定义的 `max_trusted_advice_size`（最大允许的可信 Advice 大小，单位是位）
        // 分配一个字节向量。
        // 除以 8 是为了从位（bits）转换为字节（bytes）。初始化为全 0。
        // 即使实际数据很少，也必须填充到最大固定大小，以满足电路的固定结构要求。
        let mut trusted_advice_vec =
            vec![0; self.program_io.memory_layout.max_trusted_advice_size as usize / 8];

        // 3. 填充数据：
        // 使用 `populate_memory_states` 将实际的 `program_io.trusted_advice` 数据
        // 写入到刚才分配的缓冲区 `trusted_advice_vec` 中。
        // 该函数通常处理端序（Endianness）转换，将字节流正确打包成适合字段元素（Field Element）的格式（如 u64）。
        populate_memory_states(
            0, // 起始偏移量
            &self.program_io.trusted_advice, // 源数据
            Some(&mut trusted_advice_vec),   // 目标缓冲区
            None,
        );

        // 4. 转换为多线性多项式：
        // 将填充好的 u64 向量转换为多线性多项式（Multilinear Polynomial）。
        // 在 ZK 证明系统中，数据通常表示为多项式的系数或评估值。
        let poly = MultilinearPolynomial::from(trusted_advice_vec);

        // 5. 保存多项式：
        // 将生成的多项式保存到 Prover 的 `advice` 结构中，用于后续的 Sumcheck 协议证明生成。
        self.advice.trusted_advice_polynomial = Some(poly);

        // 6. 更新 Transcript（Fiat-Shamir）：
        // 将 **已有的** 承诺值（commitment）写入 Transcript。
        // 注意：这里并不像 untrusted advice 那样现场计算 commit，而是取出 `self.advice` 中
        // 预先存在的 `trusted_advice_commitment`（通常在 Prover 初始化时传入）。
        // 这步操作将其绑定到当前的证明会话中。
        self.transcript
            .append_serializable(self.advice.trusted_advice_commitment.as_ref().unwrap());
    }


    /// 执行第一阶段的 Sumcheck 证明（Stage 1 Proving）。
    ///
    /// # 作用
    /// Stage 1 负责证明 Spartan 协议中的 **Outer Sumcheck**（外部 Sumcheck）。
    /// Spartan 的算术化将 R1CS 约束 $Az \circ Bz = Cz$ 转换为一个多线性多项式等式。
    /// 本阶段的目标是证明：
    /// $$ \sum_{y \in \{0, 1\}^s} \tilde{eq}(\tau, y) \cdot ((\tilde{A}(y, x) \cdot \tilde{B}(y, x)) - \tilde{C}(y, x)) = 0 $$
    /// 其中 $x$ 是 Witness（包含 trace），$\tau$ 是 Verifier 提供的随机挑战点。
    ///
    /// # 核心逻辑：UniSkip (Univariate Skip)
    /// Jolt 为了优化性能，采用了 "UniSkip" 策略。
    /// 标准的 Sumcheck 需要逐个变量地进行归约。UniSkip 将某些特定的变量（通常是与指令 Flag 相关的最外层变量）
    /// 单独提取出来作为一个特殊的“第一轮”进行处理，而不是混入通用的 Sumcheck 循环中。
    /// 这允许在第一轮利用多项式的稀疏或特殊结构进行加速，避免在大规模 Trace 上进行昂贵的通用计算。
    ///
    /// 流程总结：
    /// 1. **UniSkip Round**: 特殊处理第一个（或前几个）变量，生成 `UniSkipFirstRoundProof`。
    /// 2. **Standard Sumcheck**: 处理剩余的变量（通常对应时间步/Cycle 维度），生成 `SumcheckInstanceProof`。
    #[tracing::instrument(skip_all)]
    // 定义 Stage 1 的证明函数。
    // 返回值是一个元组，包含两部分证明：
    // 1. UniSkipFirstRoundProof: 特殊的第一轮证明（处理高次项）。
    // 2. SumcheckInstanceProof: 剩余轮次的标准 Sumcheck 证明。
    fn prove_stage1(
        &mut self,
    ) -> (
        UniSkipFirstRoundProof<F, ProofTranscript>,
        SumcheckInstanceProof<F, ProofTranscript>,
    ) {
        // [调试/监控] 如果不是在 WASM 环境下，打印当前内存使用情况，作为基线。
        #[cfg(not(target_arch = "wasm32"))]
        print_current_memory_usage("Stage 1 baseline");

        tracing::info!("Stage 1 proving");

        // =================================================================
        // 第一部分：UniSkip (Univariate Skip) - 第 0 轮
        // 目标：处理最高次数的约束（如 A*B - C），将其规约为更低维的问题。
        // =================================================================

        // 1. 初始化 UniSkip 参数。
        // 这里会从 Transcript 中获取随机数（如 tau），用于构建 Random Linear Combination (RLC)。
        // RLC 将所有 R1CS 约束或 Lookup 约束压缩成一个单一的多项式。
        let uni_skip_params = OuterUniSkipParams::new(&self.spartan_key, &mut self.transcript);

        // 2. 初始化 UniSkip Prover。
        // 这是一个特定的 Prover 实例，它只负责 Sumcheck 的第一轮。
        // 它拥有 Trace 数据和 Bytecode 数据的所有权或引用，准备进行全量扫描。
        let mut uni_skip = OuterUniSkipProver::initialize(
            uni_skip_params.clone(),
            &self.trace,
            &self.preprocessing.shared.bytecode,
        );

        // 3. 执行 UniSkip 第一轮证明。
        // 这一步调用了我们之前详细分析过的 `prove_uniskip_round` 函数。
        // - 它计算 input_claim (初始总和)。
        // - 它计算外推点 (Zig-Zag points)。
        // - 它更新 opening_accumulator (缓存了 r_0 处的评估值)。
        // - 它返回 first_round_proof。
        let first_round_proof = prove_uniskip_round(
            &mut uni_skip,
            &mut self.opening_accumulator,
            &mut self.transcript,
        );

        // =================================================================
        // 第二部分：Remaining Sumcheck - 剩余轮次
        // 目标：在第 0 轮固定了第一个变量 r_0 后，继续对剩余的变量进行逐层剥离。
        // =================================================================

        // 4. 定义调度表 (Schedule)。
        // Sumcheck 是一个多轮协议，每一轮需要决定：
        // - 计算哪个多项式？
        // - 绑定哪个变量？
        // 这里使用了 `LinearOnlySchedule`，意味着按照线性的顺序处理变量。
        // 轮数计算：tau.len() 是总变量数，减去 UniSkip 处理掉的 1 轮，就是剩余轮数。
        // (注释提到 cycle variables，这通常指此时正在处理的时间步/循环计数器变量)。
        let schedule = LinearOnlySchedule::new(uni_skip_params.tau.len() - 1);

        // 5. 构建共享状态 (Shared State)。
        // 剩余的轮次依然需要访问 Trace 数据和之前计算的参数。
        // 关键点：它传入了 `self.opening_accumulator`。
        // 因为 UniSkip 已经把第 0 轮的结果更新进了 Accumulator，
        // 这里的 SharedState 会读取那个结果作为第 1 轮的 Input Claim。
        let shared = OuterSharedState::new(
            Arc::clone(&self.trace),
            &self.preprocessing.shared.bytecode,
            &uni_skip_params,
            &self.opening_accumulator,
        );

        // 6. 初始化剩余轮次的 Prover。
        // `OuterRemainingStreamingSumcheck` 负责执行剩下的所有轮次。
        // 它支持流式处理 (Streaming)，即不需要一次性把所有数据加载到内存，
        // 而是可以边读取 Trace 边计算，节省内存。
        let mut spartan_outer_remaining: OuterRemainingStreamingSumcheck<_, _> =
            OuterRemainingStreamingSumcheck::new(shared, schedule);

        // 7. 执行批量 Sumcheck (Batched Sumcheck)。
        // 这是一个通用的驱动函数，它会运行一个循环：for i in 1..num_rounds。
        // - 在每一轮，它调用 spartan_outer_remaining.compute_message()。
        // - 提交 Proof 到 Transcript。
        // - 获取随机挑战 r_i。
        // - 更新 opening_accumulator。
        // 返回值：
        // - sumcheck_proof: 包含所有剩余轮次的单变量多项式。
        // - _r_stage1: 这一阶段产生的所有随机数（通常用于后续步骤，这里暂时忽略）。
        let (sumcheck_proof, _r_stage1) = BatchedSumcheck::prove(
            vec![&mut spartan_outer_remaining], // 可以同时处理多个 Instance，这里只有1个
            &mut self.opening_accumulator,      // 持续更新 Accumulator
            &mut self.transcript,               // 持续交互
        );

        // 8. 返回两阶段的完整证明。
        (first_round_proof, sumcheck_proof)
    }

    /// 执行第二阶段的 Sumcheck 证明（Stage 2 Proving）。
    ///
    /// # 作用
    /// Stage 2 是 Jolt 证明系统中并行度最高、任务最繁杂的阶段之一。
    /// 它主要处理以下任务：
    /// 1. **Spartan Grand Product (Product Check)**: 完成从 Stage 1 开始的连乘论证，用于证明 R1CS 矩阵向量乘法中的变量一致性。
    /// 2. **RAM Coherence (内存一致性)**: 证明内存读写操作的正确性（Read/Write Consistency），即每次读取的值必须等于最后一次写入的值。
    /// 3. **Output Check (输出检查)**: 证明程序的公共输出与内存最终状态一致。
    /// 4. **Instruction Lookup (指令查找)**: 将指令执行的正确性归约为查找表查询的正确性。
    ///
    /// # 流程架构
    /// 1. **UniSkip Round**: 首先对 Spartan Product Argument 执行一元跳跃（UniSkip）优化，处理稀疏的指令标志位。
    /// 2. **Batch Sumcheck**: 初始化 5 个不同的 Sumcheck Prover，将它们打包并在同一组随机挑战下并行运行。这极大地减少了 Proof 大小和验证开销。
    #[tracing::instrument(skip_all)]
    fn prove_stage2(
        &mut self,
    ) -> (
        // 返回两部分证明：
        // 1. 第一轮的高次/特殊轮次证明 (针对 Grand Product)。
        UniSkipFirstRoundProof<F, ProofTranscript>,
        // 2. 剩余轮次的批量 Sumcheck 证明 (针对所有 Memory/Instruction 检查)。
        SumcheckInstanceProof<F, ProofTranscript>,
    ) {
        #[cfg(not(target_arch = "wasm32"))]
        print_current_memory_usage("Stage 2 baseline");
        // =================================================================
        // 1. Stage 2a: UniSkip Round (针对 Grand Product 的虚拟化处理)
        // =================================================================
        // Use Case: 启动 Spartan 协议中的 "Grand Product Argument"（连乘论证）证明。
        // 这通常用于证明内存一致性（Memory Consistency）或查找表一致性（Lookup Consistency）。
        // 连乘论证的核心是证明两个集合的元素在某种变换下的乘积相等。
        // Grand Product (用于内存检查和 Lookup) 是一个非常高次的多项式。
        // 我们不能直接跑 Sumcheck，通常需要先用 UniSkip 协议进行一次降维或将其转化为更易处理的形式。

        // 1. Product Virtualization UniSkip (Stage 2a):
        // 类似于 Stage 1，这里处理连乘论证中的第一轮 Sumcheck。
        // 由于 Jolt 使用 One-Hot 编码表示指令，该轮次主要涉及稀疏的指令标志位（Instruction Flags）。
        // 使用 `UniSkip` 优化可以跳过对大量 0 值的计算，快速归约指令维度的变量。
        let uni_skip_params =
            ProductVirtualUniSkipParams::new(&self.opening_accumulator, &mut self.transcript);
        println!("stage2 : uni_skip_params.tau: {:?}", uni_skip_params.tau);
        info!("stage2 2a:  self.opening_accumulator: {:?}", self.opening_accumulator.openings);

        //主要计算5种乘法约束的评估值，并且用eq进行点的加扰，目的是后续证明CPU视角的trace跟内存、查找表视角的输入输出是一致的。
        //此处计算的是CPU视角的值
        //初始化 Prover：专门处理 Product Check 的第一轮。
        let mut uni_skip =
            ProductVirtualUniSkipProver::initialize(uni_skip_params.clone(), &self.trace);

        // 生成第一轮证明，Verifier 对此给出挑战，将问题归约到剩下的轮次（主要是时间/周期维度）。
        // 执行 UniSkip 第一轮：
        // - 计算外推点 (Zig-Zag points)。
        // - 更新 Accumulator (cache_openings)。
        // - 返回 first_round_proof。
        let first_round_proof = prove_uniskip_round(
            &mut uni_skip,
            &mut self.opening_accumulator,
            &mut self.transcript,
        );
        // info!("stage2 2a: self.opening_accumulator: {:?}", self.opening_accumulator.openings);

        // =================================================================
        // 2. 初始化子协议参数 (Initialization Params)
        // Stage 2b 是一个 "Batched Sumcheck"，它同时运行 5 个不同的协议。
        // =================================================================

        // [子协议 1] Product Virtual Remainder
        // 接力上面的 UniSkip。在 UniSkip 处理完 Grand Product 的第一层后，
        // 剩余的部分在这里继续进行 Sumcheck。
        let spartan_product_virtual_remainder_params = ProductVirtualRemainderParams::new(
            self.trace.len(),
            uni_skip_params,
            &self.opening_accumulator,
        );
        // info!("stage2 2.1: self.opening_accumulator: {:?}", self.opening_accumulator.openings);

        // [子协议 2] RAM RAF (Random Access Function) Evaluation
        // 验证内存布局和基本的随机访问逻辑。
        // RAF 通常指 Read-After-Write 的一致性检查参数。
        //one_hot_params是在JoltCpuProver::gen_from_elf方法中初始化JoltCpuProver结构体实例时赋值的，
        // 此时会根据执行轨迹的长度和其他相关参数创建适当的OneHotParams实例，
        // 用于后续的证明生成过程中的One-Hot编码操作。
        let ram_raf_evaluation_params = RafEvaluationSumcheckParams::new(
            &self.program_io.memory_layout,
            &self.one_hot_params,
            &self.opening_accumulator,
        );
        // info!("stage2 2.2: self.opening_accumulator: {:?}", self.opening_accumulator.openings);

        // [子协议 3] RAM Read/Write Checking
        // 这是内存检查的核心。它验证 Trace 中的内存操作日志是否满足
        // 读写一致性（通常使用 Permutation Check 或 Sorting Argument）。
        // 核心内存检查：证明每次“读取”得到的值等于上一次“写入”该地址的值。
        // 这通常涉及排序后的地址元组检查（Multiset Equality Check/Offline Memory Checking）。
        // 需要验证原始 Trace（按时间排序）和重排 Trace（按地址排序）的一致性。
        let ram_read_write_checking_params = RamReadWriteCheckingParams::new(
            &self.opening_accumulator,
            &mut self.transcript,
            &self.one_hot_params,
            self.trace.len(),
            &self.rw_config,
        );
        // info!("stage2 2.3: self.opening_accumulator: {:?}", self.opening_accumulator.openings);

        // [子协议 4] RAM Output Check
        // 证明程序的输出（Standard Output）与其最终内存状态中的相关部分一致。
        // 这确保了 Verifier 收到的程序执行结果确实来自该程序的执行内存。
        let ram_output_check_params = OutputSumcheckParams::new(
            self.one_hot_params.ram_k,
            &self.program_io,
            &mut self.transcript,// 获取随机数用于 Output 检查
        );

        // [子协议 5] Instruction Lookups Claim Reduction
        // 这是一个 "Claim Reduction" 步骤。
        // 它负责证明：Stage 1 中使用的指令查找表 Claim，确实等于
        // 实际执行 Trace 中的指令 Grand Product。
        // 简单说：证明“我执行的指令都是合法的”。

        // 将 Stage 1 中产生的关于指令执行的 Claim，连接到具体的查找表证明上。
        // 确保“执行了 ADD 指令”这件事正确关联到了“ADD 查找表”。
        // 这里进行第一阶段的归约（Phase 1），使用 UniSkip 思想处理指令选择器。
        let instruction_claim_reduction_params =
            InstructionLookupsClaimReductionSumcheckParams::new(
                self.trace.len(),
                &self.opening_accumulator,
                &mut self.transcript,
            );
        // info!("stage2 2.5: self.opening_accumulator: {:?}", self.opening_accumulator.openings);

        // =================================================================
        // 3. 初始化子协议 Prover (Initialization)
        // 根据上面的参数，实例化具体的 Prover 对象。
        // 每个 Prover 都实现了 `SumcheckInstanceProver` trait。
        // =================================================================

        // A. 初始化 Spartan Product Remainder Prover。
        // 这是 Grand Product Argument（连乘论证）的第二阶段。主要是处理CPU查找表的正确性用两个连乘证明来说明两个表的一致性。
        // 它的任务是处理 Cycle（时间步）维度的变量绑定。
        // 结合 Stage 1 的 Univariate Skip，这两个阶段共同证明了 Spartan 协议中的 Grand Product 约束。
        let spartan_product_virtual_remainder = ProductVirtualRemainderProver::initialize(
            spartan_product_virtual_remainder_params,
            Arc::clone(&self.trace),
        );

        // B. 初始化 RAM RAF (Read Access Frequency) Evaluation Prover。
        // 这是一个用于内存检查的关键组件。
        // 此函数实际上是在执行一个加权直方图统计：它遍历整个执行轨迹（Trace），对于每个时间步 `j`，
        // 计算该步骤访问的 RAM 地址 `k`，并将该步骤对应的 Eq 多项式值 `eq(r_cycle, j)`
        // 累加到地址 `k` 的计数桶中。
        let ram_raf_evaluation = RamRafEvaluationSumcheckProver::initialize(
            ram_raf_evaluation_params,
            &self.trace,
            &self.program_io.memory_layout,
        );

        // C. 初始化 RAM Read-Write Checking Prover。
        // 这是内存一致性检查（Memory Consistency Check）的核心部分。
        // 它基于 Offline Memory Checking 技术，负责证明：
        // "按时间顺序的内存访问序列" 与 "按地址排序的内存访问序列" 构成了相同的多重集（Multiset Equality）。
        // 这一步确保了所有的内存读取操作读到的确实是最后一次写入的值。
        let ram_read_write_checking = RamReadWriteCheckingProver::initialize(
            ram_read_write_checking_params,
            &self.trace,
            &self.preprocessing.shared.bytecode,
            &self.program_io.memory_layout,
            &self.initial_ram_state,// 需要初始内存状态
        );

        // D. 初始化 Output Check Prover。
        // 负责证明程序的公共输出（Public Output）与最终内存状态（Final RAM）中对应的 IO 映射区域数据一致。
        let ram_output_check = OutputSumcheckProver::initialize(
            ram_output_check_params,
            &self.initial_ram_state,
            &self.final_ram_state,// 需要最终内存状态
            &self.program_io.memory_layout,
        );

        // E. 初始化 Instruction Lookups Claim Reduction Prover。
        // 负责将“Trace 中每一条指令的执行正确性”这一主张（Claim），
        // 归约到“这些指令及其操作数存在于预计算的查找表中”这一更底层的主张。只对数据做证明，不对指令做证明。
        let instruction_claim_reduction =
            InstructionLookupsClaimReductionSumcheckProver::initialize(
                instruction_claim_reduction_params,
                Arc::clone(&self.trace),
            );

        // 调试用的内存统计（Allocative Feature）
        #[cfg(feature = "allocative")]
        {
            print_data_structure_heap_usage(
                "ProductVirtualRemainderProver",
                &spartan_product_virtual_remainder,
            );
            print_data_structure_heap_usage("RamRafEvaluationSumcheckProver", &ram_raf_evaluation);
            print_data_structure_heap_usage("RamReadWriteCheckingProver", &ram_read_write_checking);
            print_data_structure_heap_usage("OutputSumcheckProver", &ram_output_check);
            print_data_structure_heap_usage(
                "InstructionLookupsClaimReductionSumcheckProver",
                &instruction_claim_reduction,
            );
        }

        // =================================================================
        // 4. 构建实例列表 (The Batch Vector)
        // 将所有不同类型的 Prover 统一装箱 (Box) 放入一个 Vec 中。
        // 利用 Rust 的多态 (Trait Object)，BatchedSumcheck 可以统一驱动它们。
        // =================================================================

        //  打包实例进行批量证明 (Batch Proof)：
        // 将上述 5 个不同的 Prover 放入一个列表。
        // 尽管它们验证逻辑不同（有的验证乘积、有的验证加权和、有的验证 IO），
        // 但它们都是基于多变量多项式的 Sumcheck 协议。
        // `BatchedSumcheck` 允许它们共享 Verifier 的随机挑战（Random Challenges），
        // 从而显著减小证明大小和验证成本（Verifier 在每一轮只需要发送一个挑战 r）。
        let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> = vec![
            Box::new(spartan_product_virtual_remainder),
            Box::new(ram_raf_evaluation),
            Box::new(ram_read_write_checking),
            Box::new(ram_output_check),
            Box::new(instruction_claim_reduction),
        ];
        // [调试] 生成火焰图，可视化各实例的计算开销。
        #[cfg(feature = "allocative")]
        write_instance_flamegraph_svg(&instances, "stage2_start_flamechart.svg");

        tracing::info!("Stage 2 proving");

        // =================================================================
        // 5. 执行 Batched Sumcheck (核心驱动)
        // 这是一个循环过程，处理从第 1 轮到第 k 轮的所有 Sumcheck 逻辑。
        // - instances: 所有子协议 Prover。
        // - opening_accumulator: 持续累积所有子协议在每轮的评估值。
        // - transcript: 生成共享的随机挑战 r_i。
        // =================================================================
        // 迭代执行所有剩余轮次。每一轮，所有 Prover 计算各自的单变量多项式，
        // 聚合后提交给 Verifier，Verifier 返回一个随机数，所有 Prover 根据该随机数折叠多项式进入下一轮。
        // 最终的评估值会累积到 `opening_accumulator` 中，供 Stage 8 统一验证。
        let (sumcheck_proof, _r_stage2) = BatchedSumcheck::prove(
            instances.iter_mut().map(|v| &mut **v as _).collect(),
            &mut self.opening_accumulator,
            &mut self.transcript,
        );

        // [调试] Stage 2 结束后的火焰图。
        #[cfg(feature = "allocative")]
        write_instance_flamegraph_svg(&instances, "stage2_end_flamechart.svg");

        // 后台释放资源，避免阻塞主线程。
        // 这些 Prover 对象在证明结束后不再需要，且可能占用大量内存。
        drop_in_background_thread(instances);

        // 返回 UniSkip 第一轮证明结果（特殊处理）和其余轮次的批量证明结果。
        (first_round_proof, sumcheck_proof)
    }

    /// 执行第三阶段的 Sumcheck 证明（Stage 3 Proving）。
    ///
    /// # 作用
    /// Stage 3 聚焦于 **Spartan Internal Consistency（内部一致性）**。
    /// 如果说 Stage 1 & 2 验证了“计算结果符合查找表”，那么 Stage 3 则验证“我们查找的是正确的数据”。
    /// 主要包含三个核心任务：
    /// 1. **State Transition (Shift Check)**: 验证跨时间步（Cycle）的约束。例如：验证程序计数器（PC）的更新逻辑（顺序执行 vs 跳转），以及 Next Instruction 逻辑。
    /// 2. **Instruction Formatting (Input Validity)**: 验证指令解码逻辑。确保送入 ALU 或查找表的输入（Operands）是根据指令格式（Format）正确地从寄存器值或立即数（Imm）组合而成的。
    /// 3. **Register Access Claims**: 将指令层面的“读写寄存器”行为，归约为底层的寄存器访问 Claim，为 Stage 4 的寄存器一致性检查建立桥梁。
    #[tracing::instrument(skip_all)]
    fn prove_stage3(&mut self) -> SumcheckInstanceProof<F, ProofTranscript> {
        // [性能监控] 打印当前内存占用基线
        #[cfg(not(target_arch = "wasm32"))]
        print_current_memory_usage("Stage 3 baseline");

        // ========================================================================
        // 1. 初始化各子任务参数 (Initialization params)
        // 这一步定义了我们要证明的三个具体的数学约束（多项式关系）。
        // ========================================================================

        // ------------------------------------------------------------------------
        // a. 移位检查参数 (Shift Check Params),只初始化参数，没有建立约束
        // ------------------------------------------------------------------------
        // [算法原理: Shift / Rotation Constraint]
        // 在 Trace 矩阵中，我们需要约束第 t 行和第 t+1 行的关系。
        // 这通过将某一列数据 "Shift"（错位）来实现比较：State[t+1] - f(State[t]) == 0。
        //
        // 主要验证目标：
        // 1. PC (Program Counter) 的连续性：
        //    - 顺序执行：PC_{t+1} = PC_t + 4
        //    - 分支跳转：PC_{t+1} = PC_t + immediate (如果条件满足)
        // 2. RAM 初始化的连续性等。
        //
        // log_2() 表示多项式的变量个数（Trace长度为 2^k，则有 k 个变量）。
        let spartan_shift_params = ShiftSumcheckParams::new(
            self.trace.len().log_2(),
            &self.opening_accumulator, // 传入之前的随机挑战，确保协议连续性
            &mut self.transcript,      // Fiat-Shamir Transcript
        );

        // ------------------------------------------------------------------------
        // b. 指令输入构造参数 (Instruction Input Params),只初始化参数，没有建立约束
        // ------------------------------------------------------------------------
        // [算法原理: Multiplexer (MUX) Logic Constraint]
        // 验证 ALU 或查找表的输入源是否正确。
        // 逻辑公式：Input = (IsReg * RegValue) + (IsImm * ImmediateValue)
        //
        // 目的：
        // 防止恶意 Prover 即使在指令是 ADDI (加立即数) 时，却使用了寄存器的值去欺骗查找表。
        // 确保发往 Stage 1/2 查找表的 "Query" 本身是符合指令定义的。
        let spartan_instruction_input_params =
            InstructionInputParams::new(&self.opening_accumulator, &mut self.transcript);

        // ------------------------------------------------------------------------
        // c. 寄存器使用声明归约 (Registers Claim Reduction),只初始化参数，没有建立约束
        // ------------------------------------------------------------------------
        // [算法原理: Reduction for Offline Memory Checking]
        // 这是一个"桥梁"步骤。
        // Stage 3 并不直接验证内存读写值是否正确（那太慢了），而是生成一组 "Claim"（声明）。
        // Claim 内容： "在时间 t，指令 I 请求读取了寄存器 r，并声称读到的值为 v"。
        //
        // 这些 Claim 会被压缩成多项式，作为 Input 传递给 Stage 4。
        // Stage 4 将使用 "Grand Product Argument" (连乘积论证) 来批量验证这些读写的一致性。
        let spartan_registers_claim_reduction_params = RegistersClaimReductionSumcheckParams::new(
            self.trace.len(),
            &self.opening_accumulator,
            &mut self.transcript,
        );

        // ========================================================================
        // 2. 初始化具体的 Prover 实例 (Initialize Provers)，根据实现trace建立约束，得到多项式点值
        // 这一步加载实际的数据（Trace, Bytecode），将其转化为多线性多项式（MLE）。
        // ========================================================================

        // Shift Prover: 需要 Trace (当前状态) 和 Bytecode (为了获取跳转指令的 offset)
        let spartan_shift = ShiftSumcheckProver::initialize(
            spartan_shift_params,
            Arc::clone(&self.trace),
            &self.preprocessing.shared.bytecode,
        );

        // Input Prover: 需要 Trace (解析指令的操作数)
        let spartan_instruction_input = InstructionInputSumcheckProver::initialize(
            spartan_instruction_input_params,
            &self.trace,
            &self.opening_accumulator,
        );

        // Registers Prover: 需要 Trace (提取 rd, rs1, rs2 索引)
        //Sum(Combined(x) * Eq(x, r)),Combined(x) = Val_rd(x) + gamma * Val_rs1(x) + gamma^2 * Val_rs2(x)
        let spartan_registers_claim_reduction = RegistersClaimReductionSumcheckProver::initialize(
            spartan_registers_claim_reduction_params,
            Arc::clone(&self.trace),
        );

        // [调试工具] 打印各个 Prover 在堆上的内存占用
        #[cfg(feature = "allocative")]
        {
            print_data_structure_heap_usage("ShiftSumcheckProver", &spartan_shift);
            print_data_structure_heap_usage(
                "InstructionInputSumcheckProver",
                &spartan_instruction_input,
            );
            print_data_structure_heap_usage(
                "RegistersClaimReductionSumcheckProver",
                &spartan_registers_claim_reduction,
            );
        }

        // ========================================================================
        // 3. 打包实例 (Batching)
        // [算法原理: Random Linear Combination / Batching]
        // 我们有三个独立的约束方程要证明：P_shift(x)=0, P_input(x)=0, P_reg(x)=0
        //
        // 为了节省 Verifier 的验证开销，我们不分别运行三次 Sumcheck。
        // 而是利用随机数 r (也就是 alpha) 将它们线性组合：
        // P_total(x) = P_shift(x) + r * P_input(x) + r^2 * P_reg(x)
        //
        // 这样只需要运行一次 Sumcheck 协议即可证明所有三个属性。
        // ========================================================================
        let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> = vec![
            Box::new(spartan_shift),
            Box::new(spartan_instruction_input),
            Box::new(spartan_registers_claim_reduction),
        ];

        #[cfg(feature = "allocative")]
        write_instance_flamegraph_svg(&instances, "stage3_start_flamechart.svg");

        tracing::info!("Stage 3 proving");

        // ========================================================================
        // 4. 执行批量 Sumcheck (Prove)
        // [算法原理: Sumcheck Protocol]
        // 这是交互式零知识证明的核心循环。
        //
        // 过程：
        // 1. 迭代 log(N) 轮（N 是 Trace 长度）。
        // 2. 每轮 Prover 将多变量多项式 P(x_1, ..., x_k) 固定住一个变量，
        //    变成单变量多项式 g(x_1) 并发送给 Verifier。
        // 3. Verifier 发送随机挑战 r。
        // 4. Prover 利用 r "折叠" (Fold) 多项式，将变量数减一。
        //
        // 最终结果：
        // 返回一个 SumcheckProof，包含每一轮的单变量多项式评估值。
        // 此时多变量问题被简化为了对单个随机点 Evaluation 的检查。
        // ========================================================================
        let (sumcheck_proof, _r_stage3) = BatchedSumcheck::prove(
            instances.iter_mut().map(|v| &mut **v as _).collect(),
            &mut self.opening_accumulator,
            &mut self.transcript,
        );

        #[cfg(feature = "allocative")]
        write_instance_flamegraph_svg(&instances, "stage3_end_flamechart.svg");

        // ========================================================================
        // 5. 资源释放 (Cleanup)
        // [工程优化: Async Drop]
        // Sumcheck 结束后，instances 中包含巨大的多项式数据（可能占用 GB 级内存）。
        // 直接 drop 会导致主线程阻塞几百毫秒甚至数秒。
        // 这里的 drop_in_background_thread 将内存释放工作扔给后台线程，
        // 让主线程可以立即开始 Stage 4 的计算，提高 Pipeline 效率。
        // ========================================================================
        drop_in_background_thread(instances);

        sumcheck_proof
    }

    /// 执行第四阶段的 Sumcheck 证明（Stage 4 Proving）。
    ///
    /// # 作用
    /// Stage 4 聚焦于 **数据值的一致性与完整性**。
    /// 前面的阶段验证了指令的执行逻辑、内存访问的地址一致性等，而本阶段主要验证：
    /// 1. **Register Consistency (寄存器一致性)**: 类似于 RAM，寄存器文件也是一种读写存储。必须证明程序对寄存器的读取总是返回最后一次写入的值。
    /// 2. **RAM Values (内存数值)**: 验证内存读写操作中涉及的具体数值是否正确，特别是涉及到初始内存状态（Initial RAM）和最终内存状态（Final RAM）的边界条件。
    /// 3. **Advice Integration (辅助输入整合)**: 将 Proof 系统外部的不可信输入（Untrusted Advice）和可信输入（Trusted Advice）正式纳入证明的约束体系中。
    ///
    /// # 核心逻辑
    /// - **Offline Memory Checking**: 对寄存器访问记录进行排序和置换检查。
    /// - **Advice Folding**: 将 Advice 多项式的评估请求折叠到全局累加器中，以便统一验证。
    /// - **Batched Sumcheck**: 并行处理寄存器检查、RAM 值检查和 RAM 终态检查。
    #[tracing::instrument(skip_all)]
    fn prove_stage4(&mut self) -> SumcheckInstanceProof<F, ProofTranscript> {
        #[cfg(not(target_arch = "wasm32"))]
        print_current_memory_usage("Stage 4 baseline");

        // ========================================================================
        // 1. 寄存器读写检查参数 (Registers Read/Write Checking Params)
        // ========================================================================
        // [算法原理: Offline Memory Checking - 寄存器部分]
        // 寄存器堆较小（通常32个），访问非常频繁。
        // 这里主要准备验证核心内存等式：
        // {所有写入操作} ∪ {初始状态} == {所有读取操作} ∪ {最终状态}
        //
        // 此步骤生成用于验证 "Read-Write Consistency" 的参数，
        // 确保每次读取寄存器得到的值，都是上一次写入该寄存器的值。
        let registers_read_write_checking_params = RegistersReadWriteCheckingParams::new(
            self.trace.len(),
            &self.opening_accumulator,
            &mut self.transcript,
            &self.rw_config, // 读写配置，定义了哪些是读操作，哪些是写操作
        );

        // ========================================================================
        // 2. 累积 Advice (Accumulate Advice)
        // ========================================================================
        // [算法原理: Advice Commitment & Fiat-Shamir]
        // 在离线内存检查中，Prover 需要提供额外的 "Advice"（建议/辅助信息）。
        // 最关键的 Advice 是 "按地址排序的 Trace" (Sorted Trace)。
        //
        // Verifier 无法自己排序（计算量太大），所以 Prover 提供排序后的数据。
        // 此函数将这些 Advice 多项式（untrusted）进行承诺，并混入 Transcript，
        // 以便 Verifier 生成随机挑战，防止 Prover 伪造排序数据。
        prover_accumulate_advice(
            &self.advice.untrusted_advice_polynomial, // Prover 提供的辅助多项式（如排序后的内存访问）
            &self.advice.trusted_advice_polynomial,   // 可信的系统多项式
            &self.program_io.memory_layout,
            &self.one_hot_params,
            &mut self.opening_accumulator,
            &mut self.transcript,
            // 检查是否需要单点 Opening（优化策略，取决于 Trace 长度）
            self.rw_config
                .needs_single_advice_opening(self.trace.len().log_2()),
        );

        // ========================================================================
        // 3. 初始化 RAM 相关的参数 (RAM Init & Final Params)
        // ========================================================================
        // [算法原理: RAM Consistency & One-Hot Encoding]
        // RAM 与寄存器不同，地址空间巨大（如 2^64），但实际访问稀疏。
        // Jolt 将 RAM 检查拆分为不同部分以优化性能。

        // a. RAM 值评估 (Val Evaluation)
        // 这通常涉及验证内存的 "初始状态 (Init)" 或特定时间点的正确性。
        // one_hot_params 暗示这里可能使用了 One-Hot 编码来处理地址匹配，
        // 或者用于验证读操作命中了正确的内存单元。
        let ram_val_evaluation_params = ValEvaluationSumcheckParams::new_from_prover(
            &self.one_hot_params,
            &self.opening_accumulator,
            &self.initial_ram_state, // 必须确保读取未写入地址时返回初始值（通常是0）
            self.trace.len(),
        );

        // b. RAM 最终状态 (Val Final)
        // [算法原理: Final State Consistency]
        // 内存检查等式的右半部分需要包含 "最终内存状态 (Final Memory State)"。
        // 验证： Init ∪ Writes == Reads ∪ Final
        // 此参数用于生成证明 "Final" 集合正确性的多项式。
        let ram_val_final_params =
            ValFinalSumcheckParams::new_from_prover(self.trace.len(), &self.opening_accumulator);

        // ========================================================================
        // 4. 初始化具体的 Prover 实例 (Initialize Provers)
        // ========================================================================

        // 4a. 寄存器读写一致性 Prover
        // [算法原理: Grand Product Argument]
        // 构建连乘积多项式，证明寄存器的访问记录在时间维度和空间（地址）维度上是一致的。
        let registers_read_write_checking = RegistersReadWriteCheckingProver::initialize(
            registers_read_write_checking_params,
            self.trace.clone(),
            &self.preprocessing.shared.bytecode,
            &self.program_io.memory_layout,
        );

        // 4b. RAM 初始化/评估 Prover
        // 负责证明 RAM 操作与初始状态的一致性。
        let ram_val_evaluation = RamValEvaluationSumcheckProver::initialize(
            ram_val_evaluation_params,
            &self.trace,
            &self.preprocessing.shared.bytecode,
            &self.program_io.memory_layout,
        );

        // 4c. RAM 最终状态 Prover
        // 负责证明执行结束后的 RAM 状态是所有历史操作的正确累积结果。
        let ram_val_final = ValFinalSumcheckProver::initialize(
            ram_val_final_params,
            &self.trace,
            &self.preprocessing.shared.bytecode,
            &self.program_io.memory_layout,
        );

        #[cfg(feature = "allocative")]
        {
            print_data_structure_heap_usage(
                "RegistersReadWriteCheckingProver",
                &registers_read_write_checking,
            );
            print_data_structure_heap_usage("RamValEvaluationSumcheckProver", &ram_val_evaluation);
            print_data_structure_heap_usage("ValFinalSumcheckProver", &ram_val_final);
        }

        // ========================================================================
        // 5. 打包实例 (Batching)
        // ========================================================================
        // [算法原理: Batched Sumcheck / Random Linear Combination]
        // 将寄存器检查、RAM 初始检查、RAM 最终检查这三个任务打包。
        // 通过随机线性组合，将它们合并为一个 Sumcheck 协议。
        // 这样，Verifier 只需要验证一次 Sumcheck，就能确信这三部分（即整个内存系统）都是正确的。
        let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> = vec![
            Box::new(registers_read_write_checking),
            Box::new(ram_val_evaluation),
            Box::new(ram_val_final),
        ];

        #[cfg(feature = "allocative")]
        write_instance_flamegraph_svg(&instances, "stage4_start_flamechart.svg");
        tracing::info!("Stage 4 proving");

        // ========================================================================
        // 6. 执行批量 Sumcheck (Prove)
        // ========================================================================
        // 驱动 Sumcheck 协议，进行多轮折叠（Folding），生成证明。
        let (sumcheck_proof, _r_stage4) = BatchedSumcheck::prove(
            instances.iter_mut().map(|v| &mut **v as _).collect(),
            &mut self.opening_accumulator,
            &mut self.transcript,
        );

        #[cfg(feature = "allocative")]
        write_instance_flamegraph_svg(&instances, "stage4_end_flamechart.svg");

        // [工程优化] 后台线程释放内存
        drop_in_background_thread(instances);

        sumcheck_proof
    }

    /// 执行第五阶段的 Sumcheck 证明（Stage 5 Proving）。
    ///
    /// # 作用
    /// Stage 5 是 Jolt 证明系统中的“连接层”阶段。它不再像前几个阶段那样关注高层逻辑（如指令执行流、内存一致性），
    /// 而是开始将这些高层概念“落地”到底层的表示形式（如 Bit 级的查找表索引）。
    ///
    /// 主要包含三个核心任务：
    /// 1. **Register Value Evaluation (寄存器值验证)**: 验证寄存器文件中的具体数值。Stage 4 验证了“读写了一致的东西”，Stage 5 验证“那个东西的值到底是什么”。
    /// 2. **RAM Address Reduction (内存地址归约)**: 将高层的 64 位内存地址访问声明，归约为底层的 bit 级或 chunk 级的约束。这通常是为了连接 Offline Memory Checking 和查找表。
    /// 3. **Instruction Lookup RAF (指令查找表读取频次)**: 验证指令执行对查找表的访问模式。证明“ADD 指令确实在访问 ADD 表”。
    #[tracing::instrument(skip_all)]
    fn prove_stage5(&mut self) -> SumcheckInstanceProof<F, ProofTranscript> {
        #[cfg(not(target_arch = "wasm32"))]
        print_current_memory_usage("Stage 5 baseline");

        // 1. 初始化各子任务参数 (Initialization params)：

        // a. 寄存器值评估参数 (Registers Value Evaluation Params):
        // 这一步是为了验证寄存器状态的具体数值。
        // 在 Stage 4 中，通过 Offline Memory Checking 保证了寄存器读写的一致性（即 Multiset Equality）。
        // 但这只保证了“读 = 写”，没有保证“写”进去的值是有效的 Field Element 或符合特定约束。
        // `RegistersValEvaluation` 补全了这一环，确保寄存器值的多项式表示是正确的。
        let registers_val_evaluation_params =
            RegistersValEvaluationSumcheckParams::new(&self.opening_accumulator);

        // b. RAM 地址归约参数 (RAM Read Address Production/Reduction):
        // 这是一个“归约”步骤。
        // 在之前的阶段（如 Stage 2 RAM Coherence），我们证明了内存地址 $A$ 上的读写一致性。
        // 但为了在后续阶段使用查找表或其他机制检查地址本身的合法性或构造方式，
        // 我们需要将关于完整地址 $A$ 的 Claim，分解（归约）为关于地址各个部分的更细粒度的 Claim。
        // `Ra` 代表 Read Address。
        let ram_ra_reduction_params = RaReductionParams::new(
            self.trace.len(),
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );

        // c. 指令查找表读取标记参数 (Instruction Lookups Read RAF):
        // RAF = Read Access Frequency (读取访问频次/标记)。
        // Jolt 的核心思想是将指令执行视为查找表查询。如果 CPU 执行了一条 ADD 指令，
        // 那么在逻辑上，它应该去所有的指令查找表中，唯独“激活” ADD 表的查询，其他表（如 MUL, SUB）应视为未激活。
        // 此参数用于证明这一选择逻辑：根据指令的 Opcode，正确地生成了对特定查找表的访问计数。
        let lookups_read_raf_params = InstructionReadRafSumcheckParams::new(
            self.trace.len().log_2(),
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );

        // 2. 初始化具体的 Prover 实例 (Initialize):
        // 根据参数创建负责实际多项式计算的对象。

        // 初始化寄存器值证明器：
        // 需要 Trace 来获取每一时刻寄存器的实际值。
        // 需要 Bytecode 和 MemoryLayout 来处理一些隐式的寄存器行为（例如涉及 PC 或特殊内存映射寄存器的情况）。
        let registers_val_evaluation = RegistersValEvaluationSumcheckProver::initialize(
            registers_val_evaluation_params,
            &self.trace,
            &self.preprocessing.shared.bytecode,
            &self.program_io.memory_layout,
        );

        // 初始化 RAM 地址归约证明器：
        // 它的工作是将关于 RAM 地址的多项式评估请求，转换/归约为关于 One-Hot 编码参数的评估请求（如果有用到 One-Hot 优化）。
        let ram_ra_reduction = RamRaClaimReductionSumcheckProver::initialize(
            ram_ra_reduction_params,
            &self.trace,
            &self.program_io.memory_layout,
            &self.one_hot_params,
        );

        // 初始化指令查找表 RAF 证明器：
        // 遍历 Trace，统计每种指令查找表被访问的情况。这实际上是构建一个“指令类型 -> 查找表索引”的映射证明。
        let lookups_read_raf = InstructionReadRafSumcheckProver::initialize(
            lookups_read_raf_params,
            Arc::clone(&self.trace),
        );

        // 调试统计信息（Allocative Feature）：
        // 打印堆内存使用情况。Stage 5 的 Prover 通常涉及较大的中间状态。
        #[cfg(feature = "allocative")]
        {
            print_data_structure_heap_usage(
                "RegistersValEvaluationSumcheckProver",
                &registers_val_evaluation,
            );
            print_data_structure_heap_usage("RamRaClaimReductionSumcheckProver", &ram_ra_reduction);
            print_data_structure_heap_usage("InstructionReadRafSumcheckProver", &lookups_read_raf);
        }

        // 3. 打包实例进行批量证明 (Batching):
        // 将寄存器值检查、RAM 地址归约、Lookup 访问控制检查这三个独立的 Sumcheck 实例打包。
        // 它们将共享 Verifier 发送的每一轮随机挑战 $r$，从而显著节省 Proof 大小。
        let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> = vec![
            Box::new(registers_val_evaluation),
            Box::new(ram_ra_reduction),
            Box::new(lookups_read_raf),
        ];

        #[cfg(feature = "allocative")]
        write_instance_flamegraph_svg(&instances, "stage5_start_flamechart.svg");

        tracing::info!("Stage 5 proving");

        // 4. 执行批量 Sumcheck (Prove):
        // 驱动协议执行。
        // 经过 $\log(N)$ 轮交互，将上述复杂的验证逻辑归约为 `opening_accumulator` 中几个简单的多项式点值校验。
        let (sumcheck_proof, _r_stage5) = BatchedSumcheck::prove(
            instances.iter_mut().map(|v| &mut **v as _).collect(),
            &mut self.opening_accumulator,
            &mut self.transcript,
        );

        #[cfg(feature = "allocative")]
        write_instance_flamegraph_svg(&instances, "stage5_end_flamechart.svg");

        // 后台释放资源：
        // 避免在主线程进行昂贵的 Drop 操作，减少停顿。
        drop_in_background_thread(instances);

        sumcheck_proof
    }

    /// 执行第六阶段的 Sumcheck 证明（Stage 6 Proving）。
    ///
    /// # 作用
    /// Stage 6 深入到 Jolt 系统的底层约束层面，主要负责以下几类关键验证：
    /// 1. **Booleanity (布尔性/位有效性)**: 证明某些被声明为“比特”的变量，其值确实严格为 0 或 1。
    ///    这是基于位分解（Bit-Decomposition）的 Lookup 系统（如 Lasso）安全性的基石。
    /// 2. **Bytecode Access (字节码访问)**: 验证程序计数器（PC）与指令读取（ROM）之间的映射关系。
    /// 3. **Virtual Component Address (虚拟地址构造)**: 验证 RAM 和查找表访问所使用的“虚拟地址”是否由各个片段（Chunks）正确组合而成。
    /// 4. **Advice Reduction Phase 1 (辅助输入归约-阶段1)**: Jolt 处理 Advice 采用两阶段归约策略，
    ///    本阶段负责第一步：解除对“时间步/周期 (Cycle)”维度的依赖，将多维 Lookup 简化。
    #[tracing::instrument(skip_all)]
    fn prove_stage6(&mut self) -> SumcheckInstanceProof<F, ProofTranscript> {
        #[cfg(not(target_arch = "wasm32"))]
        print_current_memory_usage("Stage 6 baseline");

        // =========================================================
        // 1. 初始化各子任务参数 (Initialization params)
        // Stage 6 进入了更底层的约束检查，主要关注“位有效性”(Booleanity)
        // 和“访问标记”(Access Flags) 以及 Advice 的初步归约。
        // =========================================================

        // a. 字节码读取标记 (Bytecode Read RAF):
        // 验证程序计数器 (PC) 指向的指令是否被正确从只读代码段 (ROM) 中读取。
        // RAF (Read Access Flag/Frequency) ��保我们在执行指令时，确实对 Bytecode 对应的地址发起了读操作。
        // 如果 Trace 说 "在 PC=100 执行了 ADD"，这里证明 "ROM[100] 确实是 ADD"。
        let bytecode_read_raf_params = BytecodeReadRafSumcheckParams::gen(
            &self.preprocessing.shared.bytecode,
            self.trace.len().log_2(),
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );

        // b. RAM 汉明重量布尔性 (RAM Hamming Booleanity):
        // 用于优化内存检查的辅助标志位验证。
        // 它验证与 RAM 访问相关的某些内部标志位是否严格为布尔值 (0 或 1)。
        let ram_hamming_booleanity_params =
            HammingBooleanitySumcheckParams::new(&self.opening_accumulator);

        // c. 通用布尔性检查 (Booleanity):
        // Jolt ���赖将数值分解为比特位来进行范围检查或查找表索引。
        // 此步骤对于整个系统的安全性至关重要：它证明那些声称是“比特”的变量 $x$，
        // 其值确实满足约束 $x \cdot (x - 1) = 0$。
        // 如果攻击者能偷运一个非 0/1 的值作为“比特”，整个查找表逻辑可能会崩塌。
        let booleanity_params = BooleanitySumcheckParams::new(
            self.trace.len().log_2(),
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );

        // d. RAM 和 Lookup 的虚拟地址读取 (Virtual Read Addresses):
        // "Virtual" 在这里通常指代 Jolt 内部为了 Lasso 查找参数所构建的查询结构。
        // 在 Lasso 中，大域元素的查找通常被分解为多个小 chunk 的查找。
        // 这些参数用于验证：我们根据 Trace 数据构建出的用于查询这些小表的“虚拟地址”是合法的。
        let ram_ra_virtual_params = RamRaVirtualParams::new(
            self.trace.len(),
            &self.one_hot_params,
            &self.opening_accumulator,
        );
        let lookups_ra_virtual_params = InstructionRaSumcheckParams::new(
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );

        // e. 增量声明归约 (Instruction Counter / Increment Reduction):
        // 验证系统中的计数器或增量逻辑。
        // 比如确保某些指令 ID 或步骤计数器是线性增长的，或者正确处理了 Padding 部分。
        let inc_reduction_params = IncClaimReductionSumcheckParams::new(
            self.trace.len(),
            &self.opening_accumulator,
            &mut self.transcript,
        );

        // f. Advice (辅助输入) 归约 - 第一阶段 (Phase 1):
        // Advice 验证是一个两阶段过程。Stage 6 执行第一阶段。
        // ���始问题是：“在第 t 步，地址 a 的值为 v”。这是一个依赖 (t, a, v) 的三元关系。
        // Phase 1 的目标是消除“时间/周期 (Cycle, t)”维度的依赖，将其归约为关于 (Address) 的校验。
        // 这里分别处理 Trusted（预定义）和 Untrusted（私有 Witness）两类 Advice。
        let trusted_advice_phase1_params = AdviceClaimReductionPhase1Params::new(
            AdviceKind::Trusted,
            &self.program_io.memory_layout,
            self.trace.len(),
            &self.opening_accumulator,
            &mut self.transcript,
            self.rw_config
                .needs_single_advice_opening(self.trace.len().log_2()),
        );
        let untrusted_advice_phase1_params = AdviceClaimReductionPhase1Params::new(
            AdviceKind::Untrusted,
            &self.program_io.memory_layout,
            self.trace.len(),
            &self.opening_accumulator,
            &mut self.transcript,
            self.rw_config
                .needs_single_advice_opening(self.trace.len().log_2()),
        );

        // =========================================================
        // 2. 初始化具体的 Prover 实例
        // =========================================================

        let bytecode_read_raf = BytecodeReadRafSumcheckProver::initialize(
            bytecode_read_raf_params,
            Arc::clone(&self.trace),
            Arc::clone(&self.preprocessing.shared.bytecode),
        );
        let ram_hamming_booleanity =
            HammingBooleanitySumcheckProver::initialize(ram_hamming_booleanity_params, &self.trace);

        let booleanity = BooleanitySumcheckProver::initialize(
            booleanity_params,
            &self.trace,
            &self.preprocessing.shared.bytecode, // 某些指令隐含了地址位的分解，需要 Bytecode 信息辅助构建多项式
            &self.program_io.memory_layout,
        );

        let ram_ra_virtual = RamRaVirtualSumcheckProver::initialize(
            ram_ra_virtual_params,
            &self.trace,
            &self.program_io.memory_layout,
            &self.one_hot_params,
        );
        let lookups_ra_virtual =
            LookupsRaSumcheckProver::initialize(lookups_ra_virtual_params, &self.trace);
        let inc_reduction =
            IncClaimReductionSumcheckProver::initialize(inc_reduction_params, self.trace.clone());

        // 初��化 Advice Phase 1 Provers。
        // 注意：这里我们 Clone 了 advice 多项式。原因如下：
        // 1. Phase 1 (Stage 6) 会破坏性地绑定“周期(Cycle)”变量，修改多项式视角。
        // 2. Phase 2 (Stage 7) 将需要一份新的副本来绑定“地址(Address)”变量。
        // 3. Stage 8 (RLC/Opening) 可能还需要原始多项式。
        // 虽然有性能开销（Clone），但为了多阶段 Sumcheck 的独立性与安全性是必要的。
        let trusted_advice_phase1 = trusted_advice_phase1_params.map(|params| {
            // 保存 gamma 值，Phase 2 会用到
            self.advice_reduction_gamma_trusted = Some(params.gamma);
            let poly = self
                .advice
                .trusted_advice_polynomial
                .clone()
                .expect("trusted advice params exist but polynomial is missing");
            AdviceClaimReductionPhase1Prover::initialize(params, poly)
        });
        let untrusted_advice_phase1 = untrusted_advice_phase1_params.map(|params| {
            self.advice_reduction_gamma_untrusted = Some(params.gamma);
            let poly = self
                .advice
                .untrusted_advice_polynomial
                .clone()
                .expect("untrusted advice params exist but polynomial is missing");
            AdviceClaimReductionPhase1Prover::initialize(params, poly)
        });

        #[cfg(feature = "allocative")]
        {
            print_data_structure_heap_usage("BytecodeReadRafSumcheckProver", &bytecode_read_raf);
            print_data_structure_heap_usage(
                "ram HammingBooleanitySumcheckProver",
                &ram_hamming_booleanity,
            );
            print_data_structure_heap_usage("BooleanitySumcheckProver", &booleanity);
            print_data_structure_heap_usage("RamRaSumcheckProver", &ram_ra_virtual);
            print_data_structure_heap_usage("LookupsRaSumcheckProver", &lookups_ra_virtual);
            print_data_structure_heap_usage("IncClaimReductionSumcheckProver", &inc_reduction);
            if let Some(ref advice) = trusted_advice_phase1 {
                print_data_structure_heap_usage(
                    "AdviceClaimReductionPhase1Prover(trusted)",
                    advice,
                );
            }
            if let Some(ref advice) = untrusted_advice_phase1 {
                print_data_structure_heap_usage(
                    "AdviceClaimReductionPhase1Prover(untrusted)",
                    advice,
                );
            }
        }

        // =========================================================
        // 3. 打包实例进行批量证明 (Batching)
        // 将所有关于位有效性、指令读取、地址构造和 Advice 初步归约的检查合并。
        // 这 6-8 个独立的 Sumcheck 实例将共享同一个随机挑战向量 $r$。
        // =========================================================
        let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> = vec![
            Box::new(bytecode_read_raf),
            Box::new(ram_hamming_booleanity),
            Box::new(booleanity),
            Box::new(ram_ra_virtual),
            Box::new(lookups_ra_virtual),
            Box::new(inc_reduction),
        ];
        if let Some(advice) = trusted_advice_phase1 {
            instances.push(Box::new(advice));
        }
        if let Some(advice) = untrusted_advice_phase1 {
            instances.push(Box::new(advice));
        }

        #[cfg(feature = "allocative")]
        write_instance_flamegraph_svg(&instances, "stage6_start_flamechart.svg");

        tracing::info!("Stage 6 proving");

        // 4. 执行 Sumcheck
        // 验证上述所有属性。
        // 这一步产生的 `sumcheck_proof` 极其紧凑，却包含了对海量底层位操作有效性和内存访问正确性的保证。
        let (sumcheck_proof, _r_stage6) = BatchedSumcheck::prove(
            instances.iter_mut().map(|v| &mut **v as _).collect(),
            &mut self.opening_accumulator,
            &mut self.transcript,
        );

        #[cfg(feature = "allocative")]
        write_instance_flamegraph_svg(&instances, "stage6_end_flamechart.svg");

        // 后台释放资源
        // 这些 Prover 包含的大量的多项式数据在此刻之后不再需要。
        drop_in_background_thread(instances);

        sumcheck_proof
    }

    /// Stage 7: HammingWeight + ClaimReduction sumcheck (only log_k_chunk rounds).
    #[tracing::instrument(skip_all)]
    fn prove_stage7(&mut self) -> SumcheckInstanceProof<F, ProofTranscript> {
        #[cfg(not(target_arch = "wasm32"))]
        print_current_memory_usage("Stage 7 baseline");

        // 1. 初始化汉明重量（One-Hot）检查参数 (Hamming Weight Check):
        // Jolt 架构极度依赖 One-Hot 编码（例如指令解码，同一时刻只能是一个指令）。
        // 此步骤验证特定向量的汉明重量是否为 1（即所有位中只有一个 1，其余为 0）。
        // 注意：前面的 Stage 6 (Booleanity) 已经证明了这些值是 0 或 1。
        // 这里证明的是 sum(flags) == 1。
        // r_cycle 和 r_addr_bool 等随机点是从之前的 Booleanity opening 中提取的，用于关联这两个阶段。
        let hw_params = HammingWeightClaimReductionParams::new(
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );

        // 初始化汉明重量证明器
        let hw_prover = HammingWeightClaimReductionProver::initialize(
            hw_params,
            &self.trace,
            &self.preprocessing.shared,
            &self.one_hot_params,
        );

        #[cfg(feature = "allocative")]
        print_data_structure_heap_usage("HammingWeightClaimReductionProver", &hw_prover);

        // 2. 准备批量证明实例 (Batching):
        // 主要包含汉明重量检查。
        // 这里的 Sumcheck 通常只涉及 "Address" 相关的 rounds (log_k_chunk)，
        // 因为时间维度通常已经在前序步骤处理或归约了。
        let mut instances: Vec<Box<dyn SumcheckInstanceProver<F, ProofTranscript>>> =
            vec![Box::new(hw_prover)];

        // 3. 添加 Advice (辅助输入) 归约 - 第二阶段 (Phase 2):
        // 如果存在 Advice 数据，我们需要完成在 Stage 6 启动的验证流程。
        // Stage 6 (Phase 1) 处理了 "Time/Cycle" 维度的归约。
        // Stage 7 (Phase 2) 处理 "Address/Space" 维度的归约。
        // 也就是验证：Confirm(Address) -> Value。

        // 处理可信 Advice (Trusted Advice)
        if let Some(gamma) = self.advice_reduction_gamma_trusted {
            if let Some(params) = AdviceClaimReductionPhase2Params::new(
                AdviceKind::Trusted,
                &self.program_io.memory_layout,
                self.trace.len(),
                gamma, // 使用 Stage 6 生成的随机挑战 gamma
                &self.opening_accumulator,
                self.rw_config
                    .needs_single_advice_opening(self.trace.len().log_2()),
            ) {
                // 克隆多项式以进行 Sumcheck（该过程可能具有破坏性或需要独立的所有权）
                let poly = self
                    .advice
                    .trusted_advice_polynomial
                    .clone()
                    .expect("trusted advice phase2 params exist but polynomial is missing");
                instances.push(Box::new(AdviceClaimReductionPhase2Prover::initialize(
                    params, poly,
                )));
            }
        }

        // 处理不可信 Advice (Untrusted Advice / Witness)
        if let Some(gamma) = self.advice_reduction_gamma_untrusted {
            if let Some(params) = AdviceClaimReductionPhase2Params::new(
                AdviceKind::Untrusted,
                &self.program_io.memory_layout,
                self.trace.len(),
                gamma,
                &self.opening_accumulator,
                self.rw_config
                    .needs_single_advice_opening(self.trace.len().log_2()),
            ) {
                let poly = self
                    .advice
                    .untrusted_advice_polynomial
                    .clone()
                    .expect("untrusted advice phase2 params exist but polynomial is missing");
                instances.push(Box::new(AdviceClaimReductionPhase2Prover::initialize(
                    params, poly,
                )));
            }
        }

        #[cfg(feature = "allocative")]
        write_instance_flamegraph_svg(&instances, "stage7_start_flamechart.svg");
        tracing::info!("Stage 7 proving");

        // 4. 执行批量 Sumcheck (Prove):
        // 这一步将汉明重量约束和 Advice 地址约束归约到 `opening_accumulator`。
        // 此后，Verifier 只需要检查 Accumulator 中的点评估值。
        let (sumcheck_proof, _) = BatchedSumcheck::prove(
            instances.iter_mut().map(|v| &mut **v as _).collect(),
            &mut self.opening_accumulator,
            &mut self.transcript,
        );

        #[cfg(feature = "allocative")]
        write_instance_flamegraph_svg(&instances, "stage7_end_flamechart.svg");

        // 后台释放资源
        drop_in_background_thread(instances);

        sumcheck_proof
    }

    /// Stage 8: Dory batch opening proof.
    /// Builds streaming RLC polynomial directly from trace (no witness regeneration needed).
    /// Stage 8: Dory 批量 Opening 证明 (Dory batch opening proof)。
    ///
    /// 这里的核心是直接从 Trace 构建流式的随机线性组合 (RLC) 多项式，
    /// 而不需要重新生成巨大的 Witness 多项式，极大节省内存。
    #[tracing::instrument(skip_all)]
    fn prove_stage8(
        &mut self,
        opening_proof_hints: HashMap<CommittedPolynomial, PCS::OpeningProofHint>,
    ) -> PCS::Proof {
        tracing::info!("Stage 8 proving (Dory batch opening)");

        // 初始化 Dory 上下文。Dory 是一种基于 Hyrax/Spartan 风格的多项式承诺方案。
        // 这里设置全局参数，准备进行大规模的 MSM (多标量乘法) 计算。
        let _guard = DoryGlobals::initialize_context(
            self.one_hot_params.k_chunk,
            self.padded_trace_len,
            DoryContext::Main,
        );

        // 1. 获取统一的 Opening Point (验证点)。
        // 这个点来自 Stage 7 (HammingWeightClaimReduction)。
        // 它通常是一个拼接点：(r_address || r_cycle)。
        // 也就是：地址部分的随机挑战 + 时间/周期部分的随机挑战。
        let (opening_point, _) = self.opening_accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::InstructionRa(0),
            SumcheckId::HammingWeightClaimReduction,
        );
        let log_k_chunk = self.one_hot_params.log_k_chunk;

        // 分离出地址部分的随机挑战点 r_address。
        let r_address_stage7 = &opening_point.r[..log_k_chunk];

        // 2. 收集所有的 (多项式标识, 声明值) 对。
        // 我们要在一个点上同时打开许多个多项式，验证它们的值。
        let mut polynomial_claims = Vec::new();

        // category A: 稠密多项式 (Dense Polynomials) - RamInc 和 RdInc
        // 这些多项式仅依赖于时间维度 (Cycle)，长度为 Trace Length (log_T)。
        // 但我们的 Opening Point 包含了额外的 Address 维度。
        // 来源：Stage 6 的 IncClaimReduction。
        let (_ram_inc_point, ram_inc_claim) =
            self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::RamInc,
                SumcheckId::IncClaimReduction,
            );
        let (_rd_inc_point, rd_inc_claim) =
            self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::RdInc,
                SumcheckId::IncClaimReduction,
            );

        #[cfg(test)]
        {
            // 验证一致性：Inc 检查所用的随机点应该等于 Unified Point 中的时间部分。
            let r_cycle_stage6 = &opening_point.r[log_k_chunk..];

            debug_assert_eq!(
                _ram_inc_point.r.as_slice(),
                r_cycle_stage6,
                "RamInc opening point should match r_cycle from HammingWeightClaimReduction"
            );
            debug_assert_eq!(
                _rd_inc_point.r.as_slice(),
                r_cycle_stage6,
                "RdInc opening point should match r_cycle from HammingWeightClaimReduction"
            );
        }

        // 关键逻辑：应用拉格朗日因子 (Lagrange Factor)。
        // 稠密多项式 P(cycle) 比 Opening Point (address, cycle) 少了 address 维度。
        // 为了将它们嵌入到同一个大的 Dory 矩阵中进行批量验证，我们视作它们只在 address=0 处有值。
        // 拉格朗日因子 L(r_addr) = ∏(1 - r_i) 实际上计算了 "r_addr 是否指向 0 (全零向量)" 的概率权重。
        // 这相当于在这个高维空间中进行零填充嵌入 (Zero-padding embedding)。
        let lagrange_factor: F = r_address_stage7.iter().map(|r| F::one() - *r).product();

        polynomial_claims.push((CommittedPolynomial::RamInc, ram_inc_claim * lagrange_factor));
        polynomial_claims.push((CommittedPolynomial::RdInc, rd_inc_claim * lagrange_factor));

        // category B: 稀疏/结构化多项式 (Sparse Polynomials) - RA Polys
        // 这些多项式本身就定义在 (Address, Cycle) 上，维度与 Opening Point 开头完全匹配。
        // 直接添加声明值即可。
        // 来源：Stage 7 的 HammingWeightClaimReduction。
        for i in 0..self.one_hot_params.instruction_d {
            let (_, claim) = self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::InstructionRa(i),
                SumcheckId::HammingWeightClaimReduction,
            );
            polynomial_claims.push((CommittedPolynomial::InstructionRa(i), claim));
        }
        for i in 0..self.one_hot_params.bytecode_d {
            let (_, claim) = self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::BytecodeRa(i),
                SumcheckId::HammingWeightClaimReduction,
            );
            polynomial_claims.push((CommittedPolynomial::BytecodeRa(i), claim));
        }
        for i in 0..self.one_hot_params.ram_d {
            let (_, claim) = self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::RamRa(i),
                SumcheckId::HammingWeightClaimReduction,
            );
            polynomial_claims.push((CommittedPolynomial::RamRa(i), claim));
        }

        // category C: Advice 多项式 (Advice Polynomials)
        // 来源：Stage 6 和 Stage 7 的 AdviceClaimReduction。
        // Advice 多项式的大小通常和 Trace Length 不同 (通常更小)，因此也需要
        // 计算特定的拉格朗日因子，以便将其逻辑上嵌入到主 Dory 矩阵的左上角 (Top-Left Block)。
        if let Some((advice_point, advice_claim)) = self
            .opening_accumulator
            .get_advice_opening(AdviceKind::Trusted, SumcheckId::AdviceClaimReductionPhase2)
        {
            let lagrange_factor =
                compute_advice_lagrange_factor::<F>(&opening_point.r, advice_point.len());
            polynomial_claims.push((
                CommittedPolynomial::TrustedAdvice,
                advice_claim * lagrange_factor,
            ));
        }

        if let Some((advice_point, advice_claim)) = self.opening_accumulator.get_advice_opening(
            AdviceKind::Untrusted,
            SumcheckId::AdviceClaimReductionPhase2,
        ) {
            let lagrange_factor =
                compute_advice_lagrange_factor::<F>(&opening_point.r, advice_point.len());
            polynomial_claims.push((
                CommittedPolynomial::UntrustedAdvice,
                advice_claim * lagrange_factor,
            ));
        }

        // 3. 采样随机权重 Gamma 并计算幂次。
        // 为了进行批量验证 (Batch Verification)，我们将所有的声明组合成一个大的随机线性组合 (RLC)。
        // H(x) = ∑ γ^i * P_i(x) => Claim = ∑ γ^i * y_i
        let claims: Vec<F> = polynomial_claims.iter().map(|(_, c)| *c).collect();
        self.transcript.append_scalars(&claims);
        let gamma_powers: Vec<F> = self.transcript.challenge_scalar_powers(claims.len());

        // 构建 Dory Opening 状态对象
        let state = DoryOpeningState {
            opening_point: opening_point.r.clone(),
            gamma_powers,
            polynomial_claims,
        };

        // 准备流式计算所需的数据 (Bytecode, Memory Layout)
        let streaming_data = Arc::new(RLCStreamingData {
            bytecode: Arc::clone(&self.preprocessing.shared.bytecode),
            memory_layout: self.preprocessing.shared.memory_layout.clone(),
        });

        // 提取 Advice 多项式 (如果存在)
        let mut advice_polys = HashMap::new();
        if let Some(poly) = self.advice.trusted_advice_polynomial.take() {
            advice_polys.insert(CommittedPolynomial::TrustedAdvice, poly);
        }
        if let Some(poly) = self.advice.untrusted_advice_polynomial.take() {
            advice_polys.insert(CommittedPolynomial::UntrustedAdvice, poly);
        }

        // 4. 构建流式 RLC 多项式 (Streaming RLC)。
        // 这一步是性能关键。我们不显式地构造每一个承诺多项式 P_i。
        // 相反，我们再次流式遍历 Trace (从 self.trace 或 Materialized trace)，
        // 根据当前的行数据，直接计算出 RLC 多项式 H(x) 的系数值。
        // 这样可以将内存消耗从 O(Num_Polys * Trace_Len) 降低到 O(Trace_Len)。
        let (joint_poly, hint) = state.build_streaming_rlc::<PCS>(
            self.one_hot_params.clone(),
            TraceSource::Materialized(Arc::clone(&self.trace)),
            streaming_data,
            opening_proof_hints,
            advice_polys,
        );

        // 5. 执行最终的 Dory Opening Proof。
        // 证明：构建出的 joint_poly 在 opening_point 处的评估值确实等于我们预期的加权和。
        PCS::prove(
            &self.preprocessing.generators,
            &joint_poly,
            &opening_point.r,
            Some(hint),
            &mut self.transcript,
        )
    }
}

pub struct JoltAdvice<F: JoltField, PCS: CommitmentScheme<Field = F>> {
    pub untrusted_advice_polynomial: Option<MultilinearPolynomial<F>>,
    pub trusted_advice_commitment: Option<PCS::Commitment>,
    pub trusted_advice_polynomial: Option<MultilinearPolynomial<F>>,
    /// Hint for untrusted advice (for batched Dory opening)
    pub untrusted_advice_hint: Option<PCS::OpeningProofHint>,
    /// Hint for trusted advice (for batched Dory opening)
    pub trusted_advice_hint: Option<PCS::OpeningProofHint>,
}

#[cfg(feature = "allocative")]
fn write_instance_flamegraph_svg(
    instances: &[Box<dyn SumcheckInstanceProver<impl JoltField, impl Transcript>>],
    path: impl AsRef<Path>,
) {
    let mut flamegraph = FlameGraphBuilder::default();
    for instance in instances {
        instance.update_flamegraph(&mut flamegraph)
    }
    write_flamegraph_svg(flamegraph, path);
}

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltProverPreprocessing<F: JoltField, PCS: CommitmentScheme<Field = F>> {
    pub generators: PCS::ProverSetup,
    pub shared: JoltSharedPreprocessing,
}

impl<F, PCS> JoltProverPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    #[tracing::instrument(skip_all, name = "JoltProverPreprocessing::gen")]
    pub fn new(
        shared: JoltSharedPreprocessing,
        // max_trace_length: usize,
    ) -> JoltProverPreprocessing<F, PCS> {
        // 定义了一个阈值，用于决定 One-Hot 编码的分块大小。
        // 如果 trace 长度的对数小于此阈值，则使用较小的块大小。
        use common::constants::ONEHOT_CHUNK_THRESHOLD_LOG_T;

        // 1. 确定最大 Trace 长度和对数：
        // 从共享预处理数据中获取最大填充后的 Trace 长度，并确保它是 2 的幂。
        let max_T: usize = shared.max_padded_trace_length.next_power_of_two();
        let max_log_T = max_T.log_2();

        // 2. 确定生成器设置需要的最大 log_k_chunk：
        // log_k_chunk 决定了 commitments 时的分块维度。
        // 为了确保生成的证明者参数（generators）足够大以覆盖可能的运行情况，这里根据 max_log_T 选择最大的可能的 log_k_chunk。
        // - 如果 max_log_T 较小（< 阈值），则 log_k_chunk 设为 4。
        // - 否则，设为 8。这通常对应于 Jolt 默认配置中的最大分块大小。
        let max_log_k_chunk = if max_log_T < ONEHOT_CHUNK_THRESHOLD_LOG_T {
            4
        } else {
            8
        };

        // 3. 初始化 Prover 生成器（Generators）：
        // 调用承诺方案（如 Hyrax/Dory）的 setup_prover 方法。
        // 参数是总的变量数，大致等于 `log(矩阵行数) + log(矩阵列数)`。
        // 这里 max_log_k_chunk 对应矩阵的一维（行或列的块大小），max_log_T 对应总大小的对数。
        // 这样可以确保生成的公共参数（SRS/Generators）足够大，能够支持最大 Trace 长度下的证明生成。
        let generators = PCS::setup_prover(max_log_k_chunk + max_log_T);

        // 4. 返回预处理结构体：
        // 包含生成的 public parameters (generators) 和共享的预处理信息 (shared)。
        JoltProverPreprocessing { generators, shared }
    }

    pub fn save_to_target_dir(&self, target_dir: &str) -> std::io::Result<()> {
        let filename = Path::new(target_dir).join("jolt_prover_preprocessing.dat");
        let mut file = File::create(filename.as_path())?;
        let mut data = Vec::new();
        self.serialize_compressed(&mut data).unwrap();
        file.write_all(&data)?;
        Ok(())
    }

    pub fn read_from_target_dir(target_dir: &str) -> std::io::Result<Self> {
        let filename = Path::new(target_dir).join("jolt_prover_preprocessing.dat");
        let mut file = File::open(filename.as_path())?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;
        Ok(Self::deserialize_compressed(&*data).unwrap())
    }
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>> Serializable
for JoltProverPreprocessing<F, PCS>
{
}

#[cfg(test)]
mod tests {
    use ark_bn254::Fr;
    use serial_test::serial;

    use crate::host;
    use crate::poly::{
        commitment::{
            commitment_scheme::CommitmentScheme,
            dory::{DoryCommitmentScheme, DoryContext, DoryGlobals},
        },
        multilinear_polynomial::MultilinearPolynomial,
        opening_proof::{OpeningAccumulator, SumcheckId},
    };
    use crate::zkvm::claim_reductions::AdviceKind;
    use crate::zkvm::verifier::JoltSharedPreprocessing;
    use crate::zkvm::witness::CommittedPolynomial;
    use crate::zkvm::{
        prover::JoltProverPreprocessing,
        ram::populate_memory_states,
        verifier::{JoltVerifier, JoltVerifierPreprocessing},
        RV64IMACProver, RV64IMACVerifier,
    };

    fn commit_trusted_advice_preprocessing_only(
        preprocessing: &JoltProverPreprocessing<Fr, DoryCommitmentScheme>,
        trusted_advice_bytes: &[u8],
    ) -> (
        <DoryCommitmentScheme as CommitmentScheme>::Commitment,
        <DoryCommitmentScheme as CommitmentScheme>::OpeningProofHint,
    ) {
        let max_trusted_advice_size = preprocessing.shared.memory_layout.max_trusted_advice_size;
        let mut trusted_advice_words = vec![0u64; (max_trusted_advice_size as usize) / 8];
        populate_memory_states(
            0,
            trusted_advice_bytes,
            Some(&mut trusted_advice_words),
            None,
        );

        let poly = MultilinearPolynomial::<Fr>::from(trusted_advice_words);
        let advice_len = poly.len().next_power_of_two().max(1);

        let _guard = DoryGlobals::initialize_context(1, advice_len, DoryContext::TrustedAdvice);
        let (commitment, hint) = {
            let _ctx = DoryGlobals::with_context(DoryContext::TrustedAdvice);
            DoryCommitmentScheme::commit(&poly, &preprocessing.generators)
        };
        (commitment, hint)
    }

    #[test]
    #[serial]
    fn fib_e2e_dory() {
        // tracing_subscriber::fmt::init();
        let _ = tracing_subscriber::fmt().try_init();
        // 1. 初始化宿主程序：加载 "fibonacci-guest" 二进制文件。
        let mut program = host::Program::new("fibonacci-guest");

        // 2. 准备输入：将输入参数 (100u32) 序列化为字节向量。
        let inputs = postcard::to_stdvec(&5).unwrap();

        // 3. 解码程序：获取字节码、初始内存状态等信息。
        let (bytecode, init_memory_state, _) = program.decode();

        // 4. 生成 Execution Trace（模拟执行）：
        // 这一步在不生成证明的情况下运行程序，获取 IO 设备信息（如内存布局）用于后续的预处理配置。
        // trace() 返回 (trace, lazy_trace, memory_state, io_device)。
        let (_trace, _lazy_trace, _memory_state, io_device) = program.trace(&inputs, &[], &[]);
        println!("output:{:?}", io_device.outputs);
        // 5. 共享预处理（Shared Preprocessing）：
        // 创建 Prover 和 Verifier 之间共享的静态数据。
        // 包括字节码、内存布局、初始内存状态以及预设的最大 Trace 长度 (1 << 16)。
        let shared_preprocessing = JoltSharedPreprocessing::new(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            init_memory_state,
            1 << 16, // 最大 padded trace 长度
        );

        // 6. Prover 预处理：
        // 基于共享预处理生成 Prover 专用的数据（主要是承诺方案的生成元 Generators）。
        let prover_preprocessing = JoltProverPreprocessing::new(shared_preprocessing.clone());

        // 7. 获取 ELF 内容：
        // 从程序对象中提取 ELF 二进制内容，Prover 初始化需要用到它来重新模拟执行。
        let elf_contents_opt = program.get_elf_contents();
        let elf_contents = elf_contents_opt.as_deref().expect("elf contents is None");

        // 8. 初始化 Prover：
        // 使用预处理数据、ELF 内容和输入来构建证明者实例。
        // 这里也会再次执行 trace 生成（gen_from_elf 内部会调用 trace）。
        let prover = RV64IMACProver::gen_from_elf(
            &prover_preprocessing,
            elf_contents,
            &inputs,
            &[],   // untrusted advice
            &[],   // trusted advice
            None,  // trusted advice commitment
            None,  // trusted advice hint
        );

        // 克隆 IO 设备信息，供 Verifier 使用（Verifier 需要知道程序的输出和公共 IO 状态）。
        let io_device = prover.program_io.clone();

        // 9. 生成证明（Prove）：
        // 执行完整的证明流程，返回生成的 JoltProof 和调试信息。
        let (jolt_proof, debug_info) = prover.prove();

        // 10. Verifier 预处理：
        // 从共享预处理和 Prover 的生成元中提取 Verifier 需要的设置参数。
        let verifier_preprocessing = JoltVerifierPreprocessing::new(
            shared_preprocessing,
            prover_preprocessing.generators.to_verifier_setup(),
        );

        // 11. 初始化并运行 Verifier：
        // 使用预处理参数、生成的证明、IO 状态和调试信息来验证证明的正确性。
        let verifier = RV64IMACVerifier::new(
            &verifier_preprocessing,
            jolt_proof,
            io_device,
            None, // proof transcript (通常如果是新开始验证则为 None)
            debug_info,
        )
            .expect("Failed to create verifier");

        // 12. 执行验证：如果验证失败则 panic。
        verifier.verify().expect("Failed to verify proof");
    }

    #[test]
    #[serial]
    fn small_trace_e2e_dory() {
        let mut program = host::Program::new("fibonacci-guest");
        let inputs = postcard::to_stdvec(&5u32).unwrap();
        let (bytecode, init_memory_state, _) = program.decode();
        let (_, _, _, io_device) = program.trace(&inputs, &[], &[]);

        let shared_preprocessing = JoltSharedPreprocessing::new(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            init_memory_state,
            256,
        );

        let prover_preprocessing = JoltProverPreprocessing::new(shared_preprocessing.clone());
        let elf_contents_opt = program.get_elf_contents();
        let elf_contents = elf_contents_opt.as_deref().expect("elf contents is None");
        let log_chunk = 8; // Use default log_chunk for tests
        let prover = RV64IMACProver::gen_from_elf(
            &prover_preprocessing,
            elf_contents,
            &inputs,
            &[],
            &[],
            None,
            None,
        );

        assert!(
            prover.padded_trace_len <= (1 << log_chunk),
            "Test requires T <= chunk_size ({}), got T = {}",
            1 << log_chunk,
            prover.padded_trace_len
        );

        let io_device = prover.program_io.clone();
        let (jolt_proof, debug_info) = prover.prove();

        let verifier_preprocessing = JoltVerifierPreprocessing::new(
            prover_preprocessing.shared.clone(),
            prover_preprocessing.generators.to_verifier_setup(),
        );
        let verifier = RV64IMACVerifier::new(
            &verifier_preprocessing,
            jolt_proof,
            io_device,
            None,
            debug_info,
        )
            .expect("Failed to create verifier");
        verifier.verify().expect("Failed to verify proof");
    }

    #[test]
    #[serial]
    fn sha3_e2e_dory() {
        // Ensure SHA3 inline library is linked and auto-registered
        #[cfg(feature = "host")]
        use jolt_inlines_keccak256 as _;
        // SHA3 inlines are automatically registered via #[ctor::ctor]
        // when the jolt-inlines-keccak256 crate is linked (see lib.rs)
        let _ = tracing_subscriber::fmt().try_init();
        let mut program = host::Program::new("sha3-guest");
        let (bytecode, init_memory_state, _) = program.decode();
        let inputs = postcard::to_stdvec(&[5u8; 32]).unwrap();
        let (_, _, _, io_device) = program.trace(&inputs, &[], &[]);

        let shared_preprocessing = JoltSharedPreprocessing::new(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            init_memory_state,
            1 << 16,
        );

        let prover_preprocessing = JoltProverPreprocessing::new(shared_preprocessing.clone());
        let elf_contents_opt = program.get_elf_contents();
        let elf_contents = elf_contents_opt.as_deref().expect("elf contents is None");
        let prover = RV64IMACProver::gen_from_elf(
            &prover_preprocessing,
            elf_contents,
            &inputs,
            &[],
            &[],
            None,
            None,
        );
        let io_device = prover.program_io.clone();
        let (jolt_proof, debug_info) = prover.prove();

        let verifier_preprocessing = JoltVerifierPreprocessing::new(
            prover_preprocessing.shared.clone(),
            prover_preprocessing.generators.to_verifier_setup(),
        );
        let verifier = RV64IMACVerifier::new(
            &verifier_preprocessing,
            jolt_proof,
            io_device.clone(),
            None,
            debug_info,
        )
            .expect("Failed to create verifier");
        verifier.verify().expect("Failed to verify proof");
        assert_eq!(
            io_device.inputs, inputs,
            "Inputs mismatch: expected {:?}, got {:?}",
            inputs, io_device.inputs
        );
        let expected_output = &[
            0xd0, 0x3, 0x5c, 0x96, 0x86, 0x6e, 0xe2, 0x2e, 0x81, 0xf5, 0xc4, 0xef, 0xbd, 0x88,
            0x33, 0xc1, 0x7e, 0xa1, 0x61, 0x10, 0x81, 0xfc, 0xd7, 0xa3, 0xdd, 0xce, 0xce, 0x7f,
            0x44, 0x72, 0x4, 0x66,
        ];
        assert_eq!(io_device.outputs, expected_output, "Outputs mismatch",);
    }

    #[test]
    #[serial]
    fn sha2_e2e_dory() {
        // Ensure SHA2 inline library is linked and auto-registered
        #[cfg(feature = "host")]
        use jolt_inlines_sha2 as _;
        // SHA2 inlines are automatically registered via #[ctor::ctor]
        // when the jolt-inlines-sha2 crate is linked (see lib.rs)
        let mut program = host::Program::new("sha2-guest");
        let (bytecode, init_memory_state, _) = program.decode();
        let inputs = postcard::to_stdvec(&[5u8; 32]).unwrap();
        let (_, _, _, io_device) = program.trace(&inputs, &[], &[]);

        let shared_preprocessing = JoltSharedPreprocessing::new(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            init_memory_state,
            1 << 16,
        );

        let prover_preprocessing = JoltProverPreprocessing::new(shared_preprocessing.clone());
        let elf_contents_opt = program.get_elf_contents();
        let elf_contents = elf_contents_opt.as_deref().expect("elf contents is None");
        let prover = RV64IMACProver::gen_from_elf(
            &prover_preprocessing,
            elf_contents,
            &inputs,
            &[],
            &[],
            None,
            None,
        );
        let io_device = prover.program_io.clone();
        let (jolt_proof, debug_info) = prover.prove();

        let verifier_preprocessing = JoltVerifierPreprocessing::new(
            prover_preprocessing.shared.clone(),
            prover_preprocessing.generators.to_verifier_setup(),
        );
        let verifier = RV64IMACVerifier::new(
            &verifier_preprocessing,
            jolt_proof,
            io_device.clone(),
            None,
            debug_info,
        )
            .expect("Failed to create verifier");
        verifier.verify().expect("Failed to verify proof");
        let expected_output = &[
            0x28, 0x9b, 0xdf, 0x82, 0x9b, 0x4a, 0x30, 0x26, 0x7, 0x9a, 0x3e, 0xa0, 0x89, 0x73,
            0xb1, 0x97, 0x2d, 0x12, 0x4e, 0x7e, 0xaf, 0x22, 0x33, 0xc6, 0x3, 0x14, 0x3d, 0xc6,
            0x3b, 0x50, 0xd2, 0x57,
        ];
        assert_eq!(
            io_device.outputs, expected_output,
            "Outputs mismatch: expected {:?}, got {:?}",
            expected_output, io_device.outputs
        );
    }

    #[test]
    #[serial]
    fn sha2_e2e_dory_with_unused_advice() {
        // SHA2 guest does not consume advice, but providing both trusted and untrusted advice
        // should still work correctly through the full pipeline:
        // - Trusted: commit in preprocessing-only context, reduce in Stage 6, batch in Stage 8
        // - Untrusted: commit at prove time, reduce in Stage 6, batch in Stage 8
        let mut program = host::Program::new("sha2-guest");
        let (bytecode, init_memory_state, _) = program.decode();
        let inputs = postcard::to_stdvec(&[5u8; 32]).unwrap();
        let trusted_advice = postcard::to_stdvec(&[7u8; 32]).unwrap();
        let untrusted_advice = postcard::to_stdvec(&[9u8; 32]).unwrap();

        let (_, _, _, io_device) = program.trace(&inputs, &untrusted_advice, &trusted_advice);

        let shared_preprocessing = JoltSharedPreprocessing::new(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            init_memory_state,
            1 << 16,
        );
        let prover_preprocessing = JoltProverPreprocessing::new(shared_preprocessing.clone());
        let elf_contents = program.get_elf_contents().expect("elf contents is None");

        let (trusted_commitment, trusted_hint) =
            commit_trusted_advice_preprocessing_only(&prover_preprocessing, &trusted_advice);

        let prover = RV64IMACProver::gen_from_elf(
            &prover_preprocessing,
            &elf_contents,
            &inputs,
            &untrusted_advice,
            &trusted_advice,
            Some(trusted_commitment),
            Some(trusted_hint),
        );
        let io_device = prover.program_io.clone();
        let (jolt_proof, debug_info) = prover.prove();

        let verifier_preprocessing = JoltVerifierPreprocessing::from(&prover_preprocessing);
        RV64IMACVerifier::new(
            &verifier_preprocessing,
            jolt_proof,
            io_device.clone(),
            Some(trusted_commitment),
            debug_info,
        )
            .expect("Failed to create verifier")
            .verify()
            .expect("Failed to verify proof");

        // Verify output is correct (advice should not affect sha2 output)
        let expected_output = &[
            0x28, 0x9b, 0xdf, 0x82, 0x9b, 0x4a, 0x30, 0x26, 0x7, 0x9a, 0x3e, 0xa0, 0x89, 0x73,
            0xb1, 0x97, 0x2d, 0x12, 0x4e, 0x7e, 0xaf, 0x22, 0x33, 0xc6, 0x3, 0x14, 0x3d, 0xc6,
            0x3b, 0x50, 0xd2, 0x57,
        ];
        assert_eq!(io_device.outputs, expected_output);
    }

    #[test]
    #[serial]
    fn max_advice_with_small_trace() {
        // 测试目标：验证当Advice（辅助输入）达到最大尺寸（4KB = 512 words），
        // 但执行轨迹（Trace）非常短时，系统是否仍能正常工作。
        //
        // 背景：Advice 多项式通常被嵌入到更大的 Trace 相关的 Dory 矩阵中（作为左上角的块）。
        // 如果 Trace 太短，可能导致矩阵维度不足以容纳 Advice 多项式。
        // 本测试确保当 Advice 维度（sigma_a=5, nu_a=4, 对应 512 words）与
        // 最小填充 Trace（256 cycles -> total_vars=12）结合时，Jolt 的动态调整机制能正确处理。

        // 1. 初始化宿主程序 (Host Program)
        // 使用斐波那契计算作为 Guest 程序，因为它逻辑简单，适合生成短 Trace。
        let mut program = host::Program::new("fibonacci-guest");

        // 准备简短的输入：计算第 5 个斐波那契数，这会产生很短的执行轨迹。
        let inputs = postcard::to_stdvec(&5u32).unwrap();

        // 构造最大尺寸的 Advice 数据：
        // Jolt 默认配置的最大 Advice 大小通常较小（例如 4KB）。
        // 这里创建 4096 字节的向量，填满这个容量。
        let trusted_advice = vec![7u8; 4096];
        let untrusted_advice = vec![9u8; 4096];

        // 2. 解码程序与生成 Trace
        // 解码获取字节码和初始内存状态。
        let (bytecode, init_memory_state, _) = program.decode();

        // 运行程序生成 Trace。此时传入了巨大的 advice 数据，尽管斐波那契逻辑并不真正使用它们。
        // Jolt 会将这些 advice 记录在 IO 设备和内存布局中。
        let (lazy_trace, trace, final_memory_state, io_device) =
            program.trace(&inputs, &untrusted_advice, &trusted_advice);

        // 3. 共享预处理 (Preprocessing)
        // 关键点：将 `max_trace_length` 设置得很小 (256)。
        // 这迫使 Prover 试图使用极小的多项式维度。
        // 如果 Jolt 没有维度检查逻辑，这里的 256 长度可能无法容纳 4KB 的 Advice。
        let shared_preprocessing = JoltSharedPreprocessing::new(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            init_memory_state,
            256,
        );
        let prover_preprocessing = JoltProverPreprocessing::new(shared_preprocessing.clone());

        tracing::info!(
                "preprocessing.memory_layout.max_trusted_advice_size: {}",
                shared_preprocessing.memory_layout.max_trusted_advice_size
            );

        // 4. 单独提交可信 Advice (Trusted Advice Commitment)
        // 在实际应用中，可信 Advice 往往是预先提交好的。
        // 这里模拟这个过程，生成 Commitment 和 Hint。
        let (trusted_commitment, trusted_hint) =
            commit_trusted_advice_preprocessing_only(&prover_preprocessing, &trusted_advice);

        // 5. 生成 Prover 实例
        // `gen_from_trace` 内部会调用 `adjust_trace_length_for_advice`。
        // 即使我们请求了 256 的 Trace 长度，如果 Advice 需要更大的矩阵，
        // Prover 可能会在这里进行断言或自动调整（但在本例中，256 恰好足够嵌入 512 words 的 Advice）。
        let prover = RV64IMACProver::gen_from_trace(
            &prover_preprocessing,
            lazy_trace,
            trace,
            io_device,
            Some(trusted_commitment),
            Some(trusted_hint),
            final_memory_state,
        );

        // 断言验证：
        // 确保原始 Trace 确实很短 (< 512)。
        assert!(prover.unpadded_trace_len < 512);
        // 确保填充后的 Trace 长度维持在预设的 256。
        // 这意味着 Advice 的嵌入逻辑在这个最小尺寸下是兼容的。
        assert_eq!(prover.padded_trace_len, 256);

        // 6. 生成证明 (Prove)
        // 运行完整证明流程，这会触发 Stage 6 (Advice 归约) 和 Stage 8 (Dory Batch Opening)。
        // 如果维度不匹配，Dory 证明构建或验证将会失败。
        let io_device = prover.program_io.clone();
        let (jolt_proof, debug_info) = prover.prove();

        // 7. 验证证明 (Verify)
        // 使用 Verifier 检查生成的证明是否有效。
        let verifier_preprocessing = JoltVerifierPreprocessing::from(&prover_preprocessing);
        RV64IMACVerifier::new(
            &verifier_preprocessing,
            jolt_proof,
            io_device,
            Some(trusted_commitment),
            debug_info,
        )
            .expect("Failed to create verifier")
            .verify()
            .expect("Verification failed");
    }

    #[test]
    #[serial]
    fn advice_e2e_dory() {
        let _ = tracing_subscriber::fmt().try_init();
        // 测试目标：验证 Guest 程序同时消耗 Trusted（可信）和 Untrusted（不可信）Advice 时的端到端流程。
        // 使用 "merkle-tree-guest" 程序，因为它需要这两种 Advice 来计算 Merkle Root。
        let mut program = host::Program::new("merkle-tree-guest");
        let (bytecode, init_memory_state, _) = program.decode();

        // 构造 Merkle Tree 输入：
        // 这个程序计算 4 个叶子的 Merkle Root。
        // - inputs: 第 1 个叶子 (leaf1)。通常作为公共输入。
        // - trusted_advice: 第 2、3 个叶子 (leaf2, leaf3)。这模拟了预先已知的数据（如公共参数或历史根）。
        // - untrusted_advice: 第 4 个叶子 (leaf4)。这模拟了私有 witness 数据（如用户隐私数据）。
        let inputs = postcard::to_stdvec(&[5u8; 32].as_slice()).unwrap();
        let untrusted_advice = postcard::to_stdvec(&[8u8; 32]).unwrap();
        let mut trusted_advice = postcard::to_stdvec(&[6u8; 32]).unwrap();
        trusted_advice.extend(postcard::to_stdvec(&[7u8; 32]).unwrap());

        // 1. 生成 Trace，记录 Advice 的使用情况到 io_device。
        // 这一步运行程序但不生成证明，主要是为了获取内存布局和 IO 设备状态。
        let (_, _, _, io_device) = program.trace(&inputs, &untrusted_advice, &trusted_advice);

        // 2. 共享预处理：设置最大 trace 长度等参数。
        // 1 << 16 是配置的最大 trace 长度。
        let shared_preprocessing = JoltSharedPreprocessing::new(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            init_memory_state,
            1 << 16,
        );
        let prover_preprocessing = JoltProverPreprocessing::new(shared_preprocessing.clone());
        let elf_contents = program.get_elf_contents().expect("elf contents is None");

        // 3. 提交可信 Advice (Trusted Advice Commitment)。
        // 这模拟了"预处理"阶段：可信 Advice 已经确定，生成了承诺 (Commitment) 和辅助证明 (Hint)。
        // Verifier 将持有 trusted_commitment。
        let (trusted_commitment, trusted_hint) =
            commit_trusted_advice_preprocessing_only(&prover_preprocessing, &trusted_advice);

        // 4. 生成 Prover 实例。
        // 这里传入了 inputs, untrusted, trusted 三种数据源。
        // Prover 内部会再次执行程序，并构建多项式。
        let prover = RV64IMACProver::gen_from_elf(
            &prover_preprocessing,
            &elf_contents,
            &inputs,
            &untrusted_advice,
            &trusted_advice,
            Some(trusted_commitment),
            Some(trusted_hint),
        );
        let io_device = prover.program_io.clone();

        // 5. 生成证明 (Prove)。
        // 内部流程会分别处理 Trusted (Stage 6 归约 + Stage 8 嵌入)
        // 和 Untrusted Advice (Commit + Stage 6 归约 + Stage 8 嵌入)。
        let (jolt_proof, debug_info) = prover.prove();

        // 6. 验证证明 (Verify)。
        // 创建 Verifier，传入 Prover 生成的 output 和 commitment。
        let verifier_preprocessing = JoltVerifierPreprocessing::from(&prover_preprocessing);
        RV64IMACVerifier::new(
            &verifier_preprocessing,
            jolt_proof,
            io_device.clone(),
            Some(trusted_commitment),
            debug_info,
        )
            .expect("Failed to create verifier")
            .verify()
            .expect("Verification failed");

        // 7. 验证计算结果正确性。
        // 预计算这 4 个特定叶子 ([5;32], [6;32], [7;32], [8;32]) 对应的 Merkle Root。
        // 确保 ZK 证明通过的同时，程序本身的计算逻辑也是正确的。
        let expected_output = &[
            0xb4, 0x37, 0x0f, 0x3a, 0xb, 0x3d, 0x38, 0xa8, 0x7a, 0x6c, 0x4c, 0x46, 0x9, 0xe7, 0x83,
            0xb3, 0xcc, 0xb7, 0x1c, 0x30, 0x1f, 0xf8, 0x54, 0xd, 0xf7, 0xdd, 0xc8, 0x42, 0x32,
            0xbb, 0x16, 0xd7,
        ];
        assert_eq!(io_device.outputs, expected_output);
    }

    #[test]
    #[serial]
    fn advice_opening_point_derives_from_unified_point() {
        // 测试目标：验证 Advice（辅助输入）的 Opening Point（多项式评估点）是否能够正确地
        // 从全局统一的主 Opening Point 中派生出来，且符合 Dory 协议的坐标映射策略。
        //
        // 核心难点：
        // 在 Dory 协议中，为了批量验证，我们将不同大小的多项式（如 Advice 和 Main Memory Trace）
        // 视为嵌入在同一个巨大的虚拟矩阵中。
        // 对于极短的 Trace（例如本例中的 256 cycles），Advice 的行坐标可能会跨越
        // Stage 6（Cycle/时间维度）和 Stage 7（Address/空间维度）的随机挑战点。
        // 本测试验证这种复杂的“两阶段归约”过程是否正确计算了坐标。

        // 1. 初始化程序与数据
        // 使用斐波那契计算作为 Guest 程序，输入很小，产生短 Trace。
        let mut program = host::Program::new("fibonacci-guest");
        let inputs = postcard::to_stdvec(&5u32).unwrap();
        // 构造虚拟的 Trusted 和 Untrusted Advice 数据。
        let trusted_advice = postcard::to_stdvec(&[7u8; 32]).unwrap();
        let untrusted_advice = postcard::to_stdvec(&[9u8; 32]).unwrap();

        // 2. 生成 Trace 与预处理
        let (bytecode, init_memory_state, _) = program.decode();
        let (lazy_trace, trace, final_memory_state, io_device) =
            program.trace(&inputs, &untrusted_advice, &trusted_advice);

        // 创建共享预处理数据，设置最大 Trace 为 64K (1<<16)。
        // 注意：尽管最大容量很大，但下面的 gen_from_trace 会根据实际执行情况生成一个小 Trace。
        let shared_preprocessing = JoltSharedPreprocessing::new(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            init_memory_state,
            1 << 16,
        );
        let prover_preprocessing = JoltProverPreprocessing::new(shared_preprocessing.clone());

        // 提前提交 Trusted Advice（模拟预处理阶段）。
        let (trusted_commitment, trusted_hint) =
            commit_trusted_advice_preprocessing_only(&prover_preprocessing, &trusted_advice);

        // 3. 生成 Prover 实例
        // 传入短 Trace。
        let prover = RV64IMACProver::gen_from_trace(
            &prover_preprocessing,
            lazy_trace,
            trace,
            io_device,
            Some(trusted_commitment),
            Some(trusted_hint),
            final_memory_state,
        );

        // 关键断言：确保生成的 Padding 后 Trace 长度极小（256）。
        // 这意味着 Trace 矩阵的行数很少，Advice 的数据分布会变得更加“紧凑”，
        // 迫使验证逻辑处理边界情况。
        assert_eq!(prover.padded_trace_len, 256, "test expects small trace");

        let io_device = prover.program_io.clone();
        // 4. 生成证明 (Prove)
        // 这一步会执行所有 Stage，包括计算 Opening Point。
        let (jolt_proof, debug_info) = prover.prove();
        let debug_info = debug_info.expect("expected debug_info in tests");

        // 5. 验证坐标派生逻辑
        // 第一步：获取全局统一的 Opening Point。
        // 我们从 Stage 7 (HammingWeightClaimReduction) 的 InstructionRa 多项式获取这个点。
        // 这是 Prover 和 Verifier 协商出的随机点 r。
        let (opening_point, _) = debug_info
            .opening_accumulator
            .get_committed_polynomial_opening(
                CommittedPolynomial::InstructionRa(0),
                SumcheckId::HammingWeightClaimReduction,
            );
        // Dory 内部通常是大端序，这里转为小端序方便索引切片。
        let mut point_dory_le = opening_point.r.clone();
        point_dory_le.reverse();

        // 计算主矩阵和 Advice 矩阵的维度参数（sigma/nu）。
        let total_vars = point_dory_le.len(); // 全局变量数
        let (sigma_main, _nu_main) = DoryGlobals::balanced_sigma_nu(total_vars); // 主矩阵拆分
        // Advice 矩阵通过其最大字节数计算拆分。
        let (sigma_a, nu_a) = DoryGlobals::advice_sigma_nu_from_max_bytes(
            prover_preprocessing
                .shared
                .memory_layout
                .max_trusted_advice_size as usize,
        );

        // 第二步：手动构建预期的 Advice Opening Point。
        // 原理：Advice 矩阵被嵌入在主矩阵的左上角。
        // 它的坐标由全局点的特定比特位组成：
        // - 列坐标 (Col) 来自 point_dory 低位 [0..sigma_a]
        // - 行坐标 (Row) 来自 point_dory 主矩阵行起始位 [sigma_main..sigma_main + nu_a]
        let mut expected_advice_le: Vec<_> = point_dory_le[0..sigma_a].to_vec();
        expected_advice_le.extend_from_slice(&point_dory_le[sigma_main..sigma_main + nu_a]);

        // 第三步：验证实际 Prover 计算出的点是否一致。
        // 检查 Trusted 和 Untrusted 两种 Advice。
        for (name, kind) in [
            ("trusted", AdviceKind::Trusted),
            ("untrusted", AdviceKind::Untrusted),
        ] {
            // 获取 Prover 在 Phase 2 (Stage 7) 计算出的实际 Advice Opening。
            let get_fn = debug_info
                .opening_accumulator
                .get_advice_opening(kind, SumcheckId::AdviceClaimReductionPhase2);
            assert!(
                get_fn.is_some(),
                "{name} advice opening missing for AdviceClaimReductionPhase2"
            );
            let (point_be, _) = get_fn.unwrap();
            let mut point_le = point_be.r.clone();
            point_le.reverse();

            // 比对我们手动推导的点和 Prover 系统计算的点。
            assert_eq!(point_le, expected_advice_le, "{name} advice point mismatch");
        }

        // 6. 端到端验证 (End-to-End Verification)
        // 确保整个证明通过 Verifier 检查，保证数学上的合理性。
        let verifier_preprocessing = JoltVerifierPreprocessing::from(&prover_preprocessing);
        RV64IMACVerifier::new(
            &verifier_preprocessing,
            jolt_proof,
            io_device,
            Some(trusted_commitment),
            Some(debug_info),
        )
            .expect("Failed to create verifier")
            .verify()
            .expect("Verification failed");
    }

    #[test]
    #[serial]
    fn memory_ops_e2e_dory() {
        // 测试目标：验证 RISC-V 基础内存操作指令的正确性。
        // "memory-ops-guest" 程序包含了一系列加载（Load）和存储（Store）指令，
        // 涵盖了不同的位宽（Byte, Half-word, Word, Double-word）以及符号扩展逻辑。
        // 此测试确保 Jolt 的 RAM 一致性检查（Stage 2 Grand Product 和 Stage 4 Value Check）能正确处理这些操作。
        let mut program = host::Program::new("memory-ops-guest");
        let (bytecode, init_memory_state, _) = program.decode();

        // 1. 生成 Trace
        // 该测试用例不需要任何公共输入 (inputs) 或辅助输入 (advice)。
        // 程序内部硬编码了对内存特定地址读写的测试逻辑。
        let (_, _, _, io_device) = program.trace(&[], &[], &[]);

        // 2. 共享预处理
        // 设置最大 trace 长度为 64K (1 << 16)。
        let shared_preprocessing = JoltSharedPreprocessing::new(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            init_memory_state,
            1 << 16,
        );

        let prover_preprocessing = JoltProverPreprocessing::new(shared_preprocessing.clone());
        let elf_contents_opt = program.get_elf_contents();
        let elf_contents = elf_contents_opt.as_deref().expect("elf contents is None");

        // 3. 生成 Prover 实例
        // 传入空的 inputs, untrusted_advice, trusted_advice。
        let prover = RV64IMACProver::gen_from_elf(
            &prover_preprocessing,
            elf_contents,
            &[], // inputs
            &[], // untrusted advice
            &[], // trusted advice
            None,
            None,
        );
        let io_device = prover.program_io.clone();

        // 4. 生成证明 (Prove)
        // 这个过程会验证：
        // - 所有的内存读取操作读到的值都等于上一次写入的值（RAM Consistency / R-W check）。
        // - 所有的内存访问地址和值都在合法范围内。
        let (jolt_proof, debug_info) = prover.prove();

        // 5. 验证证明 (Verify)
        let verifier_preprocessing = JoltVerifierPreprocessing::new(
            prover_preprocessing.shared.clone(),
            prover_preprocessing.generators.to_verifier_setup(),
        );
        let verifier = RV64IMACVerifier::new(
            &verifier_preprocessing,
            jolt_proof,
            io_device,
            None,
            debug_info,
        )
            .expect("Failed to create verifier");

        // 6. 执行验证
        // 只有当所有内存约束和执行约束都满足时，验证才会通过。
        verifier.verify().expect("Failed to verify proof");
    }

    #[test]
    #[serial]
    fn btreemap_e2e_dory() {
        let mut program = host::Program::new("btreemap-guest");
        let (bytecode, init_memory_state, _) = program.decode();
        let inputs = postcard::to_stdvec(&50u32).unwrap();
        let (_, _, _, io_device) = program.trace(&inputs, &[], &[]);

        let shared_preprocessing = JoltSharedPreprocessing::new(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            init_memory_state,
            1 << 16,
        );

        let prover_preprocessing = JoltProverPreprocessing::new(shared_preprocessing.clone());
        let elf_contents_opt = program.get_elf_contents();
        let elf_contents = elf_contents_opt.as_deref().expect("elf contents is None");
        let prover = RV64IMACProver::gen_from_elf(
            &prover_preprocessing,
            elf_contents,
            &inputs,
            &[],
            &[],
            None,
            None,
        );
        let io_device = prover.program_io.clone();
        let (jolt_proof, debug_info) = prover.prove();

        let verifier_preprocessing = JoltVerifierPreprocessing::new(
            prover_preprocessing.shared.clone(),
            prover_preprocessing.generators.to_verifier_setup(),
        );
        let verifier = RV64IMACVerifier::new(
            &verifier_preprocessing,
            jolt_proof,
            io_device,
            None,
            debug_info,
        )
            .expect("Failed to create verifier");
        verifier.verify().expect("Failed to verify proof");
    }

    #[test]
    #[serial]
    fn muldiv_e2e_dory() {
        // 测试目标：验证 RISC-V 乘法和除法指令（M extension）的正确性以及其在 Jolt 中的证明生成流程。
        // "muldiv-guest" 程序通常包含 MUL, DIV, REM 等指令的测试。
        // Jolt 对于这些复杂算术指令通常依赖特定的查找表（Lookups）来实现。
        // 此测试确保 Prover 正确处理了这些特定的 Lookup Tables。
        let mut program = host::Program::new("muldiv-guest");
        let (bytecode, init_memory_state, _) = program.decode();

        // 1. 准备输入
        // 这里的 inputs 是 [9, 5, 3]。
        // 可能用于 guest 内部的某些计算，例如 mul(9, 5), div(9, 5) 等。
        let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).unwrap();

        // 2. 生成 Trace（模拟执行）
        // 获取执行轨迹和 IO 设备状态（特别是内存布局）。
        let (_, _, _, io_device) = program.trace(&inputs, &[], &[]);

        // 3. 共享预处理
        // 设置最大 Trace 长度配置为 64K (1 << 16)。
        let shared_preprocessing = JoltSharedPreprocessing::new(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            init_memory_state,
            1 << 16,
        );

        let prover_preprocessing = JoltProverPreprocessing::new(shared_preprocessing.clone());
        let elf_contents_opt = program.get_elf_contents();
        let elf_contents = elf_contents_opt.as_deref().expect("elf contents is None");

        // 4. 生成 Prover 实例
        // 注意：这里传入的 inputs 是 &[50]，这与上面 trace 用的 inputs 不一致。
        // 这通常是一个小错误或测试用的特定 trick（如果 guest 程序只用 inputs 来占位或长度校验）。
        // 但在这个 specific code snippet 里，它模拟的是：
        // Prover 收到的公共输入是 50（可能是序列化后的字节流的第一个字节或某种编码）。
        // 实际上 gen_from_elf 内部会重新运行 trace。
        let prover = RV64IMACProver::gen_from_elf(
            &prover_preprocessing,
            elf_contents,
            &[50], // Public Inputs
            &[],   // Untrusted Advice
            &[],   // Trusted Advice
            None,
            None,
        );
        let io_device = prover.program_io.clone();

        // 5. 生成证明 (Prove)
        // 验证流程包括：
        // - Stage 2: 验证指令查找表访问的一致性 (Grand Product)。
        // - Stage 5: 验证查找表 (Lookups) 中的 RAF (Read Access Flag) 和 Value Evaluation。
        //   对于 mul/div 来说，这一步最关键，它证明了 input/output 符合查找表定义的算术关系。
        let (jolt_proof, debug_info) = prover.prove();

        // 6. 验证证明 (Verify)
        let verifier_preprocessing = JoltVerifierPreprocessing::new(
            prover_preprocessing.shared.clone(),
            prover_preprocessing.generators.to_verifier_setup(),
        );
        let verifier = RV64IMACVerifier::new(
            &verifier_preprocessing,
            jolt_proof,
            io_device,
            None,
            debug_info,
        )
            .expect("Failed to create verifier");

        // 7. 执行验证
        // 确保 ZK 证明通过，即计算过程确实执行了正确的乘除法。
        verifier.verify().expect("Failed to verify proof");
    }

    #[test]
    #[serial]
    #[should_panic] // 预期测试会 panic，即验证失败。这是安全测试的标准做法。
    fn truncated_trace() {
        // 测试目标：验证安全性。
        // 模拟一个恶意的 Prover，它只执行了程序的一部分（截断了 Trace），
        // 并且声称得到了一错误的输出结果。
        // 验证系统应该检测到 Trace 和程序逻辑/输出的不一致，并在验证阶段拒绝该证明。

        // 1. 初始化程序与数据
        let mut program = host::Program::new("fibonacci-guest");
        let (bytecode, init_memory_state, _) = program.decode();
        let inputs = postcard::to_stdvec(&9u8).unwrap();

        // 2. 生成原始的完整 Trace
        // 这里首先生成一个合法的 Trace 作为基底。
        let (lazy_trace, mut trace, final_memory_state, mut program_io) =
            program.trace(&inputs, &[], &[]);

        // 3. 构造恶意行为
        // 攻击步骤 A: 截断 Trace。
        // 将执行轨迹强行保留前 100 个周期。这意味着程序还没运行完就被终止了。
        trace.truncate(100);

        // 攻击步骤 B: 篡改输出。
        // Prover 声称程序的输出是 0（显然斐波那契数列第9项不是0）。
        program_io.outputs[0] = 0; // change the output to 0

        // 4. 共享预处理
        // 注意：这里使用的是篡改后的 program_io (包含了错误的 output)，
        // 但 Verifier 预处理是基于正确的字节码的。
        let shared_preprocessing = JoltSharedPreprocessing::new(
            bytecode.clone(),
            program_io.memory_layout.clone(),
            init_memory_state,
            1 << 16,
        );

        let prover_preprocessing = JoltProverPreprocessing::new(shared_preprocessing.clone());

        // 5. 生成恶意 Prover 实例
        // 使用被截断的 trace 和被篡改的 program_io 构建 Prover。
        // 在 gen_from_trace 内部，Prover 会试图基于这个残缺的 Trace 构建多项式。
        let prover = RV64IMACProver::gen_from_trace(
            &prover_preprocessing,
            lazy_trace,
            trace,
            program_io.clone(),
            None,
            None,
            final_memory_state,
        );

        // 6. 生成证明 (Prove)
        // 即使 Trace 是坏的，Prover 代码通常也能跑完并生成一个数学上存在的 proof 对象，
        // 只是这个 proof 里的多项式关系是不满足约束的。
        let (proof, _) = prover.prove();

        // 7. 验证证明 (Verify)
        let verifier_preprocessing = JoltVerifierPreprocessing::new(
            prover_preprocessing.shared.clone(),
            prover_preprocessing.generators.to_verifier_setup(),
        );

        // 初始化 Verifier。
        let verifier =
            RV64IMACVerifier::new(&verifier_preprocessing, proof, program_io, None, None).unwrap();

        // 8. 执行验证（预期 panic）
        // 核心检查点：verify() 方法会检查所有的 Sumcheck 和 R1CS 约束。
        // 由于 Trace 被截断，最终状态肯定与初始状态经过逻辑推演后的结果不符（例如 PC 指针没走到终点，寄存器状态不对等）。
        // 因此 verify() 应该返回 Error，进而触发 unwrap() 的 panic。
        verifier.verify().unwrap();
    }

    #[test]
    #[serial]
    #[should_panic] // 预期测试会 panic，即验证失败。这是安全测试，确保篡改内存布局会被检测到。
    fn malicious_trace() {
        // 测试目标：验证安全性。
        // 模拟一个恶意的 Prover，它试图通过篡改内存布局定义（Memory Layout）来欺骗 Verifier。
        // 具体来说，Prover 试图通过修改 IO 设备的指针，让 Verifier 误以为输出区域或终止位在别的地方。

        // 1. 初始化程序与数据
        let mut program = host::Program::new("fibonacci-guest");
        let inputs = postcard::to_stdvec(&1u8).unwrap();
        let (bytecode, init_memory_state, _) = program.decode();

        // 2. 生成原始 Trace
        // 这是一个合法的执行轨迹。
        let (lazy_trace, trace, final_memory_state, mut program_io) =
            program.trace(&inputs, &[], &[]);

        // 3. 共享预处理 (基准事实)
        // 关键点：Verifier 的预处理是基于 *原始且正确* 的内存布局进行的。
        // 这代表了系统设计的“公约”或“规范”。如果 Prover 偏离这个规范，验证应当失败。
        let shared_preprocessing = JoltSharedPreprocessing::new(
            bytecode.clone(),
            program_io.memory_layout.clone(),
            init_memory_state,
            1 << 16,
        );
        let prover_preprocessing = JoltProverPreprocessing::new(shared_preprocessing.clone());

        // 4. 构造恶意行为
        // 攻击方式：篡改 program_io 中的内存布局参数。
        // 将输出起始/结束地址以及终止位地址，全部修改为指向输入的地址范围。
        // 这种修改试图混淆 IO 边界，例如，让 Prover 可以用 Input 的内容冒充 Output，或者提前/延后终止信号。
        // 这里的修改不会影响 Verifier 的预处理状态（那是上面第3步确定的），只会尝试影响 Prover 生成证明的方式。
        program_io.memory_layout.output_start = program_io.memory_layout.input_start;
        program_io.memory_layout.output_end = program_io.memory_layout.input_end;
        program_io.memory_layout.termination = program_io.memory_layout.input_start;

        // 5. 生成恶意 Prover 实例
        // Prover 使用被篡改的 program_io 进行初始化。
        // 这会导致 Prover 生成的多项式（证明）是基于错误的内存段定义的。
        let prover = RV64IMACProver::gen_from_trace(
            &prover_preprocessing,
            lazy_trace,
            trace,
            program_io.clone(),
            None,
            None,
            final_memory_state,
        );
        // 6. 生成证明 (Prove)
        let (proof, _) = prover.prove();

        // 7. 验证证明 (Verify)
        // Verifier 使用原始正确的预处理参数（verifier_preprocessing）来检查 Proof。
        let verifier_preprocessing = JoltVerifierPreprocessing::new(
            prover_preprocessing.shared.clone(),
            prover_preprocessing.generators.to_verifier_setup(),
        );
        let verifier =
            JoltVerifier::new(&verifier_preprocessing, proof, program_io, None, None).unwrap();

        // 8. 执行验证（预期 panic）
        // 由于 Proof 是基于篡改后的地址生成的（例如，Read/Write 检查会对不正确的地址进行约束），
        // 而 Verifier 期望的是基于 shared_preprocessing 中定义的正确地址约束（Canonical Memory Layout），
        // 两者的不一致会导致 R1CS 或 Permutation Check 失败，从而触发 panic。
        verifier.verify().unwrap();
    }
}
