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
use tracer::{
    emulator::memory::Memory, instruction::Cycle, ChunksIterator, JoltDevice, LazyTraceIterator,
};

/// Jolt CPU prover for RV64IMAC.
pub struct JoltCpuProver<
    'a,
    F: JoltField,
    PCS: StreamingCommitmentScheme<Field = F>,
    ProofTranscript: Transcript,
> {
    pub preprocessing: &'a JoltProverPreprocessing<F, PCS>,
    pub program_io: JoltDevice,
    pub lazy_trace: LazyTraceIterator,
    pub trace: Arc<Vec<Cycle>>,
    pub advice: JoltAdvice<F, PCS>,
    /// Phase-bridge randomness for two-phase advice claim reduction.
    /// Stored after Stage 6 initialization and reused in Stage 7.
    advice_reduction_gamma_trusted: Option<F>,
    advice_reduction_gamma_untrusted: Option<F>,
    pub unpadded_trace_len: usize,
    pub padded_trace_len: usize,
    pub transcript: ProofTranscript,
    pub opening_accumulator: ProverOpeningAccumulator<F>,
    pub spartan_key: UniformSpartanKey<F>,
    pub initial_ram_state: Vec<u64>,
    pub final_ram_state: Vec<u64>,
    pub one_hot_params: OneHotParams,
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
        for (row_idx, (chunk, row_tier1_commitments)) in zipped_iter {
            tracing::info!(">>> 正在处理第 {} 行 (Row Index)", row_idx);
            tracing::info!("    当前 Chunk 大小: {}", chunk.len());
            // tracing::debug!("polys： {:?}", polys); // 保留原有的 log


            // 此时已移除 par_iter，改为串行处理
            let mut row_results = Vec::with_capacity(polys.len());
            // 4. 内层循环：遍历每承诺一个多项式，即把多个trace的对应列提出出来，计算commitment
            for (poly_idx, poly) in polys.iter().enumerate() {
                // 打印多项式信息
                tracing::info!("    -> 处理第 {} 个多项式: {:?}", poly_idx, poly);

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

            tracing::info!("<<< 第 {} 行处理完毕\n", row_idx);
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

    fn generate_and_commit_untrusted_advice(&mut self) -> Option<PCS::Commitment> {
        if self.program_io.untrusted_advice.is_empty() {
            return None;
        }

        // Commit untrusted advice in its dedicated Dory context, using a preprocessing-only
        // matrix shape derived deterministically from the advice length (balanced dims).

        let mut untrusted_advice_vec =
            vec![0; self.program_io.memory_layout.max_untrusted_advice_size as usize / 8];

        populate_memory_states(
            0,
            &self.program_io.untrusted_advice,
            Some(&mut untrusted_advice_vec),
            None,
        );

        let poly = MultilinearPolynomial::from(untrusted_advice_vec);
        let advice_len = poly.len().next_power_of_two().max(1);

        let _guard = DoryGlobals::initialize_context(1, advice_len, DoryContext::UntrustedAdvice);
        let _ctx = DoryGlobals::with_context(DoryContext::UntrustedAdvice);
        let (commitment, hint) = PCS::commit(&poly, &self.preprocessing.generators);
        self.transcript.append_serializable(&commitment);

        self.advice.untrusted_advice_polynomial = Some(poly);
        self.advice.untrusted_advice_hint = Some(hint);

        Some(commitment)
    }

    fn generate_and_commit_trusted_advice(&mut self) {
        if self.program_io.trusted_advice.is_empty() {
            return;
        }

        let mut trusted_advice_vec =
            vec![0; self.program_io.memory_layout.max_trusted_advice_size as usize / 8];

        populate_memory_states(
            0,
            &self.program_io.trusted_advice,
            Some(&mut trusted_advice_vec),
            None,
        );

        let poly = MultilinearPolynomial::from(trusted_advice_vec);
        self.advice.trusted_advice_polynomial = Some(poly);
        self.transcript
            .append_serializable(self.advice.trusted_advice_commitment.as_ref().unwrap());
    }

    #[tracing::instrument(skip_all)]
    fn prove_stage1(
        &mut self,
    ) -> (
        UniSkipFirstRoundProof<F, ProofTranscript>,
        SumcheckInstanceProof<F, ProofTranscript>,
    ) {
        #[cfg(not(target_arch = "wasm32"))]
        print_current_memory_usage("Stage 1 baseline");

        tracing::info!("Stage 1 proving");
        let uni_skip_params = OuterUniSkipParams::new(&self.spartan_key, &mut self.transcript);
        let mut uni_skip = OuterUniSkipProver::initialize(
            uni_skip_params.clone(),
            &self.trace,
            &self.preprocessing.shared.bytecode,
        );
        let first_round_proof = prove_uniskip_round(
            &mut uni_skip,
            &mut self.opening_accumulator,
            &mut self.transcript,
        );

        // Every sum-check with num_rounds > 1 requires a schedule
        // which dictates the compute_message and bind methods.
        // Using LinearOnlySchedule to benchmark linear-only mode (no streaming).
        // Outer remaining sumcheck has degree 3 (multiquadratic)
        // Number of rounds = tau.len() - 1 (cycle variables only)
        let schedule = LinearOnlySchedule::new(uni_skip_params.tau.len() - 1);
        let shared = OuterSharedState::new(
            Arc::clone(&self.trace),
            &self.preprocessing.shared.bytecode,
            &uni_skip_params,
            &self.opening_accumulator,
        );
        let mut spartan_outer_remaining: OuterRemainingStreamingSumcheck<_, _> =
            OuterRemainingStreamingSumcheck::new(shared, schedule);

        let (sumcheck_proof, _r_stage1) = BatchedSumcheck::prove(
            vec![&mut spartan_outer_remaining],
            &mut self.opening_accumulator,
            &mut self.transcript,
        );

        (first_round_proof, sumcheck_proof)
    }

    #[tracing::instrument(skip_all)]
    fn prove_stage2(
        &mut self,
    ) -> (
        UniSkipFirstRoundProof<F, ProofTranscript>,
        SumcheckInstanceProof<F, ProofTranscript>,
    ) {
        #[cfg(not(target_arch = "wasm32"))]
        print_current_memory_usage("Stage 2 baseline");

        // Stage 2a: Prove univariate-skip first round for product virtualization
        let uni_skip_params =
            ProductVirtualUniSkipParams::new(&self.opening_accumulator, &mut self.transcript);
        let mut uni_skip =
            ProductVirtualUniSkipProver::initialize(uni_skip_params.clone(), &self.trace);
        let first_round_proof = prove_uniskip_round(
            &mut uni_skip,
            &mut self.opening_accumulator,
            &mut self.transcript,
        );

        // Initialization params
        let spartan_product_virtual_remainder_params = ProductVirtualRemainderParams::new(
            self.trace.len(),
            uni_skip_params,
            &self.opening_accumulator,
        );
        let ram_raf_evaluation_params = RafEvaluationSumcheckParams::new(
            &self.program_io.memory_layout,
            &self.one_hot_params,
            &self.opening_accumulator,
        );
        let ram_read_write_checking_params = RamReadWriteCheckingParams::new(
            &self.opening_accumulator,
            &mut self.transcript,
            &self.one_hot_params,
            self.trace.len(),
            &self.rw_config,
        );
        let ram_output_check_params = OutputSumcheckParams::new(
            self.one_hot_params.ram_k,
            &self.program_io,
            &mut self.transcript,
        );
        let instruction_claim_reduction_params =
            InstructionLookupsClaimReductionSumcheckParams::new(
                self.trace.len(),
                &self.opening_accumulator,
                &mut self.transcript,
            );

        // Initialization
        let spartan_product_virtual_remainder = ProductVirtualRemainderProver::initialize(
            spartan_product_virtual_remainder_params,
            Arc::clone(&self.trace),
        );
        let ram_raf_evaluation = RamRafEvaluationSumcheckProver::initialize(
            ram_raf_evaluation_params,
            &self.trace,
            &self.program_io.memory_layout,
        );
        let ram_read_write_checking = RamReadWriteCheckingProver::initialize(
            ram_read_write_checking_params,
            &self.trace,
            &self.preprocessing.shared.bytecode,
            &self.program_io.memory_layout,
            &self.initial_ram_state,
        );
        let ram_output_check = OutputSumcheckProver::initialize(
            ram_output_check_params,
            &self.initial_ram_state,
            &self.final_ram_state,
            &self.program_io.memory_layout,
        );
        let instruction_claim_reduction =
            InstructionLookupsClaimReductionSumcheckProver::initialize(
                instruction_claim_reduction_params,
                Arc::clone(&self.trace),
            );

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

        let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> = vec![
            Box::new(spartan_product_virtual_remainder),
            Box::new(ram_raf_evaluation),
            Box::new(ram_read_write_checking),
            Box::new(ram_output_check),
            Box::new(instruction_claim_reduction),
        ];

        #[cfg(feature = "allocative")]
        write_instance_flamegraph_svg(&instances, "stage2_start_flamechart.svg");
        tracing::info!("Stage 2 proving");
        let (sumcheck_proof, _r_stage2) = BatchedSumcheck::prove(
            instances.iter_mut().map(|v| &mut **v as _).collect(),
            &mut self.opening_accumulator,
            &mut self.transcript,
        );
        #[cfg(feature = "allocative")]
        write_instance_flamegraph_svg(&instances, "stage2_end_flamechart.svg");
        drop_in_background_thread(instances);

        (first_round_proof, sumcheck_proof)
    }

    #[tracing::instrument(skip_all)]
    fn prove_stage3(&mut self) -> SumcheckInstanceProof<F, ProofTranscript> {
        #[cfg(not(target_arch = "wasm32"))]
        print_current_memory_usage("Stage 3 baseline");

        // Initialization params
        let spartan_shift_params = ShiftSumcheckParams::new(
            self.trace.len().log_2(),
            &self.opening_accumulator,
            &mut self.transcript,
        );
        let spartan_instruction_input_params =
            InstructionInputParams::new(&self.opening_accumulator, &mut self.transcript);
        let spartan_registers_claim_reduction_params = RegistersClaimReductionSumcheckParams::new(
            self.trace.len(),
            &self.opening_accumulator,
            &mut self.transcript,
        );

        // Initialize
        let spartan_shift = ShiftSumcheckProver::initialize(
            spartan_shift_params,
            Arc::clone(&self.trace),
            &self.preprocessing.shared.bytecode,
        );
        let spartan_instruction_input = InstructionInputSumcheckProver::initialize(
            spartan_instruction_input_params,
            &self.trace,
            &self.opening_accumulator,
        );
        let spartan_registers_claim_reduction = RegistersClaimReductionSumcheckProver::initialize(
            spartan_registers_claim_reduction_params,
            Arc::clone(&self.trace),
        );

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

        let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> = vec![
            Box::new(spartan_shift),
            Box::new(spartan_instruction_input),
            Box::new(spartan_registers_claim_reduction),
        ];

        #[cfg(feature = "allocative")]
        write_instance_flamegraph_svg(&instances, "stage3_start_flamechart.svg");
        tracing::info!("Stage 3 proving");
        let (sumcheck_proof, _r_stage3) = BatchedSumcheck::prove(
            instances.iter_mut().map(|v| &mut **v as _).collect(),
            &mut self.opening_accumulator,
            &mut self.transcript,
        );
        #[cfg(feature = "allocative")]
        write_instance_flamegraph_svg(&instances, "stage3_end_flamechart.svg");
        drop_in_background_thread(instances);

        sumcheck_proof
    }

    #[tracing::instrument(skip_all)]
    fn prove_stage4(&mut self) -> SumcheckInstanceProof<F, ProofTranscript> {
        #[cfg(not(target_arch = "wasm32"))]
        print_current_memory_usage("Stage 4 baseline");

        let registers_read_write_checking_params = RegistersReadWriteCheckingParams::new(
            self.trace.len(),
            &self.opening_accumulator,
            &mut self.transcript,
            &self.rw_config,
        );
        prover_accumulate_advice(
            &self.advice.untrusted_advice_polynomial,
            &self.advice.trusted_advice_polynomial,
            &self.program_io.memory_layout,
            &self.one_hot_params,
            &mut self.opening_accumulator,
            &mut self.transcript,
            self.rw_config
                .needs_single_advice_opening(self.trace.len().log_2()),
        );
        let ram_val_evaluation_params = ValEvaluationSumcheckParams::new_from_prover(
            &self.one_hot_params,
            &self.opening_accumulator,
            &self.initial_ram_state,
            self.trace.len(),
        );
        let ram_val_final_params =
            ValFinalSumcheckParams::new_from_prover(self.trace.len(), &self.opening_accumulator);

        let registers_read_write_checking = RegistersReadWriteCheckingProver::initialize(
            registers_read_write_checking_params,
            self.trace.clone(),
            &self.preprocessing.shared.bytecode,
            &self.program_io.memory_layout,
        );
        let ram_val_evaluation = RamValEvaluationSumcheckProver::initialize(
            ram_val_evaluation_params,
            &self.trace,
            &self.preprocessing.shared.bytecode,
            &self.program_io.memory_layout,
        );
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

        let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> = vec![
            Box::new(registers_read_write_checking),
            Box::new(ram_val_evaluation),
            Box::new(ram_val_final),
        ];

        #[cfg(feature = "allocative")]
        write_instance_flamegraph_svg(&instances, "stage4_start_flamechart.svg");
        tracing::info!("Stage 4 proving");
        let (sumcheck_proof, _r_stage4) = BatchedSumcheck::prove(
            instances.iter_mut().map(|v| &mut **v as _).collect(),
            &mut self.opening_accumulator,
            &mut self.transcript,
        );
        #[cfg(feature = "allocative")]
        write_instance_flamegraph_svg(&instances, "stage4_end_flamechart.svg");
        drop_in_background_thread(instances);

        sumcheck_proof
    }

    #[tracing::instrument(skip_all)]
    fn prove_stage5(&mut self) -> SumcheckInstanceProof<F, ProofTranscript> {
        #[cfg(not(target_arch = "wasm32"))]
        print_current_memory_usage("Stage 5 baseline");
        let registers_val_evaluation_params =
            RegistersValEvaluationSumcheckParams::new(&self.opening_accumulator);
        let ram_ra_reduction_params = RaReductionParams::new(
            self.trace.len(),
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );
        let lookups_read_raf_params = InstructionReadRafSumcheckParams::new(
            self.trace.len().log_2(),
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );

        let registers_val_evaluation = RegistersValEvaluationSumcheckProver::initialize(
            registers_val_evaluation_params,
            &self.trace,
            &self.preprocessing.shared.bytecode,
            &self.program_io.memory_layout,
        );
        let ram_ra_reduction = RamRaClaimReductionSumcheckProver::initialize(
            ram_ra_reduction_params,
            &self.trace,
            &self.program_io.memory_layout,
            &self.one_hot_params,
        );
        let lookups_read_raf = InstructionReadRafSumcheckProver::initialize(
            lookups_read_raf_params,
            Arc::clone(&self.trace),
        );

        #[cfg(feature = "allocative")]
        {
            print_data_structure_heap_usage(
                "RegistersValEvaluationSumcheckProver",
                &registers_val_evaluation,
            );
            print_data_structure_heap_usage("RamRaClaimReductionSumcheckProver", &ram_ra_reduction);
            print_data_structure_heap_usage("InstructionReadRafSumcheckProver", &lookups_read_raf);
        }

        let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> = vec![
            Box::new(registers_val_evaluation),
            Box::new(ram_ra_reduction),
            Box::new(lookups_read_raf),
        ];

        #[cfg(feature = "allocative")]
        write_instance_flamegraph_svg(&instances, "stage5_start_flamechart.svg");
        tracing::info!("Stage 5 proving");
        let (sumcheck_proof, _r_stage5) = BatchedSumcheck::prove(
            instances.iter_mut().map(|v| &mut **v as _).collect(),
            &mut self.opening_accumulator,
            &mut self.transcript,
        );
        #[cfg(feature = "allocative")]
        write_instance_flamegraph_svg(&instances, "stage5_end_flamechart.svg");
        drop_in_background_thread(instances);

        sumcheck_proof
    }

    #[tracing::instrument(skip_all)]
    fn prove_stage6(&mut self) -> SumcheckInstanceProof<F, ProofTranscript> {
        #[cfg(not(target_arch = "wasm32"))]
        print_current_memory_usage("Stage 6 baseline");

        let bytecode_read_raf_params = BytecodeReadRafSumcheckParams::gen(
            &self.preprocessing.shared.bytecode,
            self.trace.len().log_2(),
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );

        let ram_hamming_booleanity_params =
            HammingBooleanitySumcheckParams::new(&self.opening_accumulator);

        let booleanity_params = BooleanitySumcheckParams::new(
            self.trace.len().log_2(),
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );

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
        let inc_reduction_params = IncClaimReductionSumcheckParams::new(
            self.trace.len(),
            &self.opening_accumulator,
            &mut self.transcript,
        );

        // Advice claim reduction (Phase 1 in Stage 6): trusted and untrusted are separate instances.
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
            &self.preprocessing.shared.bytecode,
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

        // Initialize Phase 1 provers (Stage 6) if advice is present.
        // Note: We clone the advice polynomial here because:
        // 1. Phase1 (Stage 6) destructively binds cycle variables
        // 2. Phase2 (Stage 7) needs a fresh copy to bind address variables
        // 3. Stage 8 RLC needs the original polynomial
        // A future optimization could use Arc<MultilinearPolynomial> with copy-on-write.
        let trusted_advice_phase1 = trusted_advice_phase1_params.map(|params| {
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
        let (sumcheck_proof, _r_stage6) = BatchedSumcheck::prove(
            instances.iter_mut().map(|v| &mut **v as _).collect(),
            &mut self.opening_accumulator,
            &mut self.transcript,
        );
        #[cfg(feature = "allocative")]
        write_instance_flamegraph_svg(&instances, "stage6_end_flamechart.svg");
        drop_in_background_thread(instances);

        sumcheck_proof
    }

    /// Stage 7: HammingWeight + ClaimReduction sumcheck (only log_k_chunk rounds).
    #[tracing::instrument(skip_all)]
    fn prove_stage7(&mut self) -> SumcheckInstanceProof<F, ProofTranscript> {
        // Create params and prover for HammingWeightClaimReduction
        // (r_cycle and r_addr_bool are extracted from Booleanity opening internally)
        let hw_params = HammingWeightClaimReductionParams::new(
            &self.one_hot_params,
            &self.opening_accumulator,
            &mut self.transcript,
        );
        let hw_prover = HammingWeightClaimReductionProver::initialize(
            hw_params,
            &self.trace,
            &self.preprocessing.shared,
            &self.one_hot_params,
        );

        #[cfg(feature = "allocative")]
        print_data_structure_heap_usage("HammingWeightClaimReductionProver", &hw_prover);

        // 3. Run Stage 7 batched sumcheck (address rounds only).
        // Includes HammingWeightClaimReduction plus Phase 2 advice reduction instances (if needed).
        let mut instances: Vec<Box<dyn SumcheckInstanceProver<F, ProofTranscript>>> =
            vec![Box::new(hw_prover)];

        if let Some(gamma) = self.advice_reduction_gamma_trusted {
            if let Some(params) = AdviceClaimReductionPhase2Params::new(
                AdviceKind::Trusted,
                &self.program_io.memory_layout,
                self.trace.len(),
                gamma,
                &self.opening_accumulator,
                self.rw_config
                    .needs_single_advice_opening(self.trace.len().log_2()),
            ) {
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
        let (sumcheck_proof, _) = BatchedSumcheck::prove(
            instances.iter_mut().map(|v| &mut **v as _).collect(),
            &mut self.opening_accumulator,
            &mut self.transcript,
        );
        #[cfg(feature = "allocative")]
        write_instance_flamegraph_svg(&instances, "stage7_end_flamechart.svg");
        drop_in_background_thread(instances);

        sumcheck_proof
    }

    /// Stage 8: Dory batch opening proof.
    /// Builds streaming RLC polynomial directly from trace (no witness regeneration needed).
    #[tracing::instrument(skip_all)]
    fn prove_stage8(
        &mut self,
        opening_proof_hints: HashMap<CommittedPolynomial, PCS::OpeningProofHint>,
    ) -> PCS::Proof {
        tracing::info!("Stage 8 proving (Dory batch opening)");

        let _guard = DoryGlobals::initialize_context(
            self.one_hot_params.k_chunk,
            self.padded_trace_len,
            DoryContext::Main,
        );

        // Get the unified opening point from HammingWeightClaimReduction
        // This contains (r_address_stage7 || r_cycle_stage6) in big-endian
        let (opening_point, _) = self.opening_accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::InstructionRa(0),
            SumcheckId::HammingWeightClaimReduction,
        );
        let log_k_chunk = self.one_hot_params.log_k_chunk;
        let r_address_stage7 = &opening_point.r[..log_k_chunk];

        // 1. Collect all (polynomial, claim) pairs
        let mut polynomial_claims = Vec::new();

        // Dense polynomials: RamInc and RdInc (from IncClaimReduction in Stage 6)
        // These are at r_cycle_stage6 only (length log_T)
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
            // Verify that Inc openings are at the same point as r_cycle from HammingWeightClaimReduction
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

        // Apply Lagrange factor for dense polys: ∏_{i<log_k_chunk} (1 - r_address[i])
        // Because dense polys have fewer variables, we need to account for this
        // Note: r_address is in big-endian, Lagrange factor uses ∏(1 - r_i)
        let lagrange_factor: F = r_address_stage7.iter().map(|r| F::one() - *r).product();

        polynomial_claims.push((CommittedPolynomial::RamInc, ram_inc_claim * lagrange_factor));
        polynomial_claims.push((CommittedPolynomial::RdInc, rd_inc_claim * lagrange_factor));

        // Sparse polynomials: all RA polys (from HammingWeightClaimReduction)
        // These are at (r_address_stage7, r_cycle_stage6)
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

        // Advice polynomials: TrustedAdvice and UntrustedAdvice (from AdviceClaimReduction in Stage 6)
        // These are committed with smaller dimensions, so we apply Lagrange factors to embed
        // them in the top-left block of the main Dory matrix.
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

        // 2. Sample gamma and compute powers for RLC
        let claims: Vec<F> = polynomial_claims.iter().map(|(_, c)| *c).collect();
        self.transcript.append_scalars(&claims);
        let gamma_powers: Vec<F> = self.transcript.challenge_scalar_powers(claims.len());

        // Build DoryOpeningState
        let state = DoryOpeningState {
            opening_point: opening_point.r.clone(),
            gamma_powers,
            polynomial_claims,
        };

        let streaming_data = Arc::new(RLCStreamingData {
            bytecode: Arc::clone(&self.preprocessing.shared.bytecode),
            memory_layout: self.preprocessing.shared.memory_layout.clone(),
        });

        // Build advice polynomials map for RLC
        let mut advice_polys = HashMap::new();
        if let Some(poly) = self.advice.trusted_advice_polynomial.take() {
            advice_polys.insert(CommittedPolynomial::TrustedAdvice, poly);
        }
        if let Some(poly) = self.advice.untrusted_advice_polynomial.take() {
            advice_polys.insert(CommittedPolynomial::UntrustedAdvice, poly);
        }

        // Build streaming RLC polynomial directly (no witness poly regeneration!)
        // Use materialized trace (default, single pass) instead of lazy trace
        let (joint_poly, hint) = state.build_streaming_rlc::<PCS>(
            self.one_hot_params.clone(),
            TraceSource::Materialized(Arc::clone(&self.trace)),
            streaming_data,
            opening_proof_hints,
            advice_polys,
        );

        // Dory opening proof at the unified point
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
        let inputs = postcard::to_stdvec(&100u32).unwrap();

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
        // Tests that max-sized advice (4KB = 512 words) works with a minimal trace.
        // With balanced dims (sigma_a=5, nu_a=4 for 512 words), the minimum padded trace
        // (256 cycles -> total_vars=12) is sufficient to embed advice.
        let mut program = host::Program::new("fibonacci-guest");
        let inputs = postcard::to_stdvec(&5u32).unwrap();
        let trusted_advice = vec![7u8; 4096];
        let untrusted_advice = vec![9u8; 4096];

        let (bytecode, init_memory_state, _) = program.decode();
        let (lazy_trace, trace, final_memory_state, io_device) =
            program.trace(&inputs, &untrusted_advice, &trusted_advice);

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

        let (trusted_commitment, trusted_hint) =
            commit_trusted_advice_preprocessing_only(&prover_preprocessing, &trusted_advice);

        let prover = RV64IMACProver::gen_from_trace(
            &prover_preprocessing,
            lazy_trace,
            trace,
            io_device,
            Some(trusted_commitment),
            Some(trusted_hint),
            final_memory_state,
        );

        // Trace is tiny but advice is max-sized
        assert!(prover.unpadded_trace_len < 512);
        assert_eq!(prover.padded_trace_len, 256);

        let io_device = prover.program_io.clone();
        let (jolt_proof, debug_info) = prover.prove();

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
        // Tests a guest (merkle-tree) that actually consumes both trusted and untrusted advice.
        let mut program = host::Program::new("merkle-tree-guest");
        let (bytecode, init_memory_state, _) = program.decode();

        // Merkle tree with 4 leaves: input=leaf1, trusted=[leaf2, leaf3], untrusted=leaf4
        let inputs = postcard::to_stdvec(&[5u8; 32].as_slice()).unwrap();
        let untrusted_advice = postcard::to_stdvec(&[8u8; 32]).unwrap();
        let mut trusted_advice = postcard::to_stdvec(&[6u8; 32]).unwrap();
        trusted_advice.extend(postcard::to_stdvec(&[7u8; 32]).unwrap());

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
            .expect("Verification failed");

        // Expected merkle root for leaves [5;32], [6;32], [7;32], [8;32]
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
        // Tests that advice opening points are correctly derived from the unified main opening
        // point using Dory's balanced dimension policy.
        //
        // For a small trace (256 cycles), the advice row coordinates span both Stage 6 (cycle)
        // and Stage 7 (address) challenges, verifying the two-phase reduction works correctly.
        let mut program = host::Program::new("fibonacci-guest");
        let inputs = postcard::to_stdvec(&5u32).unwrap();
        let trusted_advice = postcard::to_stdvec(&[7u8; 32]).unwrap();
        let untrusted_advice = postcard::to_stdvec(&[9u8; 32]).unwrap();

        let (bytecode, init_memory_state, _) = program.decode();
        let (lazy_trace, trace, final_memory_state, io_device) =
            program.trace(&inputs, &untrusted_advice, &trusted_advice);

        let shared_preprocessing = JoltSharedPreprocessing::new(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            init_memory_state,
            1 << 16,
        );
        let prover_preprocessing = JoltProverPreprocessing::new(shared_preprocessing.clone());
        let (trusted_commitment, trusted_hint) =
            commit_trusted_advice_preprocessing_only(&prover_preprocessing, &trusted_advice);

        let prover = RV64IMACProver::gen_from_trace(
            &prover_preprocessing,
            lazy_trace,
            trace,
            io_device,
            Some(trusted_commitment),
            Some(trusted_hint),
            final_memory_state,
        );

        assert_eq!(prover.padded_trace_len, 256, "test expects small trace");

        let io_device = prover.program_io.clone();
        let (jolt_proof, debug_info) = prover.prove();
        let debug_info = debug_info.expect("expected debug_info in tests");

        // Get unified opening point and derive expected advice point
        let (opening_point, _) = debug_info
            .opening_accumulator
            .get_committed_polynomial_opening(
                CommittedPolynomial::InstructionRa(0),
                SumcheckId::HammingWeightClaimReduction,
            );
        let mut point_dory_le = opening_point.r.clone();
        point_dory_le.reverse();

        let total_vars = point_dory_le.len();
        let (sigma_main, _nu_main) = DoryGlobals::balanced_sigma_nu(total_vars);
        let (sigma_a, nu_a) = DoryGlobals::advice_sigma_nu_from_max_bytes(
            prover_preprocessing
                .shared
                .memory_layout
                .max_trusted_advice_size as usize,
        );

        // Build expected advice point: [col_bits[0..sigma_a] || row_bits[0..nu_a]]
        let mut expected_advice_le: Vec<_> = point_dory_le[0..sigma_a].to_vec();
        expected_advice_le.extend_from_slice(&point_dory_le[sigma_main..sigma_main + nu_a]);

        // Verify both advice types derive the same opening point
        for (name, kind) in [
            ("trusted", AdviceKind::Trusted),
            ("untrusted", AdviceKind::Untrusted),
        ] {
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
            assert_eq!(point_le, expected_advice_le, "{name} advice point mismatch");
        }

        // Verify end-to-end
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
        let mut program = host::Program::new("memory-ops-guest");
        let (bytecode, init_memory_state, _) = program.decode();
        let (_, _, _, io_device) = program.trace(&[], &[], &[]);

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
            &[],
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
        let mut program = host::Program::new("muldiv-guest");
        let (bytecode, init_memory_state, _) = program.decode();
        let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).unwrap();
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
            &[50],
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
    #[should_panic]
    fn truncated_trace() {
        let mut program = host::Program::new("fibonacci-guest");
        let (bytecode, init_memory_state, _) = program.decode();
        let inputs = postcard::to_stdvec(&9u8).unwrap();
        let (lazy_trace, mut trace, final_memory_state, mut program_io) =
            program.trace(&inputs, &[], &[]);
        trace.truncate(100);
        program_io.outputs[0] = 0; // change the output to 0

        let shared_preprocessing = JoltSharedPreprocessing::new(
            bytecode.clone(),
            program_io.memory_layout.clone(),
            init_memory_state,
            1 << 16,
        );

        let prover_preprocessing = JoltProverPreprocessing::new(shared_preprocessing.clone());

        let prover = RV64IMACProver::gen_from_trace(
            &prover_preprocessing,
            lazy_trace,
            trace,
            program_io.clone(),
            None,
            None,
            final_memory_state,
        );

        let (proof, _) = prover.prove();

        let verifier_preprocessing = JoltVerifierPreprocessing::new(
            prover_preprocessing.shared.clone(),
            prover_preprocessing.generators.to_verifier_setup(),
        );
        let verifier =
            RV64IMACVerifier::new(&verifier_preprocessing, proof, program_io, None, None).unwrap();
        verifier.verify().unwrap();
    }

    #[test]
    #[serial]
    #[should_panic]
    fn malicious_trace() {
        let mut program = host::Program::new("fibonacci-guest");
        let inputs = postcard::to_stdvec(&1u8).unwrap();
        let (bytecode, init_memory_state, _) = program.decode();
        let (lazy_trace, trace, final_memory_state, mut program_io) =
            program.trace(&inputs, &[], &[]);

        // Since the preprocessing is done with the original memory layout, the verifier should fail
        let shared_preprocessing = JoltSharedPreprocessing::new(
            bytecode.clone(),
            program_io.memory_layout.clone(),
            init_memory_state,
            1 << 16,
        );
        let prover_preprocessing = JoltProverPreprocessing::new(shared_preprocessing.clone());

        // change memory address of output & termination bit to the same address as input
        // changes here should not be able to spoof the verifier result
        program_io.memory_layout.output_start = program_io.memory_layout.input_start;
        program_io.memory_layout.output_end = program_io.memory_layout.input_end;
        program_io.memory_layout.termination = program_io.memory_layout.input_start;

        let prover = RV64IMACProver::gen_from_trace(
            &prover_preprocessing,
            lazy_trace,
            trace,
            program_io.clone(),
            None,
            None,
            final_memory_state,
        );
        let (proof, _) = prover.prove();

        let verifier_preprocessing = JoltVerifierPreprocessing::new(
            prover_preprocessing.shared.clone(),
            prover_preprocessing.generators.to_verifier_setup(),
        );
        let verifier =
            JoltVerifier::new(&verifier_preprocessing, proof, program_io, None, None).unwrap();
        verifier.verify().unwrap();
    }
}
