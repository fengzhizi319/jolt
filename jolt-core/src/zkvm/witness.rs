#![allow(static_mut_refs)]

use allocative::Allocative;
use common::constants::XLEN;
use common::jolt_device::MemoryLayout;
use rayon::prelude::*;
use tracer::instruction::Cycle;

use crate::poly::commitment::commitment_scheme::StreamingCommitmentScheme;
use crate::zkvm::bytecode::BytecodePreprocessing;
use crate::zkvm::config::OneHotParams;
use crate::zkvm::instruction::InstructionFlags;
use crate::zkvm::verifier::JoltSharedPreprocessing;
use crate::{
    field::JoltField,
    poly::{multilinear_polynomial::MultilinearPolynomial, one_hot_polynomial::OneHotPolynomial},
    zkvm::ram::remap_address,
};

use super::instruction::{CircuitFlags, LookupQuery};

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Allocative)]
pub enum CommittedPolynomial {
    /*  Twist/Shout witnesses */

    /// Inc polynomial for the registers instance of Twist.
    /// **寄存器增量多项式** (Register Increment Polynomial)。
    /// 属于 Twist 算法实例。用于在离线内存检查（Offline Memory Checking）中确保寄存器读写的一致性。
    /// 它记录了每个 CPU 周期寄存器值的变化量。
    RdInc,

    /// Inc polynomial for the RAM instance of Twist.
    /// **RAM 增量多项式** (RAM Increment Polynomial)。
    /// 属于 Twist 算法实例。用于确保随机访问内存（RAM）读写操作的一致性。
    /// 类似于 RdInc，但针对的是整个内存地址空间。
    RamInc,

    /// One-hot ra polynomial for the instruction lookups instance of Shout.
    /// There are d=8 of these polynomials, `InstructionRa(0) .. InstructionRa(7)`
    /// **指令查找的 One-Hot 读访问多项式** (Instruction Read-Access)。
    /// 属于 Shout (Lasso) 算法实例。用于证明指令解码和查找表的正确性。
    /// 为了处理巨大的查找表，输入被切分为多个块（Chunks）。
    /// `usize` 参数表示这是第几个块（例如 `InstructionRa(0)` 到 `InstructionRa(7)`）。
    InstructionRa(usize),

    /// One-hot ra polynomial for the bytecode instance of Shout.
    /// **字节码的 One-Hot 读访问多项式**。
    /// 属于 Shout 算法实例。用于证明程序计数器（PC）所指向的指令字节码被正确读取。
    /// `usize` 参数同样表示切分后的块索引。
    BytecodeRa(usize),

    /// One-hot ra/wa polynomial for the RAM instance of Twist.
    /// Note that for RAM, ra and wa are the same polynomial because
    /// there is at most one load or store per cycle.
    /// **RAM 的 One-Hot 读/写访问多项式**。
    /// 属于 Twist 算法实例。用于内存地址的访问检查。
    /// 注意：在 Jolt 架构中，每个周期最多发生一次 RAM 加载或存储，因此读访问 (ra) 和写访问 (wa) 被合并为同一个多项式处理。
    /// `usize` 参数表示 RAM 地址切分后的块索引。
    RamRa(usize),

    /// Trusted advice polynomial - committed before proving, verifier has commitment.
    /// Length cannot exceed max_trace_length.
    /// **可信建议多项式** (Trusted Advice)。
    /// 这类多项式在证明生成（Proving）开始之前就已经构建并承诺，验证者（Verifier）已知其承诺值。
    /// 通常包含公共参数或预处理阶段生成的固定数据。长度不能超过最大执行轨迹长度。
    TrustedAdvice,

    /// Untrusted advice polynomial - committed during proving, commitment in proof.
    /// Length cannot exceed max_trace_length.
    /// **不可信建议多项式** (Untrusted Advice)。
    /// 这类多项式是在证明过程中由 Prover 动态生成并承诺的，其承诺值作为证明（Proof）的一部分发送给验证者。
    /// 用于存储证明计算过程中产生的中间辅助数据（Advice）。
    UntrustedAdvice,
}

/// Returns a list of symbols representing all committed polynomials.
/// 返回代表所有已承诺多项式的符号列表。
/// 这些多项式包含了执行轨迹（Trace）中的关键数据，如寄存器读写、RAM 访问和指令查找信息。
pub fn all_committed_polynomials(one_hot_params: &OneHotParams) -> Vec<CommittedPolynomial> {
    // 1. 初始化列表，包含两个基础的增量多项式：
    // - RdInc: 用于追踪寄存器读写的一致性（Twist 实例）。
    // - RamInc: 用于追踪 RAM 读写的一致性（Twist 实例）。
    let mut polynomials = vec![CommittedPolynomial::RdInc, CommittedPolynomial::RamInc];

    // 2. 添加指令查找相关的 One-Hot 读访问多项式 (Instruction Read-Access)。
    // instruction_d 表示为了 Lasso 查找，将指令数据切分成了多少个块（Chunk）。
    // 每个块对应一个 InstructionRa 多项式。
    for i in 0..one_hot_params.instruction_d {
        polynomials.push(CommittedPolynomial::InstructionRa(i));
    }

    // 3. 添加 RAM 访问相关的 One-Hot 读/写访问多项式。
    // ram_d 表示将 RAM 地址切分成了多少个块。
    for i in 0..one_hot_params.ram_d {
        polynomials.push(CommittedPolynomial::RamRa(i));
    }

    // 4. 添加字节码（Bytecode）访问相关的 One-Hot 读访问多项式。
    // bytecode_d 表示将程序计数器（PC）地址切分成了多少个块。
    for i in 0..one_hot_params.bytecode_d {
        polynomials.push(CommittedPolynomial::BytecodeRa(i));
    }

    polynomials
}

impl CommittedPolynomial {
    /// Generate witness data and compute tier 1 commitment for a single row
    /// 生成 Witness 数据并计算单行（Chunk）的一级承诺（Tier 1 Commitment）。
    ///
    /// 此函数是将执行轨迹（Trace）转换为密码学承诺的核心桥梁。
    /// 它遍历输入的 `row_cycles`（一组 CPU 周期），根据当前多项式类型（self）提取相应的数据，
    /// 并将其流式传输给承诺方案（PCS）。
    ///
    /// # 参数
    /// - `setup`: PCS（多项式承诺方案）的证明者设置参数。
    /// - `preprocessing`: 包含字节码映射、内存布局等预处理数据。
    /// - `row_cycles`: 当前处理的一块执行轨迹数据（通常包含多个 CPU 周期）。
    /// - `one_hot_params`: 用于将大整数（如地址）切分为小块（Chunk）的配置参数。
    pub fn stream_witness_and_commit_rows<F, PCS>(
        &self,
        setup: &PCS::ProverSetup,
        preprocessing: &JoltSharedPreprocessing,
        row_cycles: &[tracer::instruction::Cycle],
        one_hot_params: &OneHotParams,
    ) -> <PCS as StreamingCommitmentScheme>::ChunkState
    where
        F: JoltField,
        PCS: StreamingCommitmentScheme<Field = F>,
    {
        match self {
            // Case 1: 寄存器增量多项式
            // 逻辑：计算 (PostValue - PreValue)。
            // 用于离线内存检查，确保寄存器读写的一致性。
            CommittedPolynomial::RdInc => {
                let row: Vec<i128> = row_cycles
                    .iter()
                    .map(|cycle| {
                        // 步骤 1: 获取 rd (目标寄存器) 的写操作信息
                        // `rd_write()` 返回 Option<(RegisterIndex, PreValue, PostValue)>
                        // 如果指令不写寄存器（如 STORE），返回 None，unwrap_or_default 后变成 (0, 0, 0)
                        let (_, pre_value, post_value) = cycle.rd_write().unwrap_or_default();

                        // 步骤 2: 计算数学差值
                        // 使用 i128 防止溢出，并允许负数结果
                        post_value as i128 - pre_value as i128
                    })
                    .collect();

                // 步骤 3: 提交给 PCS (多项式承诺方案)
                // 这是一个 "Dense" (密集) 数据块，直接处理具体数值
                PCS::process_chunk(setup, &row)
            }

            // Case 2: RAM 增量多项式
            // 逻辑：如果是写操作，计算 (PostValue - PreValue)；否则为 0。
            // 类似于 RdInc，但仅针对 RAM 的写操作有效。
            CommittedPolynomial::RamInc => {
                let row: Vec<i128> = row_cycles
                    .iter()
                    .map(|cycle| match cycle.ram_access() {
                        // 分支 A: 只有明确的写入操作 (Store系列指令) 才计算增量
                        tracer::instruction::RAMAccess::Write(write) => {
                            // 计算公式：写入后的新值 - 写入前的旧值
                            write.post_value as i128 - write.pre_value as i128
                        }
                        // 分支 B: 读取操作 (Load) 或 无内存操作 (ALU计算)
                        _ => 0,
                    })
                    .collect();
                PCS::process_chunk(setup, &row)
            }

            // Case 3: 指令查找 One-Hot 多项式
            // 逻辑：Cycle -> LookupIndex (大整数) -> Chunk (切片) -> OneHot 索引。
            // 用于证明指令查找表的正确性。

            CommittedPolynomial::InstructionRa(idx) => {
                // 步骤 1: 遍历当前批次的 CPU 周期 (Row Cycles)
                let row: Vec<Option<usize>> = row_cycles
                    .iter() // 这里使用标准的 iter() 确保是单线程顺序执行，方便日志阅读
                    .map(|cycle| {
                        // 步骤 2: 生成查找表索引 (Lookup Index)
                        // 每一个指令周期的输入操作数 (x, y) 此时被转化为一个巨大的整数 (128位)。
                        // 关键点：这个索引仅由操作数决定，不包含 Opcode。
                        let lookup_index = LookupQuery::<XLEN>::to_lookup_index(cycle);

                        // 步骤 3: 切片提取 (Chunking)
                        // 将巨大的 lookup_index 切分为若干小段 (chunks)。
                        // 'idx' 参数指定我们要提取第几段 (例如第0段是低16位)。
                        // Jolt 使用这种方式避免构建大小为 2^128 的不可能存在的表，而是构建多个 2^16 的小子表。
                        let chunk_val = one_hot_params.lookup_index_chunk(lookup_index, *idx) as usize;


                        // 详细打印调试信息：
                        // idx: 当前正在处理第几个切片（Chunk）
                        // lookup_index: 原始的完整查找键（以十六进制显示，方便查看位模式）
                        // chunk_val: 切分出来的单字节值
                        tracing::info!(
                                        "InstructionRa(chunk_idx={}): cycle={:?}, lookup_index={:#034x}, chunk_val={}",
                                        idx,
                                        cycle,
                                        lookup_index,
                                        chunk_val
                                    );

                        // 步骤 4: 记录证明数据 (Witness)
                        // 这里返回的 chunk_val 稍后将通过 One-Hot 编码进行承诺。
                        Some(chunk_val)
                    })
                    .collect();
                // 调用 PCS 处理 One-Hot 稀疏数据块 (性能优化)
                PCS::process_chunk_onehot(setup, one_hot_params.k_chunk, &row)
            }

            // Case 4: 字节码 One-Hot 多项式
            // 逻辑：Cycle -> PC (程序计数器) -> Chunk -> OneHot 索引。
            // 用于证明取指（Fetch）操作读取了正确的字节码。
            CommittedPolynomial::BytecodeRa(idx) => {
                let row: Vec<Option<usize>> = row_cycles
                    .iter()
                    .map(|cycle| {
                        // 步骤 1: 获取当前周期的真实 PC 值
                        // Jolt 在预处理阶段 (Preprocessing) 已经记录了每一行代码对应的 PC。
                        let pc = preprocessing.bytecode.get_pc(cycle);

                        // 步骤 2: 切片 (Chunking)
                        // 将巨大的 PC 地址 (如 64位) 切分成多个小段 (如 16位一段)。
                        // idx 参数决定了我们要取哪一段。
                        // 比如 idx=0 取低16位，idx=1 取次低16位...
                        Some(one_hot_params.bytecode_pc_chunk(pc, *idx) as usize)
                    })
                    .collect();
                PCS::process_chunk_onehot(setup, one_hot_params.k_chunk, &row)
            }

            // Case 5: RAM 地址 One-Hot 多项式
            // 逻辑：Cycle -> 物理地址 -> 虚拟地址(Remap) -> Chunk -> OneHot 索引。
            // 用于证明内存访问地址的正确性。
            CommittedPolynomial::RamRa(idx) => {
                let row: Vec<Option<usize>> = row_cycles
                    .iter()
                    .map(|cycle| {
                        // 步骤 1: 获取原始物理地址 (Get Raw Address)
                        // 从当前 cycle 中提取涉及的内存地址。
                        // 如果是 RAM 指令 (LB, SW)，则是计算出的实际地址；
                        // 如果是 ALU 指令 (ADD)，这里通常是 0 或无效值。
                        let raw_addr = cycle.ram_access().address() as u64;

                        // 步骤 2: 地址重映射 (Remap Address)
                        // 虚拟机的地址空间很大 (如 64位)，但实际使用的内存是稀疏的 (代码段、堆、栈)。
                        // remap_address 将稀疏的物理地址映射到连续的、较小的证明系统索引空间。
                        // 关键点：如果当前 cycle 不涉及 RAM 访问，或者地址非法，这里会返回 None。
                        remap_address(raw_addr, &preprocessing.memory_layout)

                            // 步骤 3: 地址切片 (Chunking)
                            // map 仅在 remap_address 成功(Some)时执行。
                            // 将重映射后的地址切分为多个小块 (Chunk)。
                            // idx 参数指定我们当前关注的是第几块 (比如低 16 位)。
                            .map(|address| one_hot_params.ram_address_chunk(address, *idx) as usize)
                    })
                    .collect();
                PCS::process_chunk_onehot(setup, one_hot_params.k_chunk, &row)
            }

            // Case 6: Advice 多项式
            // 异常：Advice 多项式由专门的逻辑处理，不应通过流式生成。
            CommittedPolynomial::TrustedAdvice | CommittedPolynomial::UntrustedAdvice => {
                panic!("Advice polynomials should not use streaming witness generation")
            }
        }
    }

    #[tracing::instrument(skip_all, name = "CommittedPolynomial::generate_witness")]
    pub fn generate_witness<F>(
        &self,
        bytecode_preprocessing: &BytecodePreprocessing,
        memory_layout: &MemoryLayout,
        trace: &[Cycle],
        one_hot_params: Option<&OneHotParams>,
    ) -> MultilinearPolynomial<F>
    where
        F: JoltField,
    {
        match self {
            CommittedPolynomial::BytecodeRa(i) => {
                let one_hot_params = one_hot_params.unwrap();
                let addresses: Vec<_> = trace
                    .par_iter()
                    .map(|cycle| {
                        let pc = bytecode_preprocessing.get_pc(cycle);
                        Some(one_hot_params.bytecode_pc_chunk(pc, *i))
                    })
                    .collect();
                MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
                    addresses,
                    one_hot_params.k_chunk,
                ))
            }
            CommittedPolynomial::RamRa(i) => {
                let one_hot_params = one_hot_params.unwrap();
                let addresses: Vec<_> = trace
                    .par_iter()
                    .map(|cycle| {
                        remap_address(cycle.ram_access().address() as u64, memory_layout)
                            .map(|address| one_hot_params.ram_address_chunk(address, *i))
                    })
                    .collect();
                MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
                    addresses,
                    one_hot_params.k_chunk,
                ))
            }
            CommittedPolynomial::RdInc => {
                let coeffs: Vec<i128> = trace
                    .par_iter()
                    .map(|cycle| {
                        let (_, pre_value, post_value) = cycle.rd_write().unwrap_or_default();
                        post_value as i128 - pre_value as i128
                    })
                    .collect();
                coeffs.into()
            }
            CommittedPolynomial::RamInc => {
                let coeffs: Vec<i128> = trace
                    .par_iter()
                    .map(|cycle| {
                        let ram_op = cycle.ram_access();
                        match ram_op {
                            tracer::instruction::RAMAccess::Write(write) => {
                                write.post_value as i128 - write.pre_value as i128
                            }
                            _ => 0,
                        }
                    })
                    .collect();
                coeffs.into()
            }
            CommittedPolynomial::InstructionRa(i) => {
                let one_hot_params = one_hot_params.unwrap();
                let addresses: Vec<_> = trace
                    .par_iter()
                    .map(|cycle| {
                        let lookup_index = LookupQuery::<XLEN>::to_lookup_index(cycle);
                        Some(one_hot_params.lookup_index_chunk(lookup_index, *i))
                    })
                    .collect();
                MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
                    addresses,
                    one_hot_params.k_chunk,
                ))
            }
            CommittedPolynomial::TrustedAdvice | CommittedPolynomial::UntrustedAdvice => {
                panic!("Advice polynomials should not use generate_witness")
            }
        }
    }

    pub fn get_onehot_k(&self, one_hot_params: &OneHotParams) -> Option<usize> {
        match self {
            CommittedPolynomial::InstructionRa(_)
            | CommittedPolynomial::BytecodeRa(_)
            | CommittedPolynomial::RamRa(_) => Some(one_hot_params.k_chunk),
            _ => None,
        }
    }
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Allocative)]
pub enum VirtualPolynomial {
    PC,
    UnexpandedPC,
    NextPC,
    NextUnexpandedPC,
    NextIsNoop,
    NextIsVirtual,
    NextIsFirstInSequence,
    LeftLookupOperand,
    RightLookupOperand,
    LeftInstructionInput,
    RightInstructionInput,
    Product,
    ShouldJump,
    ShouldBranch,
    WritePCtoRD,
    WriteLookupOutputToRD,
    Rd,
    Imm,
    Rs1Value,
    Rs2Value,
    RdWriteValue,
    Rs1Ra,
    Rs2Ra,
    RdWa,
    LookupOutput,
    InstructionRaf,
    InstructionRafFlag,
    InstructionRa(usize),
    RegistersVal,
    RamAddress,
    RamRa,
    RamReadValue,
    RamWriteValue,
    RamVal,
    RamValInit,
    RamValFinal,
    RamHammingWeight,
    UnivariateSkip,
    OpFlags(CircuitFlags),
    InstructionFlags(InstructionFlags),
    LookupTableFlag(usize),
}

#[cfg(test)]
mod tests1 {
    use super::*;
    use crate::zkvm::config::OneHotParams;
    use crate::poly::commitment::commitment_scheme::{CommitmentScheme, StreamingCommitmentScheme};
    use crate::transcripts::{AppendToTranscript, Transcript};
    use crate::utils::small_scalar::SmallScalar;
    use crate::utils::errors::ProofVerifyError;
    use crate::poly::multilinear_polynomial::MultilinearPolynomial;
    use ark_bn254::Fr;
    use ark_serialize::{CanonicalSerialize, CanonicalDeserialize};
    use std::borrow::Borrow;

    #[derive(Clone, Debug, Default)]
    struct MockPCS;

    #[derive(Clone, Debug, Default, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
    struct MockCommitment;
    impl AppendToTranscript for MockCommitment {
        fn append_to_transcript<ProofTranscript: Transcript>(&self, _transcript: &mut ProofTranscript) {}
    }

    #[derive(Clone, Debug, PartialEq)]
    struct MockChunkState {
        processed_dense_len: Option<usize>,
        processed_onehot: Option<Vec<Option<usize>>>,
    }

    impl CommitmentScheme for MockPCS {
        type Field = Fr;
        type ProverSetup = ();
        type VerifierSetup = ();
        type Commitment = MockCommitment;
        type Proof = ();
        type BatchedProof = ();
        type OpeningProofHint = ();

        fn setup_prover(_: usize) -> Self::ProverSetup {}
        fn setup_verifier(_: &Self::ProverSetup) -> Self::VerifierSetup {}
        fn commit(_: &MultilinearPolynomial<Self::Field>, _: &Self::ProverSetup) -> (Self::Commitment, Self::OpeningProofHint) { (MockCommitment, ()) }
        fn batch_commit<U>(_: &[U], _: &Self::ProverSetup) -> Vec<(Self::Commitment, Self::OpeningProofHint)> where U: Borrow<MultilinearPolynomial<Self::Field>> + Sync { vec![] }
        fn prove<PT: Transcript>(_: &Self::ProverSetup, _: &MultilinearPolynomial<Self::Field>, _: &[<Self::Field as JoltField>::Challenge], _: Option<Self::OpeningProofHint>, _: &mut PT) -> Self::Proof {}
        fn verify<PT: Transcript>(_: &Self::Proof, _: &Self::VerifierSetup, _: &mut PT, _: &[<Self::Field as JoltField>::Challenge], _: &Self::Field, _: &Self::Commitment) -> Result<(), ProofVerifyError> { Ok(()) }
        fn protocol_name() -> &'static [u8] { b"MockPCS" }
    }

    impl StreamingCommitmentScheme for MockPCS {
        type ChunkState = MockChunkState;

        fn process_chunk<T: SmallScalar>(_setup: &Self::ProverSetup, chunk: &[T]) -> Self::ChunkState {
            MockChunkState {
                processed_dense_len: Some(chunk.len()),
                processed_onehot: None,
            }
        }

        fn process_chunk_onehot(
            _setup: &Self::ProverSetup,
            _k: usize,
            row: &[Option<usize>],
        ) -> Self::ChunkState {
            MockChunkState {
                processed_dense_len: None,
                processed_onehot: Some(row.to_vec()),
            }
        }

        fn aggregate_chunks(
            _setup: &Self::ProverSetup,
            _onehot_k: Option<usize>,
            _tier1_commitments: &[Self::ChunkState],
        ) -> (Self::Commitment, Self::OpeningProofHint) {
            (MockCommitment, ())
        }
    }

    #[test]
    fn test_stream_witness_rd_inc() {
        let cycles: Vec<tracer::instruction::Cycle> = vec![tracer::instruction::Cycle::ADD(Default::default())];
        let poly = CommittedPolynomial::RdInc;
        let setup = ();
        let preprocessing_ptr = std::ptr::NonNull::<JoltSharedPreprocessing>::dangling().as_ptr();
        let preprocessing = unsafe { &*preprocessing_ptr };
        let one_hot_params = OneHotParams::new(10, 1024, 1024);

        let result = poly.stream_witness_and_commit_rows::<Fr, MockPCS>(
            &setup,
            preprocessing,
            &cycles,
            &one_hot_params
        );

        assert!(result.processed_dense_len.is_some());
        assert_eq!(result.processed_dense_len.unwrap(), 1);
    }

    #[test]
    fn test_stream_witness_instruction_ra_dispatch() {
        let _ = tracing_subscriber::fmt().try_init();
        let cycles: Vec<tracer::instruction::Cycle> = vec![tracer::instruction::Cycle::ADD(Default::default())];
        tracing::info!("Cycles: {:?}", cycles);
        let poly = CommittedPolynomial::InstructionRa(0);
        let setup = ();
        let preprocessing_ptr = std::ptr::NonNull::<JoltSharedPreprocessing>::dangling().as_ptr();
        let preprocessing = unsafe { &*preprocessing_ptr };
        let one_hot_params = OneHotParams::new(10, 1024, 1024);

        let result = poly.stream_witness_and_commit_rows::<Fr, MockPCS>(
            &setup,
            preprocessing,
            &cycles,
            &one_hot_params
        );

        assert!(result.processed_onehot.is_some());
        assert_eq!(result.processed_onehot.unwrap().len(), 1);
    }

    #[test]
    fn test_all_committed_polynomials() {
        let mut params = OneHotParams::default();
        params.instruction_d = 2;
        params.ram_d = 3;
        params.bytecode_d = 4;

        let polys = all_committed_polynomials(&params);
        assert_eq!(polys.len(), 11);

        let mut expected = vec![CommittedPolynomial::RdInc, CommittedPolynomial::RamInc];
        for i in 0..2 {
            expected.push(CommittedPolynomial::InstructionRa(i));
        }
        for i in 0..3 {
            expected.push(CommittedPolynomial::RamRa(i));
        }
        for i in 0..4 {
            expected.push(CommittedPolynomial::BytecodeRa(i));
        }

        assert_eq!(polys, expected);
    }

    #[test]
    fn test_all_committed_polynomials_zero_depths() {
        let mut params = OneHotParams::default();
        params.instruction_d = 0;
        params.ram_d = 0;
        params.bytecode_d = 0;

        let polys = all_committed_polynomials(&params);

        assert_eq!(polys.len(), 2);
        assert_eq!(polys, vec![CommittedPolynomial::RdInc, CommittedPolynomial::RamInc]);
    }
}
