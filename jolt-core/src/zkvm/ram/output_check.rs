use crate::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
        },
        program_io_polynomial::ProgramIOPolynomial,
        range_mask_polynomial::RangeMaskPolynomial,
        split_eq_poly::GruenSplitEqPolynomial,
        unipoly::UniPoly,
    },
    subprotocols::{
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
    utils::math::Math,
    zkvm::{ram::remap_address, witness::VirtualPolynomial},
};
use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use common::{constants::RAM_START_ADDRESS, jolt_device::MemoryLayout};
use rayon::prelude::*;
use tracer::JoltDevice;

// RAM output sumchecks
//
// OutputSumcheck:
//   Proves the zero-check
//     Σ_k eq(r_address, k) ⋅ io_mask(k) ⋅ (Val_final(k) − Val_io(k)) = 0,
//   where:
//   - r_address is a random address challenge vector.
//   - io_mask is the MLE of the I/O-region indicator (1 on matching {0,1}-points).
//   - Val_final(k) is the final memory value at address k.
//   - Val_io(k) is the publicly claimed output value at address k.
//
// ValFinalSumcheck:
//   Proves the relation
//     Val_final(r_address) − Val_init(r_address) = Σ_j inc(r_address, j) ⋅ wa(r_address, j),
//   where:
//   - Val_init(r_address) is the initial memory value at r_address.
//   - inc is the MLE of the per-cycle increment; wa is the MLE of the write indicator.

/// Degree bonud of the sumcheck round polynomials in [`OutputSumcheckVerifier`].
const OUTPUT_SUMCHECK_DEGREE_BOUND: usize = 3;

#[derive(Allocative, Clone)]
pub struct OutputSumcheckParams<F: JoltField> {
    pub K: usize,
    pub r_address: Vec<F::Challenge>,
    pub program_io: JoltDevice,
}

impl<F: JoltField> OutputSumcheckParams<F> {
    /// 初始化 `OutputSumcheckParams` 实例。
    ///
    /// # 作用
    /// 构建用于 RAM 输出一致性检查 Sumcheck 协议的参数集。
    ///
    /// # 核心逻辑
    /// 1. **生成随机挑战**:
    ///    利用 Fiat-Shamir 变换，从 Transcript 中抽取随机挑战向量 $r_{address}$。
    ///    向量长度为 $\log_2 K$，对应于 RAM 地址空间的比特数（变量数）。
    ///    这个 $r_{address}$ 决定了 Sumcheck 协议需要证明的具体随机线性组合点，
    ///    即验证方程 $\sum eq(r_{address}, k) \cdot \dots = 0$ 中的 $r_{address}$。
    ///
    /// # 参数
    /// * `ram_K`: RAM 的大小（地址条目数量），必须是 2 的幂。
    /// * `program_io`: Jolt 设备的 I/O 配置信息，包含内存布局。
    /// * `transcript`: 用于生成伪随机数的 Fiat-Shamir Transcript。
    pub fn new(ram_K: usize, program_io: &JoltDevice, transcript: &mut impl Transcript) -> Self {
        // 计算地址空间的位数 (log2(K))，并生成对应数量的随机挑战标量。
        // 这些挑战标量构成了 Sumcheck 验证方程中 eq(r_address, k) 的 r_address 部分。
        let r_address = transcript.challenge_vector_optimized::<F>(ram_K.log_2());
        Self {
            K: ram_K,
            r_address,
            program_io: program_io.clone(),
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for OutputSumcheckParams<F> {
    fn degree(&self) -> usize {
        OUTPUT_SUMCHECK_DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.K.log_2()
    }

    fn input_claim(&self, _: &dyn OpeningAccumulator<F>) -> F {
        F::zero()
    }

    fn normalize_opening_point(
        &self,
        challenges: &[<F as JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }
}

/// Sumcheck prover for [`OutputSumcheckVerifier`].
#[derive(Allocative)]
pub struct OutputSumcheckProver<F: JoltField> {
    /// Val(k, 0)
    val_init: MultilinearPolynomial<F>,
    /// The MLE of the final RAM state
    val_final: MultilinearPolynomial<F>,
    /// Val_io(k) = Val_final(k) if k is in the "IO" region of memory,
    /// and 0 otherwise.
    /// Equivalently, Val_io(k) = Val(k, T) * io_mask(k) for
    /// k \in {0, 1}^log(K)
    val_io: MultilinearPolynomial<F>,
    /// Split-EQ structure over the address variables (Gruen + Dao-Thaler)
    eq_r_address: GruenSplitEqPolynomial<F>,
    /// io_mask(k) serves as a "mask" for the IO region of memory,
    /// i.e. io_mask(k) = 1 if k is in the "IO" region of memory,
    /// and 0 otherwise.
    io_mask: MultilinearPolynomial<F>,
    #[allocative(skip)]
    pub params: OutputSumcheckParams<F>,
}

impl<F: JoltField> OutputSumcheckProver<F> {
    /// 初始化 `OutputSumcheckProver` 实例。
    ///
    /// # 作用
    /// 准备 RAM 输出一致性检查 Sumcheck 协议的 Prover。
    /// 该协议旨在证明：对于所有的内存地址 $k$，如果 $k$ 属于 I/O 区域，则最终内存值 `Val_final(k)` 等于公开声明的输出值 `Val_io(k)`。
    ///
    /// 具体证明的方程为：
    /// $$ \sum_{k} eq(r_{address}, k) \cdot io\_mask(k) \cdot (Val_{final}(k) - Val_{io}(k)) = 0 $$
    ///
    /// # 核心逻辑
    /// 1. **校验输入**: 确保初始和最终内存状态长度一致且为 2 的幂。
    /// 2. **确定 I/O 区域**:
    ///    计算内存布局中 I/O 区域的起始和结束索引（经过地址重映射）。
    ///    Jolt 设备的内存中并非所有地址都涉及 I/O，只有特定段是 I/O 映射区。
    /// 3. **构建 `val_io` 多项式**:
    ///    `val_io` 代表了 Prover 声称的 I/O 输出值。
    ///    逻辑上，$Val_{io}(k) = Val_{final}(k)$ 当 $k \in [io\_start, io\_end)$，否则为 0。
    ///    这里通过并行复制 `final_ram_state` 的对应片段来构建它。
    /// 4. **构建 `io_mask` 多项式**:
    ///    这是一个指示多项式（Indicator Polynomial），用于“屏蔽”非 I/O 区域。
    ///    当 $k$ 在 I/O 区域内时为 1 (true)，否则为 0 (false)。
    /// 5. **初始化 Eq 多项式**:
    ///    使用预先生成的挑战点 $r_{address}$ 初始化 `GruenSplitEqPolynomial`，用于后续 Sumcheck 轮次中的线性组合系数计算。
    /// 6. **封装多项式**:
    ///    将所有向量（`val_init`, `val_final`, `val_io`, `io_mask`）转换为 `MultilinearPolynomial` 格式以供 Sumcheck 协议使用。
    ///
    /// # 参数
    /// * `params`: 包含随机挑战点 $r_{address}$ 的参数集。
    /// * `initial_ram_state`: 初始 RAM 状态向量（在本协议中不参与计算 message，但需要绑定以供后续步骤使用）。
    /// * `final_ram_state`: 最终 RAM 状态向量。
    /// * `memory_layout`: Jolt 设备的内存布局配置。
    #[tracing::instrument(skip_all, name = "OutputSumcheckProver::initialize")]
    pub fn initialize(
        params: OutputSumcheckParams<F>,
        initial_ram_state: &[u64],
        final_ram_state: &[u64],
        memory_layout: &MemoryLayout,
    ) -> Self {
        let K = final_ram_state.len();
        debug_assert_eq!(initial_ram_state.len(), final_ram_state.len());
        debug_assert!(K.is_power_of_two());

        // 计算内存中 IO 区域对应的 Witness 索引范围（起始和结束）
        // remap_address 将物理地址映射为密集的数组索引
        let io_start = remap_address(memory_layout.input_start, memory_layout).unwrap() as usize;
        let io_end = remap_address(RAM_START_ADDRESS, memory_layout).unwrap() as usize;

        // 构建 Val_io 多项式向量
        // 逻辑：仅拷贝 Final RAM 中属于 IO 区域的值，其余部分保持为 0。
        // 这表示了 Prover 声称的程序输出。
        let mut val_io = vec![0; K];
        val_io[io_start..io_end]
            .par_iter_mut()
            .zip(final_ram_state[io_start..io_end].par_iter())
            .for_each(|(dest, src)| *dest = *src);

        // 构建 io_mask 多项式向量
        // 这是一个布尔掩码向量：在 IO 区域内为 true (1)，区域外为 false (0)。
        // 它在 Sumcheck 方程中充当选择器，确保只检查 IO 区域的一致性。
        let mut io_mask = vec![false; K];
        io_mask[io_start..io_end]
            .par_iter_mut()
            .for_each(|k| *k = true);

        // 初始化用于地址绑定的 Split-Eq 多项式
        // BindingOrder::LowToHigh 意味着 Sumcheck 将从低位到高位处理变量
        let eq_r_address = GruenSplitEqPolynomial::new(&params.r_address, BindingOrder::LowToHigh);

        Self {
            val_init: initial_ram_state.to_vec().into(),
            val_final: final_ram_state.to_vec().into(), // 转换为 MultilinearPolynomial
            val_io: val_io.into(),
            eq_r_address,
            io_mask: io_mask.into(),
            params,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for OutputSumcheckProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(skip_all, name = "OutputSumcheckProver::compute_message")]
    fn compute_message(&mut self, _: usize, previous_claim: F) -> UniPoly<F> {
        let Self {
            eq_r_address,
            io_mask,
            val_final,
            val_io,
            ..
        } = self;

        // For s(X) = eq_lin(X) * q(X), where q(X) = io_mask(X) * (val_final(X) - val_io(X))
        // q is quadratic in the current variable. Compute:
        //   c0 = q(0) = io0 * (vf0 - vio0)
        //   e  = coeff of X^2 in q(X) = (io1 - io0) * ((vf1 - vio1) - (vf0 - vio0))
        let [q_constant, q_quadratic] = eq_r_address.par_fold_out_in_unreduced::<9, 2>(&|g| {
            let io0 = io_mask.get_bound_coeff(2 * g);
            let io1 = io_mask.get_bound_coeff(2 * g + 1);
            let vf0 = val_final.get_bound_coeff(2 * g);
            let vf1 = val_final.get_bound_coeff(2 * g + 1);
            let vio0 = val_io.get_bound_coeff(2 * g);
            let vio1 = val_io.get_bound_coeff(2 * g + 1);

            let v0 = vf0 - vio0;
            let v1 = vf1 - vio1;
            let c0 = io0 * v0;
            let e = (io1 - io0) * (v1 - v0);
            [c0, e]
        });

        eq_r_address.gruen_poly_deg_3(q_constant, q_quadratic, previous_claim)
    }

    #[tracing::instrument(skip_all, name = "OutputSumcheckProver::ingest_challenge")]
    fn ingest_challenge(&mut self, r_j: F::Challenge, _: usize) {
        // Bind address variable
        let Self {
            val_init,
            val_final,
            val_io,
            eq_r_address,
            io_mask,
            ..
        } = self;

        // We bind Val_init here despite the fact that it is not used in `compute_message`
        // because we'll need Val_init(r) in `ValFinalSumcheck`
        val_init.bind_parallel(r_j, BindingOrder::LowToHigh);
        val_final.bind_parallel(r_j, BindingOrder::LowToHigh);
        val_io.bind_parallel(r_j, BindingOrder::LowToHigh);
        eq_r_address.bind(r_j);
        io_mask.bind_parallel(r_j, BindingOrder::LowToHigh);
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let Self {
            val_final,
            val_init,
            ..
        } = self;
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RamValFinal,
            SumcheckId::RamOutputCheck,
            opening_point.clone(),
            val_final.final_sumcheck_claim(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RamValInit,
            SumcheckId::RamOutputCheck,
            opening_point,
            val_init.final_sumcheck_claim(),
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

pub struct OutputSumcheckVerifier<F: JoltField> {
    params: OutputSumcheckParams<F>,
}

impl<F: JoltField> OutputSumcheckVerifier<F> {
    pub fn new(ram_K: usize, program_io: &JoltDevice, transcript: &mut impl Transcript) -> Self {
        let params = OutputSumcheckParams::new(ram_K, program_io, transcript);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for OutputSumcheckVerifier<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let val_final_claim = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RamValFinal,
                SumcheckId::RamOutputCheck,
            )
            .1;

        let r_address = &self.params.r_address;
        // Derive r' using the same endianness conversion as used when caching openings
        let r_address_prime = self.params.normalize_opening_point(sumcheck_challenges).r;
        let program_io = &self.params.program_io;

        let io_mask = RangeMaskPolynomial::<F>::new(
            remap_address(
                program_io.memory_layout.input_start,
                &program_io.memory_layout,
            )
                .unwrap() as u128,
            remap_address(RAM_START_ADDRESS, &program_io.memory_layout).unwrap() as u128,
        );
        let val_io = ProgramIOPolynomial::new(program_io);

        let eq_eval: F = EqPolynomial::<F>::mle(r_address, &r_address_prime);
        let io_mask_eval = io_mask.evaluate_mle(&r_address_prime);
        let val_io_eval: F = val_io.evaluate(&r_address_prime);

        // Recall that the sumcheck expression is:
        //   0 = \sum_k eq(r_address, k) * io_range(k) * (Val_final(k) - Val_io(k))
        eq_eval * io_mask_eval * (val_final_claim - val_io_eval)
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
            VirtualPolynomial::RamValFinal,
            SumcheckId::RamOutputCheck,
            opening_point.clone(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RamValInit,
            SumcheckId::RamOutputCheck,
            opening_point,
        );
    }
}
