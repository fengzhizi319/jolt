use num_traits::Zero;

use crate::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
        },
        unipoly::UniPoly,
    },
    subprotocols::{
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
    utils::math::Math,
    zkvm::{
        bytecode::BytecodePreprocessing,
        claim_reductions::AdviceKind,
        ram::remap_address,
        witness::{CommittedPolynomial, VirtualPolynomial},
    },
};
use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use common::jolt_device::MemoryLayout;
use rayon::prelude::*;
use tracer::{instruction::Cycle, JoltDevice};

/// Degree bound of the sumcheck round polynomials in [`ValFinalSumcheckVerifier`].
const VAL_FINAL_SUMCHECK_DEGREE_BOUND: usize = 2;

#[derive(Allocative, Clone)]
pub struct ValFinalSumcheckParams<F: JoltField> {
    pub T: usize,
    pub r_address: Vec<F::Challenge>,
    pub val_init_eval: F,
}

impl<F: JoltField> ValFinalSumcheckParams<F> {
    pub fn new_from_prover(
        trace_len: usize,
        opening_accumulator: &ProverOpeningAccumulator<F>,
    ) -> Self {
        let (r_address, val_init_eval) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamValInit,
            SumcheckId::RamOutputCheck,
        );

        Self {
            T: trace_len,
            val_init_eval,
            r_address: r_address.r.clone(),
        }
    }

    pub fn new_from_verifier(
        initial_ram_state: &[u64],
        program_io: &JoltDevice,
        trace_len: usize,
        ram_K: usize,
        opening_accumulator: &VerifierOpeningAccumulator<F>,
    ) -> Self {
        let r_address = opening_accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RamValFinal,
                SumcheckId::RamOutputCheck,
            )
            .0
            .r;

        let n_memory_vars = ram_K.log_2();

        // When needs_single_advice_opening is true, advice is only opened at RamValEvaluation
        // (the two points are identical). Otherwise, we use RamValFinalEvaluation.
        let log_T = trace_len.log_2();
        let rw_config = crate::zkvm::config::ReadWriteConfig::new(log_T, ram_K.log_2());
        let advice_sumcheck_id = if rw_config.needs_single_advice_opening(log_T) {
            SumcheckId::RamValEvaluation
        } else {
            SumcheckId::RamValFinalEvaluation
        };

        let untrusted_advice_contribution = super::calculate_advice_memory_evaluation(
            opening_accumulator.get_advice_opening(AdviceKind::Untrusted, advice_sumcheck_id),
            (program_io.memory_layout.max_untrusted_advice_size as usize / 8)
                .next_power_of_two()
                .log_2(),
            program_io.memory_layout.untrusted_advice_start,
            &program_io.memory_layout,
            &r_address,
            n_memory_vars,
        );

        let trusted_advice_contribution = super::calculate_advice_memory_evaluation(
            opening_accumulator.get_advice_opening(AdviceKind::Trusted, advice_sumcheck_id),
            (program_io.memory_layout.max_trusted_advice_size as usize / 8)
                .next_power_of_two()
                .log_2(),
            program_io.memory_layout.trusted_advice_start,
            &program_io.memory_layout,
            &r_address,
            n_memory_vars,
        );

        // Compute the public part of val_init evaluation
        let val_init_public: MultilinearPolynomial<F> =
            MultilinearPolynomial::from(initial_ram_state.to_vec());

        // Combine all contributions: untrusted + trusted + public
        let val_init_eval = untrusted_advice_contribution
            + trusted_advice_contribution
            + val_init_public.evaluate(&r_address);

        ValFinalSumcheckParams {
            T: trace_len,
            val_init_eval,
            r_address,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for ValFinalSumcheckParams<F> {
    fn degree(&self) -> usize {
        VAL_FINAL_SUMCHECK_DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.T.log_2()
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let (_, val_final_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamValFinal,
            SumcheckId::RamOutputCheck,
        );
        val_final_claim - self.val_init_eval
    }

    fn normalize_opening_point(
        &self,
        challenges: &[<F as JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }
}

#[derive(Allocative)]
pub struct ValFinalSumcheckProver<F: JoltField> {
    inc: MultilinearPolynomial<F>,
    wa: MultilinearPolynomial<F>,
    pub params: ValFinalSumcheckParams<F>,
}

impl<F: JoltField> ValFinalSumcheckProver<F> {
    #[tracing::instrument(skip_all, name = "ValFinalSumcheckProver::initialize")]
    pub fn initialize(
        params: ValFinalSumcheckParams<F>,
        trace: &[Cycle],
        bytecode_preprocessing: &BytecodePreprocessing,
        memory_layout: &MemoryLayout,
    ) -> Self {
        // ========================================================================
        // 1. 预计算地址指纹表 (Precompute Eq Evaluations)
        // ========================================================================
        // [算法原理: Multilinear Extension & Lookup Table]
        // 算法公式：Table[k] = Eq(k, r_address) = \prod_{i=0}^{m-1} ( (1-k_i)(1-r_i) + k_i r_i )
        //
        // 我们不希望在遍历百万级 Trace 时，对每一行都重新计算 Eq 函数（那需要 O(log K) 次乘法）。
        // 既然内存地址空间被重新映射到了较小的范围 K (Memory Layout)，
        // 我们可以先用 O(K) 的时间算出所有可能的地址指纹，存入表中。
        //
        // [优化点 1: 查表法代替计算]
        // 复杂度从 O(TraceLen * log K) 降低到 O(K + TraceLen)。
        let eq_r_address = EqPolynomial::evals(&params.r_address);

        let span = tracing::span!(tracing::Level::INFO, "compute wa(r_address, j)");
        let _guard = span.enter();

        // ========================================================================
        // 2. 构造地址选择多项式 (Construct WA Polynomial)
        // ========================================================================
        // [算法原理: Projection / Filtering]
        // `wa` (Weighted Address) 是一个选择器向量。
        // 对于 Trace 中的每一行 (Cycle t)，如果它操作了地址 A_t：
        // wa[t] = Eq(A_t, r_address)
        //
        // 物理意义：
        // Verifier 想检查随机地址 r_address 上的值。
        // `wa` 将 Trace 中所有操作了 "类似 r_address" 的行筛选出来（赋予高权重），
        // 而操作其他地址的行赋予接近 0 的权重。
        //
        // [优化点 2: 并行迭代 (Rayon Parallel Iterator)]
        // 每一行的 wa 计算是完全独立的，因此使用 .par_iter() 进行数据并行处理。
        // 这对于大规模 Trace (如 2^20 行) 带来线性加速比。
        let wa: Vec<F> = trace
            .par_iter()
            .map(|cycle| {
                // remap_address: 将 64位物理地址映射到紧凑的索引空间 (0..K)
                remap_address(cycle.ram_access().address() as u64, memory_layout)
                    // 如果地址映射成功，查表获取指纹；否则 (非RAM操作) 权重为 0
                    .map_or(F::zero(), |k| eq_r_address[k as usize])
            })
            .collect();

        // 将向量转化为多线性多项式对象
        let wa = MultilinearPolynomial::from(wa);

        drop(_guard);
        drop(span);

        // ========================================================================
        // 3. 生成增量/值多项式 (Witness Generation)
        // ========================================================================
        // [约束原理: Memory Consistency Formula]
        // `inc` (Increment/Value) 代表了每一次内存操作对最终状态的贡献值。
        //
        // 核心约束公式 (由 Sumcheck 证明)：
        // Val_final(r) = Val_init(r) + \sum_{t=0}^{T-1} wa(t) * inc(t)
        //
        // - Val_final(r): 最终内存状态在 r 处的评估值 (Fingerprint of Final Memory)。
        // - Val_init(r): 初始内存状态在 r 处的评估值。
        // - wa(t): 选择器，只有当第 t 步操作的地址匹配 r 时才非零。
        // - inc(t): 第 t 步操作写入的值（或值的变化量）。
        //
        // 这个公式本质上在说：
        // "最终内存里的东西 = 初始有的东西 + 过程中所有写入东西的总和"
        // (注：具体是覆盖写还是增量写取决于 Jolt 的具体 Memory Argument 变体，
        // 但数学形式都是线性累加)。
        let inc = CommittedPolynomial::RamInc.generate_witness(
            bytecode_preprocessing,
            memory_layout,
            trace,
            None,
        );

        // [代码中保留的测试逻辑 - 用于验证约束]
        // 下面的注释代码解释了约束的具体含义：
        // #[cfg(test)]
        // {
        //     let expected = val_final.final_sumcheck_claim(); // 左边：最终状态的 Claim
        //     // 右边：初始状态 Claim + (地址权重 * 操作值) 的总和
        //     let actual = val_init.final_sumcheck_claim()
        //         + wa_r_address
        //             .par_iter()
        //             .enumerate()
        //             .map(|(j, wa)| inc.get_coeff(j) * wa)
        //             .sum::<F>();
        //     assert_eq!(expected, actual);
        // }

        Self { wa, inc, params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for ValFinalSumcheckProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(skip_all, name = "ValFinalSumcheckProver::compute_message")]
    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let evals = (0..self.inc.len() / 2)
            .into_par_iter()
            .map(|j| {
                let inc_evals = self
                    .inc
                    .sumcheck_evals_array::<VAL_FINAL_SUMCHECK_DEGREE_BOUND>(
                        j,
                        BindingOrder::LowToHigh,
                    );
                let wa_evals = self
                    .wa
                    .sumcheck_evals_array::<VAL_FINAL_SUMCHECK_DEGREE_BOUND>(
                        j,
                        BindingOrder::LowToHigh,
                    );
                [
                    inc_evals[0].mul_unreduced::<9>(wa_evals[0]),
                    inc_evals[1].mul_unreduced::<9>(wa_evals[1]),
                ]
            })
            .reduce(
                || [F::Unreduced::zero(); VAL_FINAL_SUMCHECK_DEGREE_BOUND],
                |running, new| [running[0] + new[0], running[1] + new[1]],
            )
            .map(F::from_montgomery_reduce);

        UniPoly::from_evals_and_hint(previous_claim, &evals)
    }

    #[tracing::instrument(skip_all, name = "ValFinalSumcheckProver::ingest_challenge")]
    fn ingest_challenge(&mut self, r_j: F::Challenge, _: usize) {
        self.inc.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.wa.bind_parallel(r_j, BindingOrder::LowToHigh);
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let r_cycle_prime = self.params.normalize_opening_point(sumcheck_challenges);
        let r_address = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RamValFinal,
                SumcheckId::RamOutputCheck,
            )
            .0;
        let wa_opening_point = OpeningPoint::new([&*r_address.r, &*r_cycle_prime.r].concat());

        accumulator.append_dense(
            transcript,
            CommittedPolynomial::RamInc,
            SumcheckId::RamValFinalEvaluation,
            r_cycle_prime.r,
            self.inc.final_sumcheck_claim(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RamRa,
            SumcheckId::RamValFinalEvaluation,
            wa_opening_point,
            self.wa.final_sumcheck_claim(),
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

pub struct ValFinalSumcheckVerifier<F: JoltField> {
    params: ValFinalSumcheckParams<F>,
}

impl<F: JoltField> ValFinalSumcheckVerifier<F> {
    pub fn new(
        initial_ram_state: &[u64],
        program_io: &JoltDevice,
        trace_len: usize,
        ram_K: usize,
        opening_accumulator: &VerifierOpeningAccumulator<F>,
    ) -> Self {
        let params = ValFinalSumcheckParams::new_from_verifier(
            initial_ram_state,
            program_io,
            trace_len,
            ram_K,
            opening_accumulator,
        );
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for ValFinalSumcheckVerifier<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        _sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let inc_claim = accumulator
            .get_committed_polynomial_opening(
                CommittedPolynomial::RamInc,
                SumcheckId::RamValFinalEvaluation,
            )
            .1;
        let wa_claim = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RamRa,
                SumcheckId::RamValFinalEvaluation,
            )
            .1;
        inc_claim * wa_claim
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[<F as JoltField>::Challenge],
    ) {
        let r_cycle_prime = self.params.normalize_opening_point(sumcheck_challenges);
        let r_address = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RamValFinal,
                SumcheckId::RamOutputCheck,
            )
            .0;
        let wa_opening_point = OpeningPoint::new([&*r_address.r, &*r_cycle_prime.r].concat());

        accumulator.append_dense(
            transcript,
            CommittedPolynomial::RamInc,
            SumcheckId::RamValFinalEvaluation,
            r_cycle_prime.r,
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RamRa,
            SumcheckId::RamValFinalEvaluation,
            wa_opening_point,
        );
    }
}
