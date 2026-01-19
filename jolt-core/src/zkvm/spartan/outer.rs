use std::marker::PhantomData;
use std::sync::Arc;

use allocative::Allocative;
use ark_std::Zero;
use rayon::prelude::*;
use tracer::instruction::Cycle;

use crate::field::BarrettReduce;
use crate::field::{FMAdd, JoltField, MontgomeryReduce};
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::lagrange_poly::LagrangePolynomial;
use crate::poly::multilinear_polynomial::{BindingOrder, PolynomialBinding};
use crate::poly::multiquadratic_poly::MultiquadraticPolynomial;
use crate::poly::opening_proof::{
    OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
    VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
};
use crate::poly::split_eq_poly::GruenSplitEqPolynomial;
use crate::poly::unipoly::UniPoly;
use crate::subprotocols::streaming_sumcheck::{
    LinearSumcheckStage, SharedStreamingSumcheckState, StreamingSumcheck, StreamingSumcheckWindow,
};
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use crate::subprotocols::sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier};
use crate::subprotocols::univariate_skip::build_uniskip_first_round_poly;
use crate::transcripts::Transcript;
use crate::utils::accumulation::{Acc5U, Acc6S, Acc7S, Acc8S};
use crate::utils::expanding_table::ExpandingTable;
use crate::utils::math::Math;
#[cfg(feature = "allocative")]
use crate::utils::profiling::print_data_structure_heap_usage;
use crate::utils::thread::unsafe_allocate_zero_vec;
use crate::zkvm::bytecode::BytecodePreprocessing;
use crate::zkvm::r1cs::constraints::OUTER_FIRST_ROUND_POLY_DEGREE_BOUND;
use crate::zkvm::r1cs::key::UniformSpartanKey;
use crate::zkvm::r1cs::{
    constraints::{
        OUTER_FIRST_ROUND_POLY_NUM_COEFFS, OUTER_UNIVARIATE_SKIP_DEGREE,
        OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE, OUTER_UNIVARIATE_SKIP_EXTENDED_DOMAIN_SIZE,
    },
    evaluation::R1CSEval,
    inputs::{R1CSCycleInputs, ALL_R1CS_INPUTS},
};
use crate::zkvm::witness::VirtualPolynomial;

/// Degree bound of the sumcheck round polynomials for [`OuterRemainingSumcheckVerifier`].
const OUTER_REMAINING_DEGREE_BOUND: usize = 3;
// this represents the index position in multi-quadratic poly array
// This should actually be d where degree is the degree of the streaming data structure
// For example : MultiQuadratic has d=2; for cubic this would be 3 etc.
const INFINITY: usize = 2;

// Spartan Outer sumcheck
// (with univariate-skip first round on Z, and no Cz term given all eq conditional constraints)
//
// We define a univariate in Z first-round polynomial
//   s1(Y) := L(τ_high, Y) · Σ_{x_out ∈ {0,1}^{m_out}} Σ_{x_in ∈ {0,1}^{m_in}}
//              E_out(r_out, x_out) · E_in(r_in, x_in) ·
//              [ Az(x_out, x_in, Y) · Bz(x_out, x_in, Y) ],
// where L(τ_high, Y) is the Lagrange basis polynomial over the univariate-skip
// base domain evaluated at τ_high, and Az(·,·,Y), Bz(·,·,Y) are the
// per-row univariate polynomials in Y induced by the R1CS row (split into two
// internal groups in code, but algebraically composing to Az·Bz at Y).
// The prover sends s1(Y) via univariate-skip by evaluating t1(Y) := Σ Σ E_out·E_in·(Az·Bz)
// on an extended grid Y ∈ {−D..D} outside the base window, interpolating t1,
// multiplying by L(τ_high, Y) to obtain s1, and the verifier samples r0.
//
// Subsequent outer rounds bind the cycle variables r_tail = (r1, r2, …) using
// a streaming first cycle-bit round followed by linear-time rounds:
//   • Streaming round (after r0): compute
//       t(0)  = Σ_{x_out} E_out · Σ_{x_in} E_in · (Az(0)·Bz(0))
//       t(∞)  = Σ_{x_out} E_out · Σ_{x_in} E_in · ((Az(1)−Az(0))·(Bz(1)−Bz(0)))
//     send a cubic built from these endpoints, and bind cached coefficients by r1.
//   • Remaining rounds: reuse bound coefficients to compute the same endpoints
//     in linear time for each subsequent bit and bind by r_i.
//
// Final check (verifier): with r = [r0 || r_tail] and outer binding order from
// the top, evaluate Eq_τ(τ, r) and verify
//  L(τ_high, r_high) · Eq_τ(τ, r) · (Az(r) · Bz(r)).

#[derive(Allocative, Clone)]
pub struct OuterUniSkipParams<F: JoltField> {
    pub tau: Vec<F::Challenge>,
}

impl<F: JoltField> OuterUniSkipParams<F> {
    pub fn new<T: Transcript>(key: &UniformSpartanKey<F>, transcript: &mut T) -> Self {
        let num_rounds_x: usize = key.num_rows_bits();
        let tau = transcript.challenge_vector_optimized::<F>(num_rounds_x);
        Self { tau }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for OuterUniSkipParams<F> {
    fn degree(&self) -> usize {
        OUTER_FIRST_ROUND_POLY_DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        1
    }

    fn input_claim(&self, _: &dyn OpeningAccumulator<F>) -> F {
        F::zero()
    }

    fn normalize_opening_point(
        &self,
        challenges: &[<F as JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        challenges.to_vec().into()
    }
}

/// Uni-skip instance for Spartan outer sumcheck, computing the first-round polynomial only.
#[derive(Allocative)]
pub struct OuterUniSkipProver<F: JoltField> {
    /// Evaluations of t1(Z) at the extended univariate-skip targets (outside base window)
    extended_evals: [F; OUTER_UNIVARIATE_SKIP_DEGREE],
    /// Verifier challenge for this univariate skip round
    r0: Option<F::Challenge>,
    /// Prover message for this univariate skip round
    uni_poly: Option<UniPoly<F>>,
    pub params: OuterUniSkipParams<F>,
}

impl<F: JoltField> OuterUniSkipProver<F> {
    #[tracing::instrument(skip_all, name = "OuterUniSkipInstanceProver::initialize")]
    /// 初始化 OuterUniSkipInstance 证明者实例。
    ///
    /// 该函数是在 Sumcheck 协议开始前调用的，用于准备证明所需的数据结构。
    /// 它主要负责根据验证者提供的随机挑战点 `tau`，预先计算 Trace 多项式在这一点的评估值。
    ///
    /// # 参数
    ///
    /// * `params`: 包含协议参数，最重要的是 `tau` (随机挑战向量，通常来自上一轮 Sumcheck 或 Fiat-Shamir 变换)。
    /// * `trace`: 程序的执行轨迹 (Execution Trace)，包含了每一步的 CPU 状态 (Cycle)。
    /// * `bytecode_preprocessing`: 对程序字节码的静态预处理信息 (如指令解码信息)，用于辅助多项式评估。
    ///
    /// # 返回值
    ///
    /// 返回一个初始化的 `Self` (即 `OuterUniSkipInstance`)，包含后续 Sumcheck 步骤所需的所有状态。
    pub fn initialize(
        params: OuterUniSkipParams<F>,
        trace: &[Cycle],
        bytecode_preprocessing: &BytecodePreprocessing,
    ) -> Self {
        // 核心步骤：计算扩展评估值 (Extended Evaluations)。
        // 算法逻辑：
        // 这里的 `tau` 是一个多线性扩展的随机挑战点（Random Challenge Point）。
        // 函数会根据 Bytecode 信息和动态的 Trace 数据，计算出相关多项式（可能是由 Trace 列构成的多项式）
        // 在该随机点 `tau` 上的值 (或者其某种形式的投影/扩展)。
        // 这些值是后续进行线性时间 Sumcheck (Linear-time Sumcheck) 的基础。
        // "Univariate Skip" 暗示这里可能涉及对 Trace 中某些行或逻辑的跳过处理，或者是一种基于单变量多项式的优化技术。
        let extended = Self::compute_univariate_skip_extended_evals(
            bytecode_preprocessing, // 用于确定每行 Trace 对应的具体计算逻辑（指令行为）
            trace,                  // 实际的 Witness 数据
            &params.tau,            // 随机评估点
        );

        // 构建实例结构体
        let instance = Self {
            params,                  // 保存参数以便后续轮次使用
            extended_evals: extended,// 保存计算好的评估值，Sumcheck 过程中会不断折叠(fold)这些值
            r0: None,                // 初始化状态，r0 可能用于存储某一轮的随机数，此时尚未生成
            uni_poly: None,          // 初始化状态，缓存当前轮次生成的单变量多项式
        };

        // 调试/性能分析：
        // 如果开启了 "allocative" 特性，则打印该结构体的堆内存使用情况。
        // 这对于优化大规模 ZK 证明器的内存消耗非常重要。
        #[cfg(feature = "allocative")]
        print_data_structure_heap_usage("OuterUniSkipInstance", &instance);

        instance
    }

    /// Compute the extended evaluations of the univariate skip polynomial, i.e.
    ///
    /// t_1(y) = \sum_{x_out} eq(tau_out, x_out) * \sum_{x_in} eq(tau_in, x_in) * Az(x_out, x_in, y) * Bz(x_out, x_in, y)
    ///
    /// for all y in the extended domain {−D..D} outside the base window
    /// (inside the base window, we have t_1(y) = 0)
    ///
    /// Note that the last of the x_in variables corresponds to the group index of the constraints
    /// (since we split the constraints in half, and y ranges over the number of constraints in each group)
    ///
    /// So we actually need to be careful and compute
    ///
    /// \sum_{x_in'} eq(tau_in, (x_in', 0)) * Az(x_out, x_in', 0, y) * Bz(x_out, x_in', 0, y)
    ///     + eq(tau_in, (x_in', 1)) * Az(x_out, x_in', 1, y) * Bz(x_out, x_in', 1, y)
    /// 计算单变量跳跃多项式（univariate skip polynomial）在扩展域上的评估值。
    ///
    /// 目标是计算：
    /// t_1(y) = \sum_{x_out} eq(tau_out, x_out) * \sum_{x_in} eq(tau_in, x_in) * Az(x_out, x_in, y) * Bz(x_out, x_in, y)
    ///
    /// 其中：
    /// - y: 在扩展域 {−D..D} 中的点（基窗口之外）。
    /// - x_in 的最后一个比特对应约束的分组（Group Index）。这意味着我们将约束分为了两半。
    /// - x_in 的其余比特与 x_out 组合，对应实际的 Trace 步骤（Step Index）。
    ///
    /// 具体的计算逻辑如下：
    /// \sum_{x_in'} eq(tau_in, (x_in', 0)) * Az(x_out, x_in', 0, y) * Bz(x_out, x_in', 0, y)
    ///     + eq(tau_in, (x_in', 1)) * Az(x_out, x_in', 1, y) * Bz(x_out, x_in', 1, y)
    fn compute_univariate_skip_extended_evals(
        bytecode_preprocessing: &BytecodePreprocessing, // 预处理数据（包含矩阵 A, B, C 的结构信息）
        trace: &[Cycle],                                // 执行轨迹（CPU 每一周期的状态）
        tau: &[F::Challenge],                           // Sum-Check 的随机挑战点向量
    ) -> [F; OUTER_UNIVARIATE_SKIP_DEGREE] {            // 返回值：多项式在特定点（通常是 0, 1, ...）的评估值

        // -------------------------------------------------------------------
        // 1. 初始化 Gruen 分裂 Eq 多项式生成器，建立缓存预计算表
        // -------------------------------------------------------------------
        // 原理：eq(τ, x) 可以分解为 eq(τ_high, x_high) * eq(τ_low, x_low)。

        let split_eq = GruenSplitEqPolynomial::<F>::new_with_scaling(
            tau,
            BindingOrder::LowToHigh,
            Some(F::MONTGOMERY_R_SQUARE), // 初始乘子设为 R^2，用于抵消蒙哥马利约简的因子
        );

        // 获取当前的外部缩放因子（此处初始为 R^2，用于修正后续蒙哥马利约简的系数）
        let outer_scale = split_eq.get_current_scalar();

        // -------------------------------------------------------------------
        // 2. 计算维度与并行参数
        // -------------------------------------------------------------------
        // num_x_in_bits: "In" (内层循环) 的比特数
        let num_x_in_bits = split_eq.E_in_current_len().log_2();

        // num_x_in_prime_bits: 真实的 Trace 索引在 "In" 部分占用的比特数。
        // 关键点：因为 Jolt 将约束分为两组，使用索引的最低位 (LSB) 作为选择位。
        // 所以，实际映射到 Trace 行号的比特数需要减 1。
        let num_x_in_prime_bits = num_x_in_bits.saturating_sub(1);

        // -------------------------------------------------------------------
        // 3. 并行折叠 (Parallel Fold) - 核心计算循环
        // -------------------------------------------------------------------

        split_eq
            .par_fold_out_in(
                // A. 初始化累加器：每个线程拥有一个全零数组
                || [Acc8S::<F>::zero(); OUTER_UNIVARIATE_SKIP_DEGREE],

                // B. 核心折叠逻辑 (Folder)
                // inner: 当前线程的累加器
                // g: 全局索引 (对应 x 的高位部分)
                // x_in: 局部索引 (对应 x 的低位部分)
                // e_in: 预计算好的 eq(τ_in, x_in) 的值
                |inner, g, x_in, e_in| {
                    // --- 步骤 I: 索引解码 (Mapping) ---

                    // x_out 是高位索引
                    let x_out = g >> num_x_in_bits;

                    // x_in_prime 是低位索引中属于 "行号" 的部分
                    // ">> 1" 操作移除了最低位 (LSB)，该位用于 Group 选择
                    let x_in_prime = x_in >> 1;

                    // 拼接高位和低位，算出当前是在处理 Trace 的第几行 (Cycle Index)
                    let base_step_idx = (x_out << num_x_in_prime_bits) | x_in_prime;

                    // --- 步骤 II: 实时计算 Az, Bz (On-the-fly) ---

                    // 根据行号 base_step_idx，从 Trace 中读取该周期的寄存器/内存值
                    // 这重建了 R1CS 的输入向量 z (大小为 Trace 列数相关)
                    let row_inputs = R1CSCycleInputs::from_trace::<F>(
                        bytecode_preprocessing,
                        trace,
                        base_step_idx,
                    );

                    // 构建评估器，准备计算矩阵乘法
                    let eval = R1CSEval::<F>::from_cycle_inputs(&row_inputs);

                    // --- 步骤 III: 分组选择 (Group Selection) ---

                    // 检查 x_in 的最低位 (LSB)。
                    // 1 -> Group 1 (第二组约束)
                    // 0 -> Group 0 (第一组约束)
                    let is_group1 = (x_in & 1) == 1;

                    // --- 步骤 IV: 累加多项式评估值 ---

                    // 遍历扩展域上的点 j (通常 j=0..degree)
                    for j in 0..OUTER_UNIVARIATE_SKIP_DEGREE {
                        // 计算 Az(j) * Bz(j)。
                        // Jolt 为了优化内存布局，将约束矩阵切分成了 First Group 和 Second Group。
                        let prod_s192 = if !is_group1 {
                            eval.extended_azbz_product_first_group(j)
                        } else {
                            eval.extended_azbz_product_second_group(j)
                        };

                        // 执行加权累加： inner[j] += e_in * prod
                        // e_in 是当前行的 eq 多项式权重
                        inner[j].fmadd(&e_in, &prod_s192);
                    }
                },

                // C. 归约器 (Reducer): 处理外层权重 eq(τ_out, x_out)
                |_x_out, e_out, inner| {
                    let mut out = [F::Unreduced::<9>::zero(); OUTER_UNIVARIATE_SKIP_DEGREE];
                    for j in 0..OUTER_UNIVARIATE_SKIP_DEGREE {
                        // 蒙哥马利约简，将累加器转回标准域
                        let reduced = inner[j].montgomery_reduce();
                        // 乘以外部权重 e_out
                        out[j] = e_out.mul_unreduced::<9>(reduced);
                    }
                    out
                },

                // D. 合并器 (Consumer): 将所有并行的结果加在一起
                |mut a, b| {
                    for j in 0..OUTER_UNIVARIATE_SKIP_DEGREE {
                        a[j] += b[j];
                    }
                    a
                },
            )
            // 4. 最终修正：乘以初始缩放因子
            .map(|x| F::from_montgomery_reduce::<9>(x) * outer_scale)



    }
    fn compute_univariate_skip_extended_evals1(
        bytecode_preprocessing: &BytecodePreprocessing,
        trace: &[Cycle],
        tau: &[F::Challenge],
    ) -> [F; OUTER_UNIVARIATE_SKIP_DEGREE] {
        // 1. 初始化 split_eq
        let split_eq = GruenSplitEqPolynomial::<F>::new_with_scaling(
            tau,
            BindingOrder::LowToHigh,
            Some(F::MONTGOMERY_R_SQUARE),
        );
        let outer_scale = split_eq.get_current_scalar();
        let num_x_in_bits = split_eq.E_in_current_len().log_2();
        let num_x_in_prime_bits = num_x_in_bits.saturating_sub(1);
        let (e_out_slice, e_in_slice) = split_eq.E_out_in_for_window(num_x_in_bits);

        // [修复]: 使用 Acc8S 累加器替代普通的 F 数组
        // Acc8S 专门用于处理 Field * SignedInt 的累加
        let mut sum_acc = [Acc8S::<F>::zero(); OUTER_UNIVARIATE_SKIP_DEGREE];

        // 2. 双层循环遍历
        for (x_out, &e_out_val) in e_out_slice.iter().enumerate() {
            for (x_in, &e_in_val) in e_in_slice.iter().enumerate() {
                // 计算 Trace 行号
                let x_in_prime = x_in >> 1;
                let base_step_idx = (x_out << num_x_in_prime_bits) | x_in_prime;

                // 恢复 Trace 数据
                let row_inputs = R1CSCycleInputs::from_trace::<F>(
                    bytecode_preprocessing,
                    trace,
                    base_step_idx,
                );
                let eval = R1CSEval::<F>::from_cycle_inputs(&row_inputs);
                let is_group1 = (x_in & 1) == 1;

                // 计算当前项的总权重 (F 类型)
                let weight = e_out_val * e_in_val;

                for j in 0..OUTER_UNIVARIATE_SKIP_DEGREE {
                    // prod 是 SignedBigInt<3> 类型 (192-bit signed integer)
                    let prod = if !is_group1 {
                        eval.extended_azbz_product_first_group(j)
                    } else {
                        eval.extended_azbz_product_second_group(j)
                    };

                    // [修复]: 使用 fmadd 来执行: sum_acc += weight * prod
                    sum_acc[j].fmadd(&weight, &prod);
                }
            }
        }

        // 3. 最终结果转换：将累加器归约为 F 并应用缩放
        let mut sum = [F::zero(); OUTER_UNIVARIATE_SKIP_DEGREE];
        for (j, acc) in sum_acc.iter().enumerate() {
            sum[j] = acc.montgomery_reduce() * outer_scale;
        }
        sum
    }
    
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for OuterUniSkipProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(skip_all, name = "OuterUniSkipInstanceProver::compute_poly")]
    fn compute_message(&mut self, _round: usize, _previous_claim: F) -> UniPoly<F> {
        // Load extended univariate-skip evaluations from prover state
        let extended_evals = &self.extended_evals;

        let tau_high = self.params.tau[self.params.tau.len() - 1];

        // Compute the univariate-skip first round polynomial s1(Y) = L(τ_high, Y) · t1(Y)
        let uni_poly = build_uniskip_first_round_poly::<
            F,
            OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE,
            OUTER_UNIVARIATE_SKIP_DEGREE,
            OUTER_UNIVARIATE_SKIP_EXTENDED_DOMAIN_SIZE,
            OUTER_FIRST_ROUND_POLY_NUM_COEFFS,
        >(None, extended_evals, tau_high);

        self.uni_poly = Some(uni_poly.clone());
        uni_poly
    }

    fn ingest_challenge(&mut self, _: F::Challenge, _round: usize) {
        // Nothing to do
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        debug_assert_eq!(opening_point.len(), 1);
        let claim = self.uni_poly.as_ref().unwrap().evaluate(&opening_point[0]);

        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::UnivariateSkip,
            SumcheckId::SpartanOuter,
            opening_point,
            claim,
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

pub struct OuterUniSkipVerifier<F: JoltField> {
    pub params: OuterUniSkipParams<F>,
}

impl<F: JoltField> OuterUniSkipVerifier<F> {
    pub fn new<T: Transcript>(key: &UniformSpartanKey<F>, transcript: &mut T) -> Self {
        let params = OuterUniSkipParams::new(key, transcript);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for OuterUniSkipVerifier<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        _accumulator: &VerifierOpeningAccumulator<F>,
        _sumcheck_challenges: &[<F as JoltField>::Challenge],
    ) -> F {
        unimplemented!("Unused for univariate skip")
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[<F as JoltField>::Challenge],
    ) {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        debug_assert_eq!(opening_point.len(), 1);
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::UnivariateSkip,
            SumcheckId::SpartanOuter,
            opening_point,
        );
    }
}

pub struct OuterRemainingSumcheckParams<F: JoltField> {
    /// Number of cycle bits for splitting opening points (consistent across prover/verifier)
    /// Total number of rounds is `1 + num_cycles_bits`
    pub num_cycles_bits: usize,
    /// Verifier challenge for univariate skip round
    pub r0: F::Challenge,
    /// The tau vector (length 1 + n_cycle_vars), available to prover and verifier
    pub tau: Vec<F::Challenge>,
}

impl<F: JoltField> OuterRemainingSumcheckParams<F> {
    pub fn new(
        trace_len: usize,
        uni_skip_params: OuterUniSkipParams<F>,
        opening_accumulator: &dyn OpeningAccumulator<F>,
    ) -> Self {
        let (r_uni_skip, _) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::UnivariateSkip,
            SumcheckId::SpartanOuter,
        );
        debug_assert_eq!(r_uni_skip.len(), 1);
        let r0 = r_uni_skip[0];

        Self {
            num_cycles_bits: trace_len.log_2(),
            tau: uni_skip_params.tau,
            r0,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for OuterRemainingSumcheckParams<F> {
    fn num_rounds(&self) -> usize {
        1 + self.num_cycles_bits
    }

    fn normalize_opening_point(
        &self,
        challenges: &[<F as JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        let r_cycle = challenges[1..].to_vec();
        OpeningPoint::<LITTLE_ENDIAN, F>::new(r_cycle).match_endianness()
    }

    fn degree(&self) -> usize {
        OUTER_REMAINING_DEGREE_BOUND
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let (_, uni_skip_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::UnivariateSkip,
            SumcheckId::SpartanOuter,
        );
        uni_skip_claim
    }
}

pub struct OuterRemainingSumcheckVerifier<F: JoltField> {
    params: OuterRemainingSumcheckParams<F>,
    key: UniformSpartanKey<F>,
}

impl<F: JoltField> OuterRemainingSumcheckVerifier<F> {
    pub fn new(
        key: UniformSpartanKey<F>,
        trace_len: usize,
        uni_skip_params: OuterUniSkipParams<F>,
        opening_accumulator: &VerifierOpeningAccumulator<F>,
    ) -> Self {
        let params =
            OuterRemainingSumcheckParams::new(trace_len, uni_skip_params, opening_accumulator);
        Self { params, key }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
for OuterRemainingSumcheckVerifier<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let r1cs_input_evals = ALL_R1CS_INPUTS.map(|input| {
            accumulator
                .get_virtual_polynomial_opening((&input).into(), SumcheckId::SpartanOuter)
                .1
        });

        // Randomness used to bind the rows of R1CS matrices A,B.
        let rx_constr = &[sumcheck_challenges[0], self.params.r0];
        // Compute sum_y A(rx_constr, y)*z(y) * sum_y B(rx_constr, y)*z(y).
        let inner_sum_prod = self
            .key
            .evaluate_inner_sum_product_at_point(rx_constr, r1cs_input_evals);

        let tau = &self.params.tau;
        let tau_high = &tau[tau.len() - 1];
        let tau_high_bound_r0 = LagrangePolynomial::<F>::lagrange_kernel::<
            F::Challenge,
            OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE,
        >(tau_high, &self.params.r0);
        let tau_low = &tau[..tau.len() - 1];
        let r_tail_reversed: Vec<F::Challenge> =
            sumcheck_challenges.iter().rev().copied().collect();
        let tau_bound_r_tail_reversed = EqPolynomial::mle(tau_low, &r_tail_reversed);
        tau_high_bound_r0 * tau_bound_r_tail_reversed * inner_sum_prod
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let r_cycle = self.params.normalize_opening_point(sumcheck_challenges);
        for input in &ALL_R1CS_INPUTS {
            accumulator.append_virtual(
                transcript,
                VirtualPolynomial::from(input),
                SumcheckId::SpartanOuter,
                r_cycle.clone(),
            );
        }
    }
}

#[derive(Allocative, Clone)]
pub struct OuterStreamingProverParams<F: JoltField> {
    /// Number of cycle bits for splitting opening points
    /// Total number of rounds equals num_cycles_bits
    pub num_cycles_bits: usize,
    /// The univariate-skip first round challenge
    pub r0_uniskip: F::Challenge,
}

impl<F: JoltField> OuterStreamingProverParams<F> {
    fn new(
        uni_skip_params: &OuterUniSkipParams<F>,
        opening_accumulator: &ProverOpeningAccumulator<F>,
    ) -> Self {
        let (r_uni_skip, _) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::UnivariateSkip,
            SumcheckId::SpartanOuter,
        );
        debug_assert_eq!(r_uni_skip.len(), 1);
        // tau.len() = num_rows_bits() = num_cycle_vars + 2
        // num_cycles_bits = num_cycle_vars = tau.len() - 2
        Self {
            num_cycles_bits: uni_skip_params.tau.len() - 2,
            r0_uniskip: r_uni_skip[0],
        }
    }

    fn num_rounds(&self) -> usize {
        // Total rounds = 1 + num_cycles_bits (one extra for streaming window)
        1 + self.num_cycles_bits
    }

    fn get_inputs_opening_point(
        sumcheck_challenges: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        let r_cycle = sumcheck_challenges[1..].to_vec();
        OpeningPoint::<LITTLE_ENDIAN, F>::new(r_cycle).match_endianness()
    }
}

pub type OuterRemainingStreamingSumcheck<F, S> =
StreamingSumcheck<F, S, OuterSharedState<F>, OuterStreamingWindow<F>, OuterLinearStage<F>>;

#[derive(Allocative)]
pub struct OuterSharedState<F: JoltField> {
    #[allocative(skip)]
    bytecode_preprocessing: BytecodePreprocessing,
    #[allocative(skip)]
    trace: Arc<Vec<Cycle>>,
    split_eq_poly: GruenSplitEqPolynomial<F>,
    t_prime_poly: Option<MultiquadraticPolynomial<F>>,
    r_grid: ExpandingTable<F>,
    #[allocative(skip)]
    lagrange_evals_r0: [F; OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE],
    pub params: OuterStreamingProverParams<F>,
}

impl<F: JoltField> OuterSharedState<F> {
    #[tracing::instrument(skip_all, name = "OuterSharedState::new")]
    pub fn new(
        trace: Arc<Vec<Cycle>>,
        bytecode_preprocessing: &BytecodePreprocessing,
        uni_skip_params: &OuterUniSkipParams<F>,
        opening_accumulator: &ProverOpeningAccumulator<F>,
    ) -> Self {
        let bytecode_preprocessing = bytecode_preprocessing.clone();
        let outer_params = OuterStreamingProverParams::new(uni_skip_params, opening_accumulator);
        let r0 = outer_params.r0_uniskip;

        let lagrange_evals_r =
            LagrangePolynomial::<F>::evals::<F::Challenge, OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE>(&r0);

        let tau_high = uni_skip_params.tau[uni_skip_params.tau.len() - 1];
        let tau_low = &uni_skip_params.tau[..uni_skip_params.tau.len() - 1];

        let lagrange_tau_r0 = LagrangePolynomial::<F>::lagrange_kernel::<
            F::Challenge,
            OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE,
        >(&r0, &tau_high);

        let split_eq_poly: GruenSplitEqPolynomial<F> =
            GruenSplitEqPolynomial::<F>::new_with_scaling(
                tau_low,
                BindingOrder::LowToHigh,
                Some(lagrange_tau_r0),
            );

        let n_cycle_vars = outer_params.num_cycles_bits;
        let mut r_grid = ExpandingTable::new(1 << n_cycle_vars, BindingOrder::LowToHigh);
        r_grid.reset(F::one());

        Self {
            split_eq_poly,
            bytecode_preprocessing,
            trace,
            t_prime_poly: None,
            r_grid,
            params: outer_params,
            lagrange_evals_r0: lagrange_evals_r,
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[tracing::instrument(
        skip_all,
        name = "OuterSharedState::extrapolate_from_binary_grid_to_tertiary_grid"
    )]
    fn extrapolate_from_binary_grid_to_tertiary_grid(
        &self,
        acc_az: &mut [Acc5U<F>],
        acc_bz_first: &mut [Acc6S<F>],
        acc_bz_second: &mut [Acc7S<F>],
        grid_az: &mut [F],
        grid_bz: &mut [F],
        jlen: usize,
        klen: usize,
        offset: usize,
        scaled_w: &[[F; OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE]],
    ) {
        let preprocess = &self.bytecode_preprocessing;
        let trace = &self.trace;
        debug_assert_eq!(scaled_w.len(), klen);
        debug_assert_eq!(grid_az.len(), jlen);
        debug_assert_eq!(grid_bz.len(), jlen);
        debug_assert_eq!(acc_az.len(), jlen);
        debug_assert_eq!(acc_bz_first.len(), jlen);
        debug_assert_eq!(acc_bz_second.len(), jlen);

        acc_az
            .par_iter_mut()
            .zip(acc_bz_first.par_iter_mut())
            .zip(acc_bz_second.par_iter_mut())
            .for_each(|((a, b), c)| {
                *a = Acc5U::zero();
                *b = Acc6S::zero();
                *c = Acc7S::zero();
            });

        acc_az
            .par_iter_mut()
            .zip(acc_bz_first.par_iter_mut())
            .zip(acc_bz_second.par_iter_mut())
            .enumerate()
            .for_each(|(j, ((acc_az_j, acc_bz_first_j), acc_bz_second_j))| {
                for k in 0..klen {
                    let full_idx = offset + j * klen + k;
                    let current_step_idx = full_idx >> 1;
                    let selector = (full_idx & 1) == 1;

                    let row_inputs =
                        R1CSCycleInputs::from_trace::<F>(preprocess, trace, current_step_idx);
                    let eval = R1CSEval::<F>::from_cycle_inputs(&row_inputs);
                    let w_k = &scaled_w[k];

                    if !selector {
                        eval.fmadd_first_group_at_r(w_k, acc_az_j, acc_bz_first_j);
                    } else {
                        eval.fmadd_second_group_at_r(w_k, acc_az_j, acc_bz_second_j);
                    }
                }
            });

        const REDUCE_CHUNK_SIZE: usize = 4096;
        grid_az
            .par_chunks_mut(REDUCE_CHUNK_SIZE)
            .zip(grid_bz.par_chunks_mut(REDUCE_CHUNK_SIZE))
            .enumerate()
            .for_each(|(chunk_idx, (az_chunk, bz_chunk))| {
                let start = chunk_idx * REDUCE_CHUNK_SIZE;
                for (local_j, (az_out, bz_out)) in
                    az_chunk.iter_mut().zip(bz_chunk.iter_mut()).enumerate()
                {
                    let j = start + local_j;
                    *az_out = acc_az[j].barrett_reduce();
                    let bz_first_j = acc_bz_first[j].barrett_reduce();
                    let bz_second_j = acc_bz_second[j].barrett_reduce();
                    *bz_out = bz_first_j + bz_second_j;
                }
            });
    }

    #[tracing::instrument(
        skip_all,
        name = "OuterSharedState::compute_evaluation_grid_from_trace"
    )]
    pub fn compute_evaluation_grid_from_trace(&mut self, window_size: usize) {
        let split_eq = &self.split_eq_poly;

        let three_pow_dim = 3_usize.pow(window_size as u32);
        let jlen = 1 << window_size;
        let klen = 1 << split_eq.num_challenges();

        let lagrange_evals_r = &self.lagrange_evals_r0;
        let r_grid = &self.r_grid;
        let scaled_w: Vec<[F; OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE]> = if klen > 1 {
            debug_assert_eq!(klen, r_grid.len());
            (0..klen)
                .into_par_iter()
                .map(|k| {
                    let weight = r_grid[k];
                    let mut row = [F::zero(); OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE];
                    for t in 0..OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE {
                        row[t] = lagrange_evals_r[t] * weight;
                    }
                    row
                })
                .collect()
        } else {
            debug_assert_eq!(klen, 1);
            let mut row = [F::zero(); OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE];
            row.copy_from_slice(lagrange_evals_r);
            vec![row]
        };

        let (e_out, e_in) = split_eq.E_out_in_for_window(window_size);
        let e_in_len = e_in.len();

        let res_unr = e_out
            .par_iter()
            .enumerate()
            .map(|(out_idx, out_val)| {
                let mut local_res_unr = vec![F::Unreduced::<9>::zero(); three_pow_dim];
                let mut buff_a: Vec<F> = vec![F::zero(); three_pow_dim];
                let mut buff_b = vec![F::zero(); three_pow_dim];
                let mut tmp = vec![F::zero(); three_pow_dim];
                let mut grid_a = vec![F::zero(); jlen];
                let mut grid_b = vec![F::zero(); jlen];
                let mut acc_az = vec![Acc5U::<F>::zero(); jlen];
                let mut acc_bz_first = vec![Acc6S::<F>::zero(); jlen];
                let mut acc_bz_second = vec![Acc7S::<F>::zero(); jlen];

                for (in_idx, in_val) in e_in.iter().enumerate() {
                    let i = out_idx * e_in_len + in_idx;

                    grid_a.fill(F::zero());
                    grid_b.fill(F::zero());
                    self.extrapolate_from_binary_grid_to_tertiary_grid(
                        &mut acc_az,
                        &mut acc_bz_first,
                        &mut acc_bz_second,
                        &mut grid_a,
                        &mut grid_b,
                        jlen,
                        klen,
                        i * jlen * klen,
                        &scaled_w,
                    );

                    MultiquadraticPolynomial::<F>::expand_linear_grid_to_multiquadratic(
                        &grid_a,
                        &mut buff_a,
                        &mut tmp,
                        window_size,
                    );
                    MultiquadraticPolynomial::<F>::expand_linear_grid_to_multiquadratic(
                        &grid_b,
                        &mut buff_b,
                        &mut tmp,
                        window_size,
                    );

                    let e_in_val = *in_val;
                    if window_size == 1 {
                        local_res_unr[0] += e_in_val.mul_unreduced::<9>(buff_a[0] * buff_b[0]);
                        local_res_unr[2] += e_in_val.mul_unreduced::<9>(buff_a[2] * buff_b[2]);
                    } else {
                        for idx in 0..three_pow_dim {
                            let val = buff_a[idx] * buff_b[idx];
                            local_res_unr[idx] += e_in_val.mul_unreduced::<9>(val);
                        }
                    }
                }

                let e_out_val = *out_val;
                for idx in 0..three_pow_dim {
                    let inner_red = F::from_montgomery_reduce::<9>(local_res_unr[idx]);
                    local_res_unr[idx] = e_out_val.mul_unreduced::<9>(inner_red);
                }
                local_res_unr
            })
            .reduce(
                || vec![F::Unreduced::<9>::zero(); three_pow_dim],
                |mut acc, local| {
                    for idx in 0..three_pow_dim {
                        acc[idx] += local[idx];
                    }
                    acc
                },
            );

        let res: Vec<F> = res_unr
            .into_iter()
            .map(|unr| F::from_montgomery_reduce::<9>(unr))
            .collect();
        self.t_prime_poly = Some(MultiquadraticPolynomial::new(window_size, res));
    }

    #[tracing::instrument(skip_all, name = "OuterSharedState::compute_t_evals")]
    pub fn compute_t_evals(&self, window_size: usize) -> (F, F) {
        let t_prime_poly = self
            .t_prime_poly
            .as_ref()
            .expect("t_prime_poly should be initialized");

        let e_active = self.split_eq_poly.E_active_for_window(window_size);
        let t_prime_0 = t_prime_poly.project_to_first_variable(&e_active, 0);
        let t_prime_inf = t_prime_poly.project_to_first_variable(&e_active, INFINITY);
        (t_prime_0, t_prime_inf)
    }
}

impl<F: JoltField> SharedStreamingSumcheckState<F> for OuterSharedState<F> {
    fn degree(&self) -> usize {
        OUTER_REMAINING_DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.params.num_rounds()
    }

    fn input_claim(&self, accumulator: &ProverOpeningAccumulator<F>) -> F {
        let (_, uni_skip_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::UnivariateSkip,
            SumcheckId::SpartanOuter,
        );
        uni_skip_claim
    }
}

#[derive(Allocative)]
#[allocative(bound = "")]
pub struct OuterStreamingWindow<F: JoltField> {
    _phantom: PhantomData<F>,
}

impl<F: JoltField> StreamingSumcheckWindow<F> for OuterStreamingWindow<F> {
    type Shared = OuterSharedState<F>;

    #[tracing::instrument(skip_all, name = "OuterStreamingWindow::initialize")]
    fn initialize(shared: &mut Self::Shared, window_size: usize) -> Self {
        shared.compute_evaluation_grid_from_trace(window_size);
        Self {
            _phantom: PhantomData,
        }
    }

    #[tracing::instrument(skip_all, name = "OuterStreamingWindow::compute_message")]
    fn compute_message(
        &self,
        shared: &Self::Shared,
        window_size: usize,
        previous_claim: F,
    ) -> UniPoly<F> {
        let (t_prime_0, t_prime_inf) = shared.compute_t_evals(window_size);
        shared
            .split_eq_poly
            .gruen_poly_deg_3(t_prime_0, t_prime_inf, previous_claim)
    }

    #[tracing::instrument(skip_all, name = "OuterStreamingWindow::ingest_challenge")]
    fn ingest_challenge(&mut self, shared: &mut Self::Shared, r_j: F::Challenge, _round: usize) {
        shared.split_eq_poly.bind(r_j);

        if let Some(t_prime_poly) = shared.t_prime_poly.as_mut() {
            t_prime_poly.bind(r_j, BindingOrder::LowToHigh);
        }

        shared.r_grid.update(r_j);
    }
}

#[derive(Allocative)]
pub struct OuterLinearStage<F: JoltField> {
    az: DensePolynomial<F>,
    bz: DensePolynomial<F>,
}

impl<F: JoltField> OuterLinearStage<F> {
    #[tracing::instrument(
        skip_all,
        name = "OuterLinearStage::fused_materialise_polynomials_general_with_multiquadratic"
    )]
    fn fused_materialise_polynomials_general_with_multiquadratic(
        shared: &mut OuterSharedState<F>,
        window_size: usize,
    ) -> (DensePolynomial<F>, DensePolynomial<F>) {
        let (E_out, E_in) = shared.split_eq_poly.E_out_in_for_window(window_size);
        let num_x_out_vals = E_out.len();
        let num_x_in_vals = E_in.len();
        let r_grid = &shared.r_grid;
        let num_r_vals = r_grid.len();

        let three_pow_dim = 3_usize.pow(window_size as u32);
        let grid_size = 1 << window_size;
        let num_evals_az = E_out.len() * E_in.len() * grid_size;

        let mut az_bound: Vec<F> = unsafe_allocate_zero_vec(num_evals_az);
        let mut bz_bound: Vec<F> = unsafe_allocate_zero_vec(num_evals_az);

        let num_r_bits = num_r_vals.log_2();
        let num_x_in_bits = num_x_in_vals.log_2();

        let lagrange_evals_r = &shared.lagrange_evals_r0;
        let scaled_w: Vec<[F; OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE]> = (0..num_r_vals)
            .into_par_iter()
            .map(|r_idx| {
                let weight = r_grid[r_idx];
                let mut row = [F::zero(); OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE];
                for t in 0..OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE {
                    row[t] = lagrange_evals_r[t] * weight;
                }
                row
            })
            .collect();

        let output_size = num_x_out_vals * num_x_in_vals;

        let num_threads = rayon::current_num_threads();
        let target_chunks = num_threads * 4;
        let min_chunk_pairs = 16;
        let pairs_per_chunk = output_size.div_ceil(target_chunks).max(min_chunk_pairs);
        let chunk_size = pairs_per_chunk * grid_size;

        let ans = az_bound
            .par_chunks_mut(chunk_size)
            .zip(bz_bound.par_chunks_mut(chunk_size))
            .enumerate()
            .fold(
                || vec![F::zero(); three_pow_dim],
                |mut local_ans, (chunk_idx, (az_chunk, bz_chunk))| {
                    let start_pair = chunk_idx * pairs_per_chunk;
                    let end_pair = (start_pair + pairs_per_chunk).min(output_size);

                    let mut buff_a = vec![F::zero(); three_pow_dim];
                    let mut buff_b = vec![F::zero(); three_pow_dim];
                    let mut tmp = vec![F::zero(); three_pow_dim];
                    let mut az_grid = vec![F::zero(); grid_size];
                    let mut bz_grid = vec![F::zero(); grid_size];

                    let mut acc_az: Vec<Acc5U<F>> = vec![Acc5U::zero(); grid_size];
                    let mut acc_bz_first: Vec<Acc6S<F>> = vec![Acc6S::zero(); grid_size];
                    let mut acc_bz_second: Vec<Acc7S<F>> = vec![Acc7S::zero(); grid_size];

                    let mut inner_sum: Vec<F::Unreduced<9>> =
                        vec![F::Unreduced::<9>::zero(); three_pow_dim];
                    let mut current_x_out = start_pair / num_x_in_vals;

                    for pair_idx in start_pair..end_pair {
                        let x_in_val = pair_idx % num_x_in_vals;
                        let x_out_val = pair_idx / num_x_in_vals;

                        if x_out_val != current_x_out {
                            let e_out = E_out[current_x_out];
                            for idx in 0..three_pow_dim {
                                local_ans[idx] +=
                                    F::from_montgomery_reduce::<9>(inner_sum[idx]) * e_out;
                                inner_sum[idx] = F::Unreduced::<9>::zero();
                            }
                            current_x_out = x_out_val;
                        }

                        for x_val in 0..grid_size {
                            acc_az[x_val] = Acc5U::zero();
                            acc_bz_first[x_val] = Acc6S::zero();
                            acc_bz_second[x_val] = Acc7S::zero();
                        }

                        let base_idx = (x_out_val << (num_x_in_bits + window_size + num_r_bits))
                            | (x_in_val << (window_size + num_r_bits));

                        for x_val in 0..grid_size {
                            let x_val_shifted = x_val << num_r_bits;
                            for r_idx in 0..num_r_vals {
                                let w_r = &scaled_w[r_idx];
                                let full_idx = base_idx | x_val_shifted | r_idx;

                                let step_idx = full_idx >> 1;
                                let selector = (full_idx & 1) == 1;

                                let row_inputs = R1CSCycleInputs::from_trace::<F>(
                                    &shared.bytecode_preprocessing,
                                    &shared.trace,
                                    step_idx,
                                );
                                let eval = R1CSEval::<F>::from_cycle_inputs(&row_inputs);

                                if !selector {
                                    eval.fmadd_first_group_at_r(
                                        w_r,
                                        &mut acc_az[x_val],
                                        &mut acc_bz_first[x_val],
                                    );
                                } else {
                                    eval.fmadd_second_group_at_r(
                                        w_r,
                                        &mut acc_az[x_val],
                                        &mut acc_bz_second[x_val],
                                    );
                                }
                            }
                        }

                        for x_val in 0..grid_size {
                            az_grid[x_val] = acc_az[x_val].barrett_reduce();
                            bz_grid[x_val] = acc_bz_first[x_val].barrett_reduce()
                                + acc_bz_second[x_val].barrett_reduce();
                        }

                        let buffer_offset = grid_size * (pair_idx - start_pair);
                        let end = buffer_offset + grid_size;
                        az_chunk[buffer_offset..end].copy_from_slice(&az_grid[..grid_size]);
                        bz_chunk[buffer_offset..end].copy_from_slice(&bz_grid[..grid_size]);

                        MultiquadraticPolynomial::<F>::expand_linear_grid_to_multiquadratic(
                            &az_grid,
                            &mut buff_a,
                            &mut tmp,
                            window_size,
                        );
                        MultiquadraticPolynomial::<F>::expand_linear_grid_to_multiquadratic(
                            &bz_grid,
                            &mut buff_b,
                            &mut tmp,
                            window_size,
                        );

                        let e_in = E_in[x_in_val];

                        if window_size == 1 {
                            let prod0 = buff_a[0] * buff_b[0];
                            let prod2 = buff_a[2] * buff_b[2];
                            inner_sum[0] += prod0.mul_unreduced::<9>(e_in);
                            inner_sum[2] += prod2.mul_unreduced::<9>(e_in);
                        } else {
                            for idx in 0..three_pow_dim {
                                let prod = buff_a[idx] * buff_b[idx];
                                inner_sum[idx] += prod.mul_unreduced::<9>(e_in);
                            }
                        }
                    }

                    let e_out = E_out[current_x_out];
                    for idx in 0..three_pow_dim {
                        local_ans[idx] += F::from_montgomery_reduce::<9>(inner_sum[idx]) * e_out;
                    }

                    local_ans
                },
            )
            .reduce(
                || vec![F::zero(); three_pow_dim],
                |mut acc, local_ans| {
                    for idx in 0..three_pow_dim {
                        acc[idx] += local_ans[idx];
                    }
                    acc
                },
            );

        shared.t_prime_poly = Some(MultiquadraticPolynomial::new(window_size, ans));
        (
            DensePolynomial::new(az_bound),
            DensePolynomial::new(bz_bound),
        )
    }

    #[tracing::instrument(
        skip_all,
        name = "OuterLinearStage::fused_materialise_polynomials_round_zero"
    )]
    fn fused_materialise_polynomials_round_zero(
        shared: &mut OuterSharedState<F>,
        num_vars: usize,
    ) -> (DensePolynomial<F>, DensePolynomial<F>) {
        let eq_poly = &shared.split_eq_poly;

        let three_pow_dim = 3_usize.pow(num_vars as u32);
        let grid_size = 1 << num_vars;
        let (E_out, E_in) = eq_poly.E_out_in_for_window(num_vars);

        let num_evals_az = E_out.len() * E_in.len() * grid_size;
        let mut az: Vec<F> = unsafe_allocate_zero_vec(num_evals_az);
        let mut bz: Vec<F> = unsafe_allocate_zero_vec(num_evals_az);

        let ans: Vec<F> = if E_in.len() == 1 {
            az.par_chunks_exact_mut(grid_size)
                .zip(bz.par_chunks_exact_mut(grid_size))
                .enumerate()
                .map(|(i, (az_chunk, bz_chunk))| {
                    let mut local_ans = vec![F::zero(); three_pow_dim];
                    let mut az_grid = vec![F::zero(); grid_size];
                    let mut bz_grid = vec![F::zero(); grid_size];
                    let mut buff_a = vec![F::zero(); three_pow_dim];
                    let mut buff_b = vec![F::zero(); three_pow_dim];
                    let mut tmp = vec![F::zero(); three_pow_dim];

                    if grid_size >= 2 {
                        let mut j = 0;
                        while j < grid_size {
                            let full_idx = grid_size * i + j;
                            let time_step_idx = full_idx >> 1;

                            let row_inputs = R1CSCycleInputs::from_trace::<F>(
                                &shared.bytecode_preprocessing,
                                &shared.trace,
                                time_step_idx,
                            );
                            let eval = R1CSEval::<F>::from_cycle_inputs(&row_inputs);

                            let az0 = eval.az_at_r_first_group(&shared.lagrange_evals_r0);
                            let bz0 = eval.bz_at_r_first_group(&shared.lagrange_evals_r0);

                            let az1 = eval.az_at_r_second_group(&shared.lagrange_evals_r0);
                            let bz1 = eval.bz_at_r_second_group(&shared.lagrange_evals_r0);

                            az_chunk[j] = az0;
                            bz_chunk[j] = bz0;
                            az_grid[j] = az0;
                            bz_grid[j] = bz0;

                            az_chunk[j + 1] = az1;
                            bz_chunk[j + 1] = bz1;
                            az_grid[j + 1] = az1;
                            bz_grid[j + 1] = bz1;

                            j += 2;
                        }
                    } else {
                        for j in 0..grid_size {
                            let full_idx = grid_size * i + j;
                            let time_step_idx = full_idx >> 1;
                            let selector = (full_idx & 1) == 1;

                            let row_inputs = R1CSCycleInputs::from_trace::<F>(
                                &shared.bytecode_preprocessing,
                                &shared.trace,
                                time_step_idx,
                            );
                            let eval = R1CSEval::<F>::from_cycle_inputs(&row_inputs);

                            let (az_at_full_idx, bz_at_full_idx) = if !selector {
                                (
                                    eval.az_at_r_first_group(&shared.lagrange_evals_r0),
                                    eval.bz_at_r_first_group(&shared.lagrange_evals_r0),
                                )
                            } else {
                                (
                                    eval.az_at_r_second_group(&shared.lagrange_evals_r0),
                                    eval.bz_at_r_second_group(&shared.lagrange_evals_r0),
                                )
                            };

                            az_chunk[j] = az_at_full_idx;
                            bz_chunk[j] = bz_at_full_idx;
                            az_grid[j] = az_at_full_idx;
                            bz_grid[j] = bz_at_full_idx;
                        }
                    }

                    MultiquadraticPolynomial::<F>::expand_linear_grid_to_multiquadratic(
                        &az_grid,
                        &mut buff_a,
                        &mut tmp,
                        num_vars,
                    );
                    MultiquadraticPolynomial::<F>::expand_linear_grid_to_multiquadratic(
                        &bz_grid,
                        &mut buff_b,
                        &mut tmp,
                        num_vars,
                    );

                    if num_vars == 1 {
                        local_ans[0] = buff_a[0] * buff_b[0] * E_out[i];
                        local_ans[2] = buff_a[2] * buff_b[2] * E_out[i];
                    } else {
                        for idx in 0..three_pow_dim {
                            local_ans[idx] = buff_a[idx] * buff_b[idx] * E_out[i];
                        }
                    }

                    local_ans
                })
                .reduce(
                    || vec![F::zero(); three_pow_dim],
                    |mut acc, local_ans| {
                        for idx in 0..three_pow_dim {
                            acc[idx] += local_ans[idx];
                        }
                        acc
                    },
                )
        } else {
            let num_xin_bits = E_in.len().log_2();
            az.par_chunks_exact_mut(grid_size * E_in.len())
                .zip(bz.par_chunks_exact_mut(grid_size * E_in.len()))
                .enumerate()
                .map(|(x_out, (az_outer_chunk, bz_outer_chunk))| {
                    let mut local_ans = vec![F::zero(); three_pow_dim];
                    let mut az_grid = vec![F::zero(); grid_size];
                    let mut bz_grid = vec![F::zero(); grid_size];
                    let mut buff_a = vec![F::zero(); three_pow_dim];
                    let mut buff_b = vec![F::zero(); three_pow_dim];
                    let mut tmp = vec![F::zero(); three_pow_dim];

                    for x_in in 0..E_in.len() {
                        let i = (x_out << num_xin_bits) | x_in;

                        if grid_size >= 2 {
                            let mut j = 0;
                            while j < grid_size {
                                let full_idx = grid_size * i + j;
                                let time_step_idx = full_idx >> 1;

                                let row_inputs = R1CSCycleInputs::from_trace::<F>(
                                    &shared.bytecode_preprocessing,
                                    &shared.trace,
                                    time_step_idx,
                                );
                                let eval = R1CSEval::<F>::from_cycle_inputs(&row_inputs);

                                let az0 = eval.az_at_r_first_group(&shared.lagrange_evals_r0);
                                let bz0 = eval.bz_at_r_first_group(&shared.lagrange_evals_r0);

                                let az1 = eval.az_at_r_second_group(&shared.lagrange_evals_r0);
                                let bz1 = eval.bz_at_r_second_group(&shared.lagrange_evals_r0);

                                let offset_in_chunk = x_in * grid_size + j;
                                az_outer_chunk[offset_in_chunk] = az0;
                                bz_outer_chunk[offset_in_chunk] = bz0;
                                az_grid[j] = az0;
                                bz_grid[j] = bz0;

                                az_outer_chunk[offset_in_chunk + 1] = az1;
                                bz_outer_chunk[offset_in_chunk + 1] = bz1;
                                az_grid[j + 1] = az1;
                                bz_grid[j + 1] = bz1;

                                j += 2;
                            }
                        } else {
                            for j in 0..grid_size {
                                let full_idx = grid_size * i + j;
                                let time_step_idx = full_idx >> 1;
                                let selector = (full_idx & 1) == 1;

                                let row_inputs = R1CSCycleInputs::from_trace::<F>(
                                    &shared.bytecode_preprocessing,
                                    &shared.trace,
                                    time_step_idx,
                                );
                                let eval = R1CSEval::<F>::from_cycle_inputs(&row_inputs);

                                let (az_at_full_idx, bz_at_full_idx) = if !selector {
                                    (
                                        eval.az_at_r_first_group(&shared.lagrange_evals_r0),
                                        eval.bz_at_r_first_group(&shared.lagrange_evals_r0),
                                    )
                                } else {
                                    (
                                        eval.az_at_r_second_group(&shared.lagrange_evals_r0),
                                        eval.bz_at_r_second_group(&shared.lagrange_evals_r0),
                                    )
                                };

                                let offset_in_chunk = x_in * grid_size + j;
                                az_outer_chunk[offset_in_chunk] = az_at_full_idx;
                                bz_outer_chunk[offset_in_chunk] = bz_at_full_idx;
                                az_grid[j] = az_at_full_idx;
                                bz_grid[j] = bz_at_full_idx;
                            }
                        }

                        MultiquadraticPolynomial::<F>::expand_linear_grid_to_multiquadratic(
                            &az_grid,
                            &mut buff_a,
                            &mut tmp,
                            num_vars,
                        );
                        MultiquadraticPolynomial::<F>::expand_linear_grid_to_multiquadratic(
                            &bz_grid,
                            &mut buff_b,
                            &mut tmp,
                            num_vars,
                        );

                        let e_product = E_out[x_out] * E_in[x_in];
                        for idx in 0..three_pow_dim {
                            local_ans[idx] += buff_a[idx] * buff_b[idx] * e_product;
                        }
                    }

                    local_ans
                })
                .reduce(
                    || vec![F::zero(); three_pow_dim],
                    |mut acc, local_ans| {
                        for idx in 0..three_pow_dim {
                            acc[idx] += local_ans[idx];
                        }
                        acc
                    },
                )
        };
        shared.t_prime_poly = Some(MultiquadraticPolynomial::new(num_vars, ans));
        (DensePolynomial::new(az), DensePolynomial::new(bz))
    }

    #[tracing::instrument(
        skip_all,
        name = "OuterLinearStage::compute_evaluation_grid_from_polynomials_parallel"
    )]
    fn compute_evaluation_grid_from_polynomials_parallel(
        &self,
        shared: &mut OuterSharedState<F>,
        num_vars: usize,
    ) {
        let eq_poly = &shared.split_eq_poly;

        let n = self.az.len();
        let az = &self.az;
        let bz = &self.bz;
        debug_assert_eq!(n, bz.len());

        let three_pow_dim = 3_usize.pow(num_vars as u32);
        let grid_size = 1 << num_vars;
        let (E_out, E_in) = eq_poly.E_out_in_for_window(num_vars);

        let ans: Vec<F> = if E_in.len() == 1 {
            (0..E_out.len())
                .into_par_iter()
                .map(|i| {
                    let mut local_ans = vec![F::zero(); three_pow_dim];
                    let mut az_grid = vec![F::zero(); grid_size];
                    let mut bz_grid = vec![F::zero(); grid_size];
                    let mut buff_a = vec![F::zero(); three_pow_dim];
                    let mut buff_b = vec![F::zero(); three_pow_dim];
                    let mut tmp = vec![F::zero(); three_pow_dim];

                    for j in 0..grid_size {
                        let index = grid_size * i + j;
                        az_grid[j] = az[index];
                        bz_grid[j] = bz[index];
                    }

                    MultiquadraticPolynomial::<F>::expand_linear_grid_to_multiquadratic(
                        &az_grid,
                        &mut buff_a,
                        &mut tmp,
                        num_vars,
                    );
                    MultiquadraticPolynomial::<F>::expand_linear_grid_to_multiquadratic(
                        &bz_grid,
                        &mut buff_b,
                        &mut tmp,
                        num_vars,
                    );

                    for idx in 0..three_pow_dim {
                        local_ans[idx] = buff_a[idx] * buff_b[idx] * E_out[i];
                    }

                    local_ans
                })
                .reduce(
                    || vec![F::zero(); three_pow_dim],
                    |mut acc, local_ans| {
                        for idx in 0..three_pow_dim {
                            acc[idx] += local_ans[idx];
                        }
                        acc
                    },
                )
        } else {
            let num_xin_bits = E_in.len().log_2();
            (0..E_out.len())
                .into_par_iter()
                .map(|x_out| {
                    let mut local_ans = vec![F::zero(); three_pow_dim];
                    let mut az_grid = vec![F::zero(); grid_size];
                    let mut bz_grid = vec![F::zero(); grid_size];
                    let mut buff_a = vec![F::zero(); three_pow_dim];
                    let mut buff_b = vec![F::zero(); three_pow_dim];
                    let mut tmp = vec![F::zero(); three_pow_dim];

                    for x_in in 0..E_in.len() {
                        let i = (x_out << num_xin_bits) | x_in;

                        for j in 0..grid_size {
                            az_grid[j] = az[grid_size * i + j];
                            bz_grid[j] = bz[grid_size * i + j];
                        }

                        MultiquadraticPolynomial::<F>::expand_linear_grid_to_multiquadratic(
                            &az_grid,
                            &mut buff_a,
                            &mut tmp,
                            num_vars,
                        );
                        MultiquadraticPolynomial::<F>::expand_linear_grid_to_multiquadratic(
                            &bz_grid,
                            &mut buff_b,
                            &mut tmp,
                            num_vars,
                        );

                        for idx in 0..three_pow_dim {
                            local_ans[idx] += buff_a[idx] * buff_b[idx] * E_in[x_in];
                        }
                    }
                    for idx in 0..three_pow_dim {
                        local_ans[idx] *= E_out[x_out];
                    }

                    local_ans
                })
                .reduce(
                    || vec![F::zero(); three_pow_dim],
                    |mut acc, local_ans| {
                        for idx in 0..three_pow_dim {
                            acc[idx] += local_ans[idx];
                        }
                        acc
                    },
                )
        };
        shared.t_prime_poly = Some(MultiquadraticPolynomial::new(num_vars, ans));
    }
}

impl<F: JoltField> LinearSumcheckStage<F> for OuterLinearStage<F> {
    type Shared = OuterSharedState<F>;
    type Streaming = OuterStreamingWindow<F>;

    #[tracing::instrument(skip_all, name = "OuterLinearStage::initialize")]
    fn initialize(
        _streaming: Option<Self::Streaming>,
        shared: &mut Self::Shared,
        window_size: usize,
    ) -> Self {
        let is_not_first_round_of_sumcheck = shared.split_eq_poly.num_challenges() > 0;
        let (az, bz) = if is_not_first_round_of_sumcheck {
            Self::fused_materialise_polynomials_general_with_multiquadratic(shared, window_size)
        } else {
            Self::fused_materialise_polynomials_round_zero(shared, window_size)
        };

        Self { az, bz }
    }

    #[tracing::instrument(skip_all, name = "OuterLinearStage::next_window")]
    fn next_window(&mut self, shared: &mut Self::Shared, window_size: usize) {
        self.compute_evaluation_grid_from_polynomials_parallel(shared, window_size);
    }

    #[tracing::instrument(skip_all, name = "OuterLinearStage::compute_message")]
    fn compute_message(
        &self,
        shared: &Self::Shared,
        window_size: usize,
        previous_claim: F,
    ) -> UniPoly<F> {
        let (t_prime_0, t_prime_inf) = shared.compute_t_evals(window_size);
        shared
            .split_eq_poly
            .gruen_poly_deg_3(t_prime_0, t_prime_inf, previous_claim)
    }

    #[tracing::instrument(skip_all, name = "OuterLinearStage::ingest_challenge")]
    fn ingest_challenge(&mut self, shared: &mut Self::Shared, r_j: F::Challenge, _round: usize) {
        shared.split_eq_poly.bind(r_j);

        if let Some(t_prime_poly) = shared.t_prime_poly.as_mut() {
            t_prime_poly.bind(r_j, BindingOrder::LowToHigh);
        }

        rayon::join(
            || self.az.bind_parallel(r_j, BindingOrder::LowToHigh),
            || self.bz.bind_parallel(r_j, BindingOrder::LowToHigh),
        );
    }

    #[tracing::instrument(skip_all, name = "OuterLinearStage::cache_openings")]
    fn cache_openings<T: Transcript>(
        &self,
        shared: &Self::Shared,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let r_cycle = OuterStreamingProverParams::get_inputs_opening_point(sumcheck_challenges);

        let claimed_witness_evals = R1CSEval::compute_claimed_inputs(
            &shared.bytecode_preprocessing,
            &shared.trace,
            &r_cycle,
        );

        for (i, input) in ALL_R1CS_INPUTS.iter().enumerate() {
            accumulator.append_virtual(
                transcript,
                VirtualPolynomial::from(input),
                SumcheckId::SpartanOuter,
                r_cycle.clone(),
                claimed_witness_evals[i],
            );
        }
    }
}
