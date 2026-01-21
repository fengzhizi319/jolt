use std::iter::zip;
use std::sync::Arc;

use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use ark_std::Zero;

use crate::field::{FMAdd, JoltField, MontgomeryReduce};
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::lagrange_poly::LagrangePolynomial;
use crate::poly::multilinear_polynomial::BindingOrder;
use crate::poly::opening_proof::{
    OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
    VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
};
use crate::poly::split_eq_poly::GruenSplitEqPolynomial;
use crate::poly::unipoly::UniPoly;
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use crate::subprotocols::sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier};
use crate::subprotocols::univariate_skip::build_uniskip_first_round_poly;
use crate::transcripts::Transcript;
use crate::utils::accumulation::Acc8S;
use crate::utils::math::Math;
#[cfg(feature = "allocative")]
use crate::utils::profiling::print_data_structure_heap_usage;
use crate::utils::thread::unsafe_allocate_zero_vec;
use crate::zkvm::instruction::{CircuitFlags, InstructionFlags};
use crate::zkvm::r1cs::constraints::{
    NUM_PRODUCT_VIRTUAL, PRODUCT_CONSTRAINTS, PRODUCT_VIRTUAL_FIRST_ROUND_POLY_DEGREE_BOUND,
    PRODUCT_VIRTUAL_FIRST_ROUND_POLY_NUM_COEFFS, PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE,
    PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE,
    PRODUCT_VIRTUAL_UNIVARIATE_SKIP_EXTENDED_DOMAIN_SIZE,
};
use crate::zkvm::r1cs::evaluation::ProductVirtualEval;
use crate::zkvm::r1cs::inputs::{ProductCycleInputs, PRODUCT_UNIQUE_FACTOR_VIRTUALS};
use crate::zkvm::witness::VirtualPolynomial;
use rayon::prelude::*;
use tracer::instruction::Cycle;

// Product virtualization with univariate skip
//
// We define a "combined" left and right polynomial
// Left(x, y) = \sum_i L(y, i) * Left_i(x),
// Right(x, y) = \sum_i R(y, i) * Right_i(x),
// where Left_i(x) = one of the five left polynomials, Right_i(x) = one of the five right polynomials
// Indexing is over i \in {-2, -1, 0, 1, 2}, though this gets mapped to the 0th, 1st, ..., 4th polynomial
//
// We also need to define the combined claim:
// claim(y) = \sum_i L(y, i) * claim_i,
// where claim_i is the claim of the i-th product virtualization sumcheck
//
// The product virtualization sumcheck is then:
// \sum_y L(tau_high, y) * \sum_x eq(tau_low, x) * Left(x, y) * Right(x, y)
//   = claim(tau_high)
//
// Final claim is:
// L(tau_high, r0) * Eq(tau_low, r_tail^rev) * Left(r_tail, r0) * Right(r_tail, r0)
//
// After this, we also need to check the consistency of the Left and Right evaluations with the
// claimed evaluations of the factor polynomials. This is done in the ProductVirtualInner sumcheck.
//
// TODO (Quang): this is essentially Spartan with non-zero claims. We should unify this with Spartan outer/inner.
// Only complication is to generalize the splitting strategy
// (i.e. Spartan outer currently does uni skip for half of the constraints,
// whereas here we do it for all of them)

/// Degree of the sumcheck round polynomials for [`ProductVirtualRemainderVerifier`].
const PRODUCT_VIRTUAL_REMAINDER_DEGREE: usize = 3;


#[derive(Allocative, Clone)]
/// 用于 Spartan 协议中 "乘积参数检查 (Product Argument)" 的辅助结构体。
///
/// 这个结构体封装了执行 "Univariate Skip" (单变量跳跃) 优化所需的参数。
/// 这里的 "Virtual Polynomial"（虚拟多项式）并不是一个实体化的多项式，
/// 而是通过这种优化技术，将 5 个分散的逻辑检查项（分支、跳转、写入等）
/// 视为一个定义在大小为 5 的小域上的单变量多项式的点值。
///
/// 这种技术允许我们在几乎不增加验证成本的情况下，批量处理指令执行逻辑中的复杂状态转换约束。
pub struct ProductVirtualUniSkipParams<F: JoltField> {
    /// 评估点向量 τ (Tau)，由两部分拼接而成：[τ_low || τ_high]。
    ///
    /// 这个向量代表了多线性多项式的评估坐标：
    /// - **τ_low**: 来自 Spartan 外层 Sumcheck 的随机挑战点 (即 `r_cycle`)。
    ///   它的长度等于 `num_cycle_vars`，用于锁定执行轨迹中的特定时间步（Cycle）。
    /// - **τ_high**: 为 Univariate Skip 优化引入的额外随机挑战点（绑定位）。
    ///   它的长度为 1。这个变量用于在这个微小的虚拟域（大小为5）上进行随机线性组合或评估。
    ///
    /// 注意：变量排序通常遵循 MSB -> LSB，其中 τ_high 位于最后（即作为最低维度的变量扩展）。
    pub tau: Vec<F::Challenge>,

    /// 在基础域上这 5 个乘积项的基础评估值 (Base Evaluations/Claims)。
    ///
    /// 这里的每一个值对应于在 `τ_low` 点处评估特定逻辑多项式得到的结果。
    /// Sumcheck 的任务是证明这些分散值的组合满足约束。
    ///
    /// 数组中的顺序必须严格对应 `NUM_PRODUCT_VIRTUAL` (通常为 5) 定义的组件顺序：
    /// Order: [Product, WriteLookupOutputToRD, WritePCtoRD, ShouldBranch, ShouldJump]
    /// 1. **Product**: 累积乘积项（通常用于 Grand Product Argument 的中间状态或累加器）。
    /// 2. **WriteLookupOutputToRD**: 指示该指令是否将查找表(Lookup)的结果写入目标寄存器 (RD)。
    /// 3. **WritePCtoRD**: 指示该指令是否将 PC+4 写入目标寄存器 (用于 JAL/JALR 指令)。
    /// 4. **ShouldBranch**: 指示该指令逻辑上是否应该触发条件分支 (Branch taken)。
    /// 5. **ShouldJump**: 指示该指令逻辑上是否应该触发无条件跳转 (Jump)。
    pub base_evals: [F; NUM_PRODUCT_VIRTUAL],
}

impl<F: JoltField> ProductVirtualUniSkipParams<F> {
    /// 初始化 `ProductVirtualUniSkipParams` 结构体。
    ///
    /// # 作用
    /// 准备乘法子协议（Product Subprotocol）所需的挑战点（Challenge Points）和基准评估值（Base Evaluations）。
    ///
    /// # 流程解析
    /// 1. **继承挑战点**: 从上一阶段（Spartan Outer Sumcheck）获取评估点 `r_cycle`，作为本阶段的低位挑战点 `τ_low`。
    /// 2. **生成新挑战**: 从 Transcript 中采样一个新的挑战点 `τ_high`。这用于通过随机线性组合将多个不同的乘法约束（如指令编码、跳转逻辑等）合并为一个多项式。
    /// 3. **收集目标值**: 从 Opening Accumulator 中提取上一阶段声称的、各个乘法项的评估值（Claims），这些值将作为本阶段 Sumcheck 的验证目标。
    pub fn new<T: Transcript>(
        opening_accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut T,
    ) -> Self {
        // 1. 复用第一阶段 (Spartan Outer) 的随机点 `r_cycle`。
        // 在乘法虚拟化中，这个向量变成了 `τ_low` (tau_low)。
        // 这样做是为了确保本阶段证明的多项式与上一阶段的主约束系统是在同一点上进行评估的，保证一致性。
        let r_cycle = opening_accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::Product, SumcheckId::SpartanOuter)
            .0
            .r;
        println!("stage2 new : r_cycle: {:?}", r_cycle);

        // 2. 从 Fiat-Shamir Transcript 中采样一个新的标量挑战 `τ_high` (tau_high)。
        // 这是一个 "Univariate Skip" 维度的挑战点。
        // Jolt 有 5 类乘法约束（Instruction, WriteRD, WritePC, Branch, Jump），
        // 我们在大小为 5 的域上使用这个随机点将它们“混合”在一起。
        let tau_high = transcript.challenge_scalar_optimized::<F>();

        // 3. 构建完整的挑战向量 τ = [τ_low || τ_high]。
        // 注意顺序：先是来自 Outer 的周期变量挑战点，最后追加新的高位挑战点。
        let mut tau = r_cycle;
        tau.push(tau_high);

        // 4. 初始化基准评估值数组。
        // 这里存储的是上一阶段声称的、各个具体乘法项的值。
        let mut base_evals: [F; NUM_PRODUCT_VIRTUAL] = [F::zero(); NUM_PRODUCT_VIRTUAL];

        // 遍历所有定义的乘法约束 (PRODUCT_CONSTRAINTS)
        for (i, cons) in PRODUCT_CONSTRAINTS.iter().enumerate() {
            // 从累加器中获取该约束对应的虚拟多项式在 `SpartanOuter` 阶段的评估值。
            // 例如：如果是 `ShouldJump` 约束，这里获取的就是该约束在 `r_cycle` 点的值。
            // 这些值构成了本次 Sumcheck 协议要证明的等式右边（Right-Hand Side）。
            let (_, eval) = opening_accumulator
                .get_virtual_polynomial_opening(cons.output, SumcheckId::SpartanOuter);
            base_evals[i] = eval;
        }

        Self { tau, base_evals }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for ProductVirtualUniSkipParams<F> {
    fn input_claim(&self, _: &dyn OpeningAccumulator<F>) -> F {
        // claim = \sum_i L_i(tau_high) * base_evals[i]
        let tau_high = self.tau[self.tau.len() - 1];
        let w = LagrangePolynomial::<F>::evals::<
            F::Challenge,
            PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE,
        >(&tau_high);
        let mut acc = F::zero();
        for i in 0..NUM_PRODUCT_VIRTUAL {
            acc += w[i] * self.base_evals[i];
        }
        acc
    }

    fn degree(&self) -> usize {
        PRODUCT_VIRTUAL_FIRST_ROUND_POLY_DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        1
    }

    fn normalize_opening_point(
        &self,
        challenges: &[<F as JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        challenges.to_vec().into()
    }
}

/// Uni-skip instance for product virtualization, computing the first-round polynomial only.
#[derive(Allocative)]
pub struct ProductVirtualUniSkipProver<F: JoltField> {
    /// Evaluations of t1(Z) at the extended univariate-skip targets (outside base window)
    extended_evals: [F; PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE],
    /// Verifier challenge for this univariate skip round
    r0: Option<F::Challenge>,
    /// Prover message for this univariate skip round
    uni_poly: Option<UniPoly<F>>,
    pub params: ProductVirtualUniSkipParams<F>,
}

impl<F: JoltField> ProductVirtualUniSkipProver<F> {
    /// Initialize a new prover for the univariate skip round
    /// The 5 base evaluations are the claimed evaluations of the 5 product terms from Spartan outer
    #[tracing::instrument(skip_all, name = "ProductVirtualUniSkipInstanceProver::initialize")]
    pub fn initialize(params: ProductVirtualUniSkipParams<F>, trace: &[Cycle]) -> Self {
        // Compute extended univariate-skip evals using split-eq fold-in-out (includes R^2 scaling)
        //计算5中乘法约束的评估值，并且用eq进行点的加扰，目的是证明CPU视角的trace跟内存、查找表视角的输入输出是一致的。
        let extended_evals = Self::compute_univariate_skip_extended_evals(trace, &params.tau);
        let instance = Self {
            extended_evals,
            params,
            r0: None,
            uni_poly: None,
        };

        #[cfg(feature = "allocative")]
        print_data_structure_heap_usage("ProductVirtualUniSkipInstance", &instance);
        instance
    }


    /// 计算单变量跳过 (Univariate Skip) 阶段第一轮多项式在扩展域上的评估值。
    ///
    /// # 目的
    /// 计算 $t_1(z)$ 在扩展点集上的值 (Outside base window)。
    /// 这里的 $z$ 对应于 `PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE` 定义的几个点。
    ///
    /// # 数学原理
    /// 我们需要计算：
    /// $t_1(z) = \sum_{x} Eq(\tau, x) \cdot P_{fused}(z, x)$
    /// 其中：
    /// * $x = (x_{out} || x_{in})$ 是 Trace 的索引（Cycle Count）。
    /// * $P_{fused}(z, x)$ 是 5 种乘法约束在点 $z$ 处的“融合”多项式值。
    ///   它实际上是 $Left(z, x) \cdot Right(z, x)$。
    ///
    /// # 优化策略 (Split-Eq / Gruen)
    /// 采用了 Gruen 的 Split-Eq 优化算法：
    /// $Eq(\tau, x) = E_{out}(x_{out}) \cdot E_{in}(x_{in})$
    ///
    /// 算法流程：
    /// 1. 并行遍历 $x_{out}$。
    /// 2. 在内层循环遍历 $x_{in}$，累加 $\sum E_{in} \cdot Val$。
    /// 3. 外层循环将内层结果乘以 $E_{out}$。
    /// Compute the extended-domain evaluations t1(z) for univariate-skip (outside base window).
    ///
    /// - For each z target, compute
    ///   t1(z) = Σ_{x_out} E_out[x_out] · Σ_{x_in} E_in[x_in] · left_z(x) · right_z(x),
    ///   where x is the concatenation of (x_out || x_in) in MSB→LSB order.
    ///
    /// Lagrange fusion per target z on (current) extended window {−4,−3,3,4}:
    /// - Compute c[0..4] = LagrangeHelper::shift_coeffs_i32(shift(z)) using the same shifted-kernel
    ///   as outer.rs (indices correspond to the 5 base points).
    /// - Define fused values at this z by linearly combining the 5 product witnesses with c:
    ///   left_z(x)  = Σ_i c[i] · Left_i(x)
    ///   right_z(x) = Σ_i c[i] · Right_i^eff(x)
    ///   with Right_4^eff(x) = 1 − NextIsNoop(x) for the ShouldJump term only.
    ///
    /// Small-value lifting rules for integer accumulation before converting to the field:
    /// - Instruction: LeftInstructionInput is u64 → lift to i128; RightInstructionInput is S64 → i128.
    /// - WriteLookupOutputToRD: IsRdNotZero is bool/u8 → i32; flag is bool/u8 → i32.
    /// - WritePCtoRD: IsRdNotZero is bool/u8 → i32; Jump flag is bool/u8 → i32.
    /// - ShouldBranch: LookupOutput is u64 → i128; Branch flag is bool/u8 → i32.
    /// - ShouldJump: Jump flag (left) is bool/u8 → i32; Right^eff = (1 − NextIsNoop) is bool/u8 → i32.
    fn compute_univariate_skip_extended_evals(
        trace: &[Cycle],
        tau: &[F::Challenge], // 挑战点向量 τ (包含 τ_low 和 τ_high)，用于构建 Eq 多项式
    ) -> [F; PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE] {
        // 初始化 Gruen 分裂 Eq 多项式工具。
        // 这将 tau 分解为两部分，并预计算 E_out (外层/高位部分) 和 E_in (内层/低位部分) 的评估值表。
        // 这种技术允许我们在遍历 Trace 时复用计算结果，将复杂度从 O(N) 乘法降低。
        // `new_with_scaling` 这里传入了 R^2 (Montgomery 常数) 作为缩放因子，用于后续 SIMD/ASM 优化的累加器。
        let split_eq = GruenSplitEqPolynomial::<F>::new_with_scaling(
            tau,
            BindingOrder::LowToHigh,
            Some(F::MONTGOMERY_R_SQUARE),
        );
        let outer_scale = split_eq.get_current_scalar(); // 获取当前的全局缩放因子 (= R^2)

        // 获取预计算的 Eq 表。
        // e_out 对应 Trace 索引的高比特部分，e_in 对应低比特部分。
        let e_out = split_eq.E_out_current();
        let e_in = split_eq.E_in_current();
        let out_len = e_out.len();
        let in_len = e_in.len();

        // 最终的评估结果数组，存储了虚拟多项式在 [0, ..., DEGREE-1] 各个点的累加和。
        let mut final_acc = [F::zero(); PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE];

        // 外层循环 (Serial Fold-Out)：遍历高位索引 x_out
        for x_out in 0..out_len {
            // 初始化内层累加器。
            // Acc8S 是一个优化的累加器结构 (通常用于 SIMD)，可以延迟取模操作以提高性能。
            let mut inner_acc = [Acc8S::<F>::zero(); PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE];

            // 内层循环 (Serial Fold-In)：遍历低位索引 x_in
            for x_in in 0..in_len {
                // 根据 x_out 和 x_in 计算全局 Trace 索引 g。
                // 相当于 g = x_out * in_len + x_in (具体取决于 BindingOrder)。
                let g = split_eq.group_index(x_out, x_in);

                // 获取当前低位对应的 Eq 值 (eq_lo)。
                let e_in_val = e_in[x_in];

                // 从 Trace 中读取第 g 行的数据，构建用于产品检查的输入数据结构。
                let row = ProductCycleInputs::from_trace::<F>(trace, g);

                // 对于 Sumcheck 协议需要构建的单变量多项式，计算其在每个评估点 j (0..degree) 的值:
                for j in 0..PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE {
                    // 计算 Grand Product 电路在该行的特定代数结构值。
                    // "prod_s256" 暗示这里可能涉及 SIMD 或者特定的 256 位操作优化。
                    let prod_s256 = ProductVirtualEval::extended_fused_product_at_j::<F>(&row, j);

                    // 累加：inner_acc[j] += eq_lo * value_at_row
                    // 使用 fmadd (fused multiply-add) 进行高效累加。
                    inner_acc[j].fmadd(&e_in_val, &prod_s256);
                }
            }

            // 获取当前高位对应的 Eq 值 (eq_hi)。
            let e_out_val = e_out[x_out];

            // 将内层累加结果归约并合并到最终结果中。
            for j in 0..PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE {
                // 将优化的累加器转回标准域元素 (Montgomery Reduction)。
                // 此时 inner_val = ∑_{x_in} (eq_lo[x_in] * P(row))
                let inner_val = inner_acc[j].montgomery_reduce();

                // final_acc[j] += eq_hi[x_out] * inner_val
                // 展开后即为：∑_{x_out} eq_hi[x_out] * (∑_{x_in} eq_lo[x_in] * P(...))
                //           = ∑_{g} Eq(g) * P(g)
                final_acc[j] += e_out_val * inner_val;
            }
        }

        // 应用全局缩放因子 (对应最初传入的 R^2)，修正数值并返回。
        final_acc.map(|x| x * outer_scale)
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for ProductVirtualUniSkipProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(
        skip_all,
        name = "ProductVirtualUniSkipInstanceProver::compute_message"
    )]
    fn compute_message(&mut self, _round: usize, _previous_claim: F) -> UniPoly<F> {
        // Load base evals from shared instance and extended from prover state
        let base = self.params.base_evals;
        let tau_high = self.params.tau[self.params.tau.len() - 1];

        // Compute the univariate-skip first round polynomial s1(Y) = L(τ_high, Y) · t1(Y)
        let uni_poly = build_uniskip_first_round_poly::<
            F,
            PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE,
            PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE,
            PRODUCT_VIRTUAL_UNIVARIATE_SKIP_EXTENDED_DOMAIN_SIZE,
            PRODUCT_VIRTUAL_FIRST_ROUND_POLY_NUM_COEFFS,
        >(Some(&base), &self.extended_evals, tau_high);

        self.uni_poly = Some(uni_poly.clone());
        uni_poly
    }

    fn ingest_challenge(&mut self, _: <F as JoltField>::Challenge, _: usize) {
        // Nothing to do
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[<F as JoltField>::Challenge],
    ) {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        debug_assert_eq!(opening_point.len(), 1);
        let claim = self.uni_poly.as_ref().unwrap().evaluate(&opening_point[0]);

        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::UnivariateSkip,
            SumcheckId::SpartanProductVirtualization,
            opening_point,
            claim,
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

pub struct ProductVirtualUniSkipVerifier<F: JoltField> {
    pub params: ProductVirtualUniSkipParams<F>,
}

impl<F: JoltField> ProductVirtualUniSkipVerifier<F> {
    pub fn new<T: Transcript>(
        opening_accumulator: &VerifierOpeningAccumulator<F>,
        transcript: &mut T,
    ) -> Self {
        let params = ProductVirtualUniSkipParams::new(opening_accumulator, transcript);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
for ProductVirtualUniSkipVerifier<F>
{
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
            SumcheckId::SpartanProductVirtualization,
            opening_point,
        );
    }
}

#[derive(Allocative, Clone)]
pub struct ProductVirtualRemainderParams<F: JoltField> {
    /// Number of cycle variables to bind in this remainder (equals log2(T))
    pub n_cycle_vars: usize,
    /// Verifier challenge for univariate skip round
    pub r0: F::Challenge,
    /// The tau vector (length 1 + n_cycle_vars), available to prover and verifier
    pub tau: Vec<F::Challenge>,
}

impl<F: JoltField> ProductVirtualRemainderParams<F> {
    /// 初始化 `ProductVirtualRemainderParams` 结构体。
    ///
    /// # 作用
    /// 准备乘法子协议第二阶段（Remainder / Remaining Rounds）所需的参数。
    ///
    /// # 背景：两阶段 Sumcheck
    /// 乘法虚拟化 Sumcheck 被分为了两个阶段：
    /// 1. **Univariate Skip (第一阶段)**: 在大小为 5 的小域上进行，将 5 类乘法约束（指令、跳转等）压缩为一个。
    ///    这产生了一个挑战点 $r_0$。
    /// 2. **Remainder (本阶段)**: 在布尔超立方体（即 Trace 的 Cycle 维度）上进行的标准 Sumcheck。
    ///
    /// 本函数主要负责从第一阶段的输出中提取 $r_0$，并设置剩余 Sumcheck 的规模（变量数）。
    ///
    /// # 参数
    /// * `trace_len`: 执行轨迹的长度。决定了本阶段 Sumcheck 需要进行的轮数 ($\log_2(\text{len})$)。
    /// * `uni_skip_params`: 第一阶段的参数，包含原始的 $\tau$ 向量。
    /// * `opening_accumulator`: 全局累加器，用于获取第一阶段产生的挑战点 $r_0$。
    pub fn new(
        trace_len: usize,
        uni_skip_params: ProductVirtualUniSkipParams<F>,
        opening_accumulator: &dyn OpeningAccumulator<F>,
    ) -> Self {
        // 1. 获取第一阶段 (Univariate Skip) 的挑战点 r0。
        // 在上一轮 Sumcheck 结束时，Verifier（或 Fiat-Shamir）生成了一个随机点 r_uni_skip。
        // 这个点用于将那一轮的多项式 $t_1(X)$ 归减为一个标量值。
        // 这里的 `r_uni_skip` 实际上是一个长度为 1 的向量，因为 Univariate Skip 仅针对一个变量。
        let (r_uni_skip, _) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::UnivariateSkip,       // 目标：第一阶段产生的虚拟多项式
            SumcheckId::SpartanProductVirtualization, // ID：当前子协议
        );

        // 验证确实只收到了一个挑战点
        debug_assert_eq!(r_uni_skip.len(), 1);
        let r0 = r_uni_skip[0];

        // 2. 构建本阶段参数
        // * n_cycle_vars: 剩余的 Sumcheck 轮数，对应于 Log2(TraceLength)。
        // * tau: 继承自第一阶段的完整挑战向量 [τ_low || τ_high]。
        //   (注：第二阶段主要用到 τ_low 来计算 Eq 多项式，τ_high 和 r0 用于组合系数)。
        // * r0: 第一阶段的挑战点，将在本阶段作为拉格朗日插值的基点。
        Self {
            n_cycle_vars: trace_len.log_2(),
            tau: uni_skip_params.tau,
            r0,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for ProductVirtualRemainderParams<F> {
    fn num_rounds(&self) -> usize {
        self.n_cycle_vars
    }

    fn degree(&self) -> usize {
        PRODUCT_VIRTUAL_REMAINDER_DEGREE
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let (_, uni_skip_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::UnivariateSkip,
            SumcheckId::SpartanProductVirtualization,
        );
        uni_skip_claim
    }

    fn normalize_opening_point(
        &self,
        challenges: &[<F as JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }
}

// TODO: Update docs after merging uni skip round with this sumcheck.
/// Remaining rounds for Product Virtualization after the univariate-skip first round.
/// Mirrors the structure of `OuterRemainingSumcheck` with product-virtualization-specific wiring.
///
/// Final claim (what the prover's last claim must equal, and what the verifier computes):
///
/// Let r₀ be the univariate-skip challenge, and r_tail the remaining cycle-variable challenges
/// bound by this instance (low-to-high from the prover's perspective; the verifier uses the
/// reversed vector `r_tail^rev` when evaluating Eq_τ over τ_low).
///
/// Define Lagrange weights over the size-5 domain at r₀:
///   w_i := L_i(r₀) for i ∈ {0..4} corresponding to
///          [Instruction, WriteLookupOutputToRD, WritePCtoRD, ShouldBranch, ShouldJump].
///
/// Define fused left/right evaluations at the cycle point r_tail:
///   left_eval  := Σ_i w_i · eval(Left_i,  r_tail)
///   right_eval := Σ_i w_i · eval(Right_i, r_tail), except for ShouldJump where
///                 Right_4^eff := 1 − NextIsNoop, i.e., use (1 − eval(NextIsNoop, r_tail)).
///
/// Let
///   E_high := L(τ_high, r₀)  (Lagrange kernel over the size-5 domain)
///   E_low  := Eq_τ_low(τ_low, r_tail^rev)  (multilinear Eq kernel on the cycle variables)
///
/// Then the expected final claim is
///   expected = E_high · E_low · left_eval · right_eval.
///
/// The verifier computes this in `expected_output_claim`. The prover’s final emitted claim
/// after all rounds must match it. Note that `final_sumcheck_evals()` returns the first entries
/// of the fully-bound fused left/right polynomials (used for openings); these are not the final
/// claim themselves but are used to perform the subsequent opening checks.
#[derive(Allocative)]
pub struct ProductVirtualRemainderProver<F: JoltField> {
    #[allocative(skip)]
    trace: Arc<Vec<Cycle>>,
    split_eq_poly: GruenSplitEqPolynomial<F>,
    left: DensePolynomial<F>,
    right: DensePolynomial<F>,
    /// The first round evals (t0, t_inf) computed from a streaming pass over the trace
    first_round_evals: (F, F),
    pub params: ProductVirtualRemainderParams<F>,
}

impl<F: JoltField> ProductVirtualRemainderProver<F> {
    /// 初始化 `ProductVirtualRemainderProver` 实例。
    ///
    /// # 作用
    /// 准备乘法子协议第二阶段（Remainder / Cycle 变量绑定阶段）的 Prover。
    /// 此阶段的任务是对已经通过 Univariate Skip 归约后的“融合”多项式进行关于 Cycle 维度的 Sumcheck。
    ///
    /// # 核心逻辑
    /// 1. **计算融合权重**:
    ///    计算拉格朗日基函数在上一阶段产生的挑战点 $r_0$ 处的评估值 `lagrange_evals_r`。
    ///    这组权重 $w_i = L_i(r_0)$ 将用于把 5 组不同的乘法多项式（指令、跳转等输入输出）线性组合（融合）成两个单一的多项式 $Left$ 和 $Right$。
    ///    即：$Left(x) = \sum w_i \cdot Left_i(x)$。
    ///
    /// 2. **分离挑战向量**:
    ///    将总挑战向量 $\tau$ 分拆为：
    ///    * $\tau_{high}$: 对应于 Univariate Skip 维度的变量。
    ///    * $\tau_{low}$: 对应于 Cycle (执行时间步) 维度的变量。
    ///
    /// 3. **计算全局缩放因子**:
    ///    计算 $L(\tau_{high}, r_0)$。因为完整的验证方程依赖于 $Eq(\tau, r)$，
    ///    即使我们现在只处理 Cycle 维度，高位维度的贡献 $Eq(\tau_{high}, r_0)$ 作为一个常数乘数（Scaling Factor）仍然存在。
    ///
    /// 4. **初始化 Eq 多项式**:
    ///    使用 $\tau_{low}$ 和上述缩放因子初始化 `GruenSplitEqPolynomial`。
    ///    这用于高效计算 $Eq(\tau_{low}, x) \cdot \text{Scaling}$。
    ///
    /// 5. **物化密集多项式与计算首轮状态**:
    ///    调用 `compute_first_quadratic_evals_and_bound_polys`：
    ///    * 将原始 Trace 数据根据权重“物化”（Materialize）为两个密集的向量 `left` 和 `right`。
    ///      这两个向量是后续每一轮 Sumcheck 折叠的基础。
    ///    * 同时计算当前 Sumcheck 首轮多项式的评估值 $t(0)$ 和 $t(inf)$，用于构造 Prover Message。
    #[tracing::instrument(skip_all, name = "ProductVirtualRemainderProver::initialize")]
    pub fn initialize(params: ProductVirtualRemainderParams<F>, trace: Arc<Vec<Cycle>>) -> Self {
        // 1. 计算权重向量 w，其中 w[i] = L_i(r0)。
        // 这里的 r0 是来自 Univariate Skip 阶段的随机点 (challenge)。
        // 这些权重用于将 5 个独立的乘法约束项融合成一组 Left 和 Right 多项式。
        let lagrange_evals_r = LagrangePolynomial::<F>::evals::<
            F::Challenge,
            PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE,
        >(&params.r0);

        // 2. 分解挑战向量 tau。
        // tau 是从 Spartan Outer 传下来的，包含了所有变量的随机挑战。
        // tau_high: Univariate Skip 维度的挑战点 (最后一项)。
        // tau_low: Cycle 维度的挑战点向量 (其余项)。
        let tau_high = params.tau[params.tau.len() - 1];
        let tau_low = &params.tau[..params.tau.len() - 1];

        // 3. 计算高位缩放因子 scalar = L(tau_high, r0)。
        // 完整的 Eq(\tau, x) 可以分解为 Eq(\tau_low, x_cycle) * Eq(\tau_high, r0)。
        // 对于 Univariate Skip 这样的小域，Eq 实际上就是 Lagrange 插值核。
        // 这个因子将在后续计算中一直伴随，确保验证等式两边的一致性。
        let lagrange_tau_r0 = LagrangePolynomial::<F>::lagrange_kernel::<
            F::Challenge,
            PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE,
        >(&tau_high, &params.r0);

        // 4. 初始化 Split-Eq 多项式。
        // 传入 scalar (lagrange_tau_r0) 作为初始缩放因子。
        // 使得该多项式后续吐出的系数实际上是 Eq(tau_low, x) * scalar。
        let split_eq_poly: GruenSplitEqPolynomial<F> =
            GruenSplitEqPolynomial::<F>::new_with_scaling(
                tau_low,
                BindingOrder::LowToHigh,
                Some(lagrange_tau_r0),
            );

        // 5. 核心计算：生成首轮多项式评估点和绑定后的密集多项式。
        // 这一步会扫描 Trace，利用权重 lagrange_evals_r 将多列数据融合，
        // 生成 Left 和 Right 向量，并同时计算出 Sumcheck 第一轮所需的 t0 和 t_inf。
        // 这个函数完成了从“原始执行轨迹”到“Sumcheck 密集多项式”的转换。
        let (t0, t_inf, left_bound, right_bound) =
            Self::compute_first_quadratic_evals_and_bound_polys(
                &trace,
                &lagrange_evals_r,
                &split_eq_poly,
            );

        Self {
            split_eq_poly,
            trace,
            left: left_bound,   // 存储融合后的密集 Left 多项式 (用于后续 rounds 绑定)
            right: right_bound, // 存储融合后的密集 Right 多项式 (用于后续 rounds 绑定)
            first_round_evals: (t0, t_inf), // 第一轮 Sumcheck 消息所需的点值
            params,
        }
    }

    /// Compute the quadratic evaluations for the streaming round (right after univariate skip).
    ///
    /// After binding the univariate-skip variable at r0, we must
    /// compute the cubic round polynomial endpoints over the cycle variables only:
    ///   t(0)  = Σ_{x_out} E_out · Σ_{x_in} E_in · Left_0(x) · Right_0(x)
    ///   t(∞)  = Σ_{x_out} E_out · Σ_{x_in} E_in · (Left_1−Left_0) · (Right_1−Right_0)
    /// We also build per-(x_out,x_in) interleaved coefficients [lo, hi] in order to bind them by r_0
    /// once, after which remaining rounds bind linearly over the cycle variables.
    ///
    /// Product virtualization specifics:
    /// - Left/Right are fused linear combinations of five per-type witnesses with Lagrange
    ///   weights w_i = L_i(r0) over the size-5 domain.
    /// - For ShouldJump, the effective right factor is (1 − NextIsNoop).
    /// - We follow outer's delayed-reduction pattern across x_in to reduce modular reductions.
    #[inline]
    fn compute_first_quadratic_evals_and_bound_polys(
        trace: &[Cycle],
        weights_at_r0: &[F; NUM_PRODUCT_VIRTUAL],
        split_eq_poly: &GruenSplitEqPolynomial<F>,
    ) -> (F, F, DensePolynomial<F>, DensePolynomial<F>) {
        let num_x_out_vals = split_eq_poly.E_out_current_len();
        let num_x_in_vals = split_eq_poly.E_in_current_len();
        let iter_num_x_in_vars = num_x_in_vals.log_2();

        let groups_exact = num_x_out_vals
            .checked_mul(num_x_in_vals)
            .expect("overflow computing groups_exact");

        // Preallocate interleaved buffers once ([lo, hi] per entry)
        let mut left_bound: Vec<F> = unsafe_allocate_zero_vec(2 * groups_exact);
        let mut right_bound: Vec<F> = unsafe_allocate_zero_vec(2 * groups_exact);

        // Parallel over x_out groups using exact-sized mutable chunks, with per-worker fold
        let (t0_acc_unr, t_inf_acc_unr) = left_bound
            .par_chunks_exact_mut(2 * num_x_in_vals)
            .zip(right_bound.par_chunks_exact_mut(2 * num_x_in_vals))
            .enumerate()
            .fold(
                || (F::Unreduced::<9>::zero(), F::Unreduced::<9>::zero()),
                |(mut acc0, mut acci), (x_out_val, (left_chunk, right_chunk))| {
                    let mut inner_sum0 = F::Unreduced::<9>::zero();
                    let mut inner_sum_inf = F::Unreduced::<9>::zero();
                    for x_in_val in 0..num_x_in_vals {
                        let base_idx = (x_out_val << iter_num_x_in_vars) | x_in_val;
                        let idx_lo = base_idx << 1;
                        let idx_hi = idx_lo + 1;

                        // Materialize rows for lo and hi once
                        let row_lo = ProductCycleInputs::from_trace::<F>(trace, idx_lo);
                        let row_hi = ProductCycleInputs::from_trace::<F>(trace, idx_hi);

                        let (left0, right0) = ProductVirtualEval::fused_left_right_at_r::<F>(
                            &row_lo,
                            &weights_at_r0[..],
                        );
                        let (left1, right1) = ProductVirtualEval::fused_left_right_at_r::<F>(
                            &row_hi,
                            &weights_at_r0[..],
                        );

                        let p0 = left0 * right0;
                        let slope = (left1 - left0) * (right1 - right0);
                        let e_in = split_eq_poly.E_in_current()[x_in_val];
                        inner_sum0 += e_in.mul_unreduced::<9>(p0);
                        inner_sum_inf += e_in.mul_unreduced::<9>(slope);
                        let off = 2 * x_in_val;
                        left_chunk[off] = left0;
                        left_chunk[off + 1] = left1;
                        right_chunk[off] = right0;
                        right_chunk[off + 1] = right1;
                    }
                    let e_out = split_eq_poly.E_out_current()[x_out_val];
                    let reduced0 = F::from_montgomery_reduce::<9>(inner_sum0);
                    let reduced_inf = F::from_montgomery_reduce::<9>(inner_sum_inf);
                    acc0 += e_out.mul_unreduced::<9>(reduced0);
                    acci += e_out.mul_unreduced::<9>(reduced_inf);
                    (acc0, acci)
                },
            )
            .reduce(
                || (F::Unreduced::<9>::zero(), F::Unreduced::<9>::zero()),
                |a, b| (a.0 + b.0, a.1 + b.1),
            );

        (
            F::from_montgomery_reduce::<9>(t0_acc_unr),
            F::from_montgomery_reduce::<9>(t_inf_acc_unr),
            DensePolynomial::new(left_bound),
            DensePolynomial::new(right_bound),
        )
    }

    /// Compute the quadratic endpoints for remaining rounds.
    fn remaining_quadratic_evals(&self) -> (F, F) {
        let n = self.left.len();
        debug_assert_eq!(n, self.right.len());
        let [t0, tinf] = self.split_eq_poly.par_fold_out_in_unreduced::<9, 2>(&|g| {
            let l0 = self.left[2 * g];
            let l1 = self.left[2 * g + 1];
            let r0 = self.right[2 * g];
            let r1 = self.right[2 * g + 1];
            let p0 = l0 * r0;
            let slope = (l1 - l0) * (r1 - r0);
            [p0, slope]
        });
        (t0, tinf)
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T>
for ProductVirtualRemainderProver<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(skip_all, name = "ProductVirtualRemainderProver::compute_message")]
    fn compute_message(&mut self, round: usize, previous_claim: F) -> UniPoly<F> {
        let (t0, t_inf) = if round == 0 {
            self.first_round_evals
        } else {
            self.remaining_quadratic_evals()
        };
        self.split_eq_poly
            .gruen_poly_deg_3(t0, t_inf, previous_claim)
    }

    #[tracing::instrument(skip_all, name = "ProductVirtualRemainderProver::ingest_challenge")]
    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        rayon::join(
            || self.left.bind_parallel(r_j, BindingOrder::LowToHigh),
            || self.right.bind_parallel(r_j, BindingOrder::LowToHigh),
        );

        // Bind eq_poly for next round
        self.split_eq_poly.bind(r_j);
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let r_cycle = self.params.normalize_opening_point(sumcheck_challenges);
        let claims = ProductVirtualEval::compute_claimed_factors::<F>(&self.trace, &r_cycle);
        for (poly, claim) in zip(PRODUCT_UNIQUE_FACTOR_VIRTUALS, claims) {
            accumulator.append_virtual(
                transcript,
                poly,
                SumcheckId::SpartanProductVirtualization,
                r_cycle.clone(),
                claim,
            );
        }
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

pub struct ProductVirtualRemainderVerifier<F: JoltField> {
    params: ProductVirtualRemainderParams<F>,
}

impl<F: JoltField> ProductVirtualRemainderVerifier<F> {
    pub fn new(
        trace_len: usize,
        uni_skip_params: ProductVirtualUniSkipParams<F>,
        opening_accumulator: &VerifierOpeningAccumulator<F>,
    ) -> Self {
        let params =
            ProductVirtualRemainderParams::new(trace_len, uni_skip_params, opening_accumulator);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
for ProductVirtualRemainderVerifier<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        // Lagrange weights at r0
        let w = LagrangePolynomial::<F>::evals::<
            F::Challenge,
            PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE,
        >(&self.params.r0);

        // Fetch factor claims
        let l_inst = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::LeftInstructionInput,
                SumcheckId::SpartanProductVirtualization,
            )
            .1;
        let r_inst = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RightInstructionInput,
                SumcheckId::SpartanProductVirtualization,
            )
            .1;
        let is_rd_not_zero = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::InstructionFlags(InstructionFlags::IsRdNotZero),
                SumcheckId::SpartanProductVirtualization,
            )
            .1;
        let wl_flag = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::OpFlags(CircuitFlags::WriteLookupOutputToRD),
                SumcheckId::SpartanProductVirtualization,
            )
            .1;
        let j_flag = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::OpFlags(CircuitFlags::Jump),
                SumcheckId::SpartanProductVirtualization,
            )
            .1;
        let lookup_out = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::LookupOutput,
                SumcheckId::SpartanProductVirtualization,
            )
            .1;
        let branch_flag = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::InstructionFlags(InstructionFlags::Branch),
                SumcheckId::SpartanProductVirtualization,
            )
            .1;
        let next_is_noop = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::NextIsNoop,
                SumcheckId::SpartanProductVirtualization,
            )
            .1;

        let fused_left = w[0] * l_inst
            + w[1] * is_rd_not_zero
            + w[2] * is_rd_not_zero
            + w[3] * lookup_out
            + w[4] * j_flag;
        let fused_right = w[0] * r_inst
            + w[1] * wl_flag
            + w[2] * j_flag
            + w[3] * branch_flag
            + w[4] * (F::one() - next_is_noop);

        // Multiply by L(τ_high, r0) and Eq(τ_low, r_tail^rev)
        let tau_high = &self.params.tau[self.params.tau.len() - 1];
        let tau_high_bound_r0 = LagrangePolynomial::<F>::lagrange_kernel::<
            F::Challenge,
            PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE,
        >(tau_high, &self.params.r0);
        let tau_low = &self.params.tau[..self.params.tau.len() - 1];
        let r_tail_reversed: Vec<F::Challenge> =
            sumcheck_challenges.iter().rev().copied().collect();
        let tau_bound_r_tail_reversed = EqPolynomial::mle(tau_low, &r_tail_reversed);

        tau_high_bound_r0 * tau_bound_r_tail_reversed * fused_left * fused_right
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[<F as JoltField>::Challenge],
    ) {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        for vp in PRODUCT_UNIQUE_FACTOR_VIRTUALS.iter() {
            accumulator.append_virtual(
                transcript,
                *vp,
                SumcheckId::SpartanProductVirtualization,
                opening_point.clone(),
            );
        }
    }
}
