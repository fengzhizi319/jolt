use common::jolt_device::MemoryLayout;
use num_traits::Zero;

use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use rayon::prelude::*;
use tracer::instruction::Cycle;

use crate::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        identity_poly::UnmapRamAddressPolynomial,
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
    utils::{math::Math, thread::unsafe_allocate_zero_vec},
    zkvm::{config::OneHotParams, ram::remap_address, witness::VirtualPolynomial},
};

// RAM RAF evaluation sumcheck
//
// Proves the relation:
//   Σ_{k=0}^{K-1} ra(k) ⋅ unmap(k) = raf_claim,
// where:
// - ra(k) = Σ_j eq(r_cycle, j) ⋅ 1[address(j) = k] aggregates access counts per address k.
// - unmap(k) converts the remapped address k back to its original address.
// - raf_claim is the claimed sum of unmapped addresses over the trace from the Spartan outer sumcheck.

/// Degree bound of the sumcheck round polynomials in [`RafEvaluationSumcheckVerifier`].
const DEGREE_BOUND: usize = 2;

#[derive(Allocative, Clone)]
pub struct RafEvaluationSumcheckParams<F: JoltField> {
    /// log K (number of rounds)
    pub log_K: usize,
    /// Start address for unmap polynomial
    pub start_address: u64,
    pub r_cycle: OpeningPoint<BIG_ENDIAN, F>,
}

impl<F: JoltField> RafEvaluationSumcheckParams<F> {
      /// 初始化 `RafEvaluationSumcheckParams` 结构体。
      ///
      /// # 作用
      /// 准备 RAM 地址频率 (RAF - Read Access Frequency) 评估 Sumcheck 协议所需的参数。
      /// 这个 Sumcheck 旨在证明 RAM 地址的重映射逻辑（Remapping）与原始 trace 中的地址访问记录是一致的。
      ///
      /// # 核心逻辑
      /// 1. 确定地址空间范围（基地址和大小）。
      /// 2. 提取上一阶段（Spartan Outer）产生的随机挑战点 `r_cycle`，确保两个证明阶段针对的是同一个多项式评估点。
      ///
      /// # 参数
      /// * `memory_layout`: 内存布局配置，定义了 RAM 的起始物理地址。
      /// * `one_hot_params`: 包含 RAM 相关的配置，如地址空间大小 $K$。
      /// * `opening_accumulator`: 累加器，用于获取两个阶段之间的关联数据（Opening）。
      pub fn new(
          memory_layout: &MemoryLayout,
          one_hot_params: &OneHotParams,
          opening_accumulator: &dyn OpeningAccumulator<F>,
      ) -> Self {
          // -------------------------------------------------------------------------
          // 1. 获取基准物理地址 (Base Address)
          // -------------------------------------------------------------------------
          // 背景：Jolt 的内存不是从 0 开始连续使用的。
          // 真实的物理内存可能分段：代码段在 0x1000，堆在 0x8000 等。
          // 但为了 ZK 证明的效率，我们在电路内部使用“连续索引” (0, 1, 2...K) 来表示内存操作。
          //
          // 这个 start_address 就是 offset。
          // 关系：Physical_Address = Index + start_address
          let start_address = memory_layout.get_lowest_address();

          // -------------------------------------------------------------------------
          // 2. 确定 Sumcheck 的规模 (Log Size)
          // -------------------------------------------------------------------------
          // RAM 的证明不是针对整个 64位地址空间，而是针对实际使用到的（或 Padding 后的）
          // 内存操作次数 K。
          // log_K 决定了 Sumcheck 多项式有几个变量 (x_0 ... x_{logK-1})。ram_k是使用到的RAM空间大小
          let log_K = one_hot_params.ram_k.log_2();//ram_k=8192,log_K=13

          // -------------------------------------------------------------------------
          // 3. 核心：获取 Stage 1 的“指纹” (The Linkage)
          // -------------------------------------------------------------------------
          // 这一步最关键。
          //
          // opening_accumulator: 这是一个记账本，里面存着 Stage 1 结束时的所有数据。
          // SumcheckId::SpartanOuter: 指的是 Stage 1 (CPU 指令执行阶段)。
          // VirtualPolynomial::RamAddress: 这是 CPU 在 Stage 1 生成的“地址多项式”。
          //
          // get_virtual_polynomial_opening 返回两个值：
          // 1. r_cycle: Stage 1 结束时生成的随机挑战点（代表了所有指令周期的随机线性组合）。
          // 2. claim (被忽略的 _): CPU 声称在 r_cycle 点计算出的“加权地址和”。
          //
          // 我们这里获取 r_cycle 是为了确保 Stage 2 的 Sumcheck 是在同一个随机点上进行的。
          // 只有在同一个 r 上，Stage 1 的 P(r) 和 Stage 2 的 Q(r) 才能比较。
          let (r_cycle, _) = opening_accumulator.get_virtual_polynomial_opening(
              VirtualPolynomial::RamAddress, // 这里的含义是：我要验证的目标是“地址”
              SumcheckId::SpartanOuter,      // 来源是：CPU 执行阶段
          );

          Self {
              log_K,
              start_address,
              r_cycle, // 拿着这个钥匙，去开启 RAM 的 Sumcheck
          }
      }
}

impl<F: JoltField> SumcheckInstanceParams<F> for RafEvaluationSumcheckParams<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.log_K
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let (_, raf_input_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamAddress,
            SumcheckId::SpartanOuter,
        );
        raf_input_claim
    }

    fn normalize_opening_point(
        &self,
        challenges: &[<F as JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }
}

/// Sumcheck prover for [`RafEvaluationSumcheckVerifier`].
#[derive(Allocative)]
pub struct RafEvaluationSumcheckProver<F: JoltField> {
    /// The ra polynomial
    ra: MultilinearPolynomial<F>,
    /// The unmap polynomial
    unmap: UnmapRamAddressPolynomial<F>,
    pub params: RafEvaluationSumcheckParams<F>,
}

impl<F: JoltField> RafEvaluationSumcheckProver<F> {
    /// 初始化 `RafEvaluationSumcheckProver` 实例。
    ///
    /// # 作用
    /// 计算并构建 `ra` (Random Access) 多项式和 `unmap` 多项式。
    /// `ra` 多项式是一个向量，长度为 $K$ (RAM 大小)，其中每个位置 $k$ 存储了该地址所有访问的加权和。
    /// 权重由之前的 Sumcheck 阶段生成的随机挑战点 $r_{cycle}$ 决定。
    ///
    /// # 数学公式
    /// $$ ra(k) = \sum_{j=0}^{T-1} eq(r_{cycle}, j) \cdot \mathbb{1}[\text{address}(j) = k] $$
    ///
    /// # 核心逻辑
    /// 1. **Split-Eq 优化**: 将 $r_{cycle}$ 拆分为高位 ($r_{hi}$) 和低位 ($r_{lo}$)，分别预计算 Eq 表。
    ///    这利用了 $eq(r, j) = eq(r_{hi}, j_{hi}) \cdot eq(r_{lo}, j_{lo})$ 的性质，
    ///    将计算结构化为两层循环，便于并行化处理。
    /// 2. **并行累加**: 使用 Rayon 对时间步（Cycle）的高位部分进行并行分块。
    ///    每个线程维护一个大小为 $K$ 的局部 `partial` 向量，用于统计该线程负责的时间片段内的地址访问贡献。
    /// 3. **全局归约**: 将所有线程的 `partial` 向量逐元素相加，得到最终的 `ra` 向量。
    /// 4. **构建多项式**: 将结果封装为 `MultilinearPolynomial`，并初始化辅助的 `unmap` 多项式。
    #[tracing::instrument(skip_all, name = "RamRafEvaluationSumcheckProver::initialize")]
    pub fn initialize(
        // [参数 1] params: 包含 Verifier 发来的随机挑战点 r_cycle。
        // 数学意义：r_cycle 是用于压缩时间维度的随机向量 $\vec{r}$。
        params: RafEvaluationSumcheckParams<F>,

        // [参数 2] trace: CPU 的完整执行痕迹。
        // 数学意义：时间序列数据，trace[t] 代表时刻 t 的状态。
        trace: &[Cycle],

        // [参数 3] memory_layout: 内存布局描述。
        // 作用：提供物理地址(如 0x8000)到密集索引(如 0, 1, 2)的映射函数 $\pi(addr) \to k$。
        memory_layout: &MemoryLayout,
    ) -> Self {
        let T = trace.len();                    // T: 总时间步数
        let K = 1 << params.log_K;              // K: 内存空间的大小 (2^log_K)

        // =======================================================================
        // 阶段 1: Split-Eq 准备 (利用张量积性质优化权重计算)
        // =======================================================================
        // 假设 r_cycle 的长度为 m (例如 m=9)，则时间空间大小为 2^9 = 512。
        // 我们将其拆分为高位和低位两部分。
        let r_cycle = &params.r_cycle.r;
        let log_T = r_cycle.len();

        // 拆分策略：对半分。例如 9 位拆分为 低位4 + 高位5。
        let lo_bits = log_T / 2;
        let hi_bits = log_T - lo_bits;

        // split_at 将向量切开。
        // r_hi 对应高位比特，控制外层循环。
        // r_lo 对应低位比特，控制内层循环。
        let (r_hi, r_lo) = r_cycle.split_at(hi_bits);

        // [核心优化] 预计算 Eq 表
        // E_hi[i] = \tilde{eq}(r_hi, i)  <-- 这是一个大小为 2^hi_bits 的表
        // E_lo[j] = \tilde{eq}(r_lo, j)  <-- 这是一个大小为 2^lo_bits 的表
        // 数学原理：\tilde{eq}(r, t) = E_hi[t_hi] * E_lo[t_lo]
        let E_hi = EqPolynomial::<F>::evals(r_hi);
        let E_lo = EqPolynomial::<F>::evals(r_lo);

        let in_len = E_lo.len(); // 内层循环长度，等于 2^lo_bits

        // =======================================================================
        // 阶段 2: 核心计算 (单线程累加)
        // =======================================================================
        // 目标：构建 ra_evals 向量
        // 公式：ra_evals[k] = \sum_{t} \mathbb{I}(addr_t == k) * weight_t

        // 分配一个全 0 的向量，大小为 K (覆盖所有可能的内存索引)
        let mut ra_evals: Vec<F> = unsafe_allocate_zero_vec(K);

        // 双层循环遍历时间 t (从 0 到 T-1)
        // t 的二进制分解： t = c_hi * 2^{lo_bits} + c_lo

        // 外层循环：遍历高位部分 (Block)
        for c_hi in 0..E_hi.len() {
            let e_hi = E_hi[c_hi];         // 获取高位权重分量
            let c_hi_base = c_hi * in_len; // 当前 Block 的起始时间索引

            // 内层循环：遍历低位部分 (Offset within Block)
            for c_lo in 0..in_len {
                // 计算绝对时间步 j (即 t)
                let j = c_hi_base + c_lo;

                // [边界检查]
                // 因为 EqPolynomial 总是按 2 的幂次生成，而 Trace 长度 T 可能不是 2 的幂次。
                // 如果 j 超出了实际 Trace 长度，停止计算，相当于把后面填充为 0。
                if j >= T {
                    break;
                }

                // [步骤 A] 获取物理地址并重映射
                // 这里的 trace[j] 是时刻 j 的操作。
                // remap_address 是映射函数 \pi: PhysicalAddr -> Index k
                // 如果地址不在 RAM 范围内 (比如是代码区访问)，返回 None，跳过。
                if let Some(k) =
                    remap_address(trace[j].ram_access().address() as u64, memory_layout)
                {
                    // [步骤 B] 计算总权重
                    // weight = \tilde{eq}(r, j)
                    //        = \tilde{eq}(r_hi, c_hi) * \tilde{eq}(r_lo, c_lo)
                    //        = e_hi * E_lo[c_lo]
                    let weight = e_hi * E_lo[c_lo];

                    // [步骤 C] 累加到桶中
                    // 这行代码实现了公式中的求和符号 \sum
                    ra_evals[k as usize] += weight;
                }
            }
        }

        // =======================================================================
        // 阶段 3: 封装结果
        // =======================================================================
        // 将计算好的密集向量封装为多线性多项式对象
        let ra = MultilinearPolynomial::from(ra_evals);

        // 构造辅助多项式 Unmap
        // 它的作用是定义反向映射： k -> PhysicalAddr
        // 在验证阶段，Verifier 需要知道索引 k 到底代表真实的哪个物理地址。
        let lowest_memory_address = memory_layout.get_lowest_address();
        let unmap = UnmapRamAddressPolynomial::new(K.log_2(), lowest_memory_address);

        Self { ra, unmap, params }
    }

}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for RafEvaluationSumcheckProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(skip_all, name = "RamRafEvaluationSumcheckProver::compute_message")]
    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        // 初始化累加器，用于存储每个度数的评估值总和
        let mut evals = [F::Unreduced::zero(); DEGREE_BOUND];

        // 顺序遍历所有索引，计算每个位置的贡献
        for i in 0..self.ra.len() / 2 {
            // 获取 ra 多项式在第 i 个位置的评估值数组
            let ra_evals = self
                .ra
                .sumcheck_evals_array::<DEGREE_BOUND>(i, BindingOrder::LowToHigh);
            // 获取 unmap 多项式在第 i 个位置的评估值
            let unmap_evals =
                self.unmap
                    .sumcheck_evals(i, DEGREE_BOUND, BindingOrder::LowToHigh);

            // 计算乘积评估值并累加到结果中
            evals[0] += ra_evals[0].mul_unreduced::<9>(unmap_evals[0]);
            evals[1] += ra_evals[1].mul_unreduced::<9>(unmap_evals[1]);
        }

        // 将累加的未约减值转换为约减后的域元素
        let evals = evals.map(F::from_montgomery_reduce);

        UniPoly::from_evals_and_hint(previous_claim, &evals)
    }

    #[tracing::instrument(skip_all, name = "RamRafEvaluationSumcheckProver::ingest_challenge")]
    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        // 单线程顺序绑定挑战值
        self.ra.bind(r_j, BindingOrder::LowToHigh);
        self.unmap.bind(r_j, BindingOrder::LowToHigh);
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let r_address = self.params.normalize_opening_point(sumcheck_challenges);
        let r_cycle = &self.params.r_cycle;
        let ra_opening_point = OpeningPoint::new([&*r_address.r, &*r_cycle.r].concat());
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RamRa,
            SumcheckId::RamRafEvaluation,
            ra_opening_point,
            self.ra.final_sumcheck_claim(),
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

pub struct RafEvaluationSumcheckVerifier<F: JoltField> {
    params: RafEvaluationSumcheckParams<F>,
}

impl<F: JoltField> RafEvaluationSumcheckVerifier<F> {
    pub fn new(
        memory_layout: &MemoryLayout,
        one_hot_params: &OneHotParams,
        opening_accumulator: &VerifierOpeningAccumulator<F>,
    ) -> Self {
        let params =
            RafEvaluationSumcheckParams::new(memory_layout, one_hot_params, opening_accumulator);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
    for RafEvaluationSumcheckVerifier<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let r = self.params.normalize_opening_point(sumcheck_challenges);
        // Compute unmap evaluation at r
        let unmap_eval =
            UnmapRamAddressPolynomial::<F>::new(self.params.log_K, self.params.start_address)
                .evaluate(&r.r);

        let (_, ra_input_claim) = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::RamRa, SumcheckId::RamRafEvaluation);

        // Return unmap(r) * ra(r)
        unmap_eval * ra_input_claim
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let r_address = self.params.normalize_opening_point(sumcheck_challenges);
        let r_cycle = &self.params.r_cycle;
        let ra_opening_point = OpeningPoint::new([&*r_address.r, &*r_cycle.r].concat());
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RamRa,
            SumcheckId::RamRafEvaluation,
            ra_opening_point,
        );
    }
}
