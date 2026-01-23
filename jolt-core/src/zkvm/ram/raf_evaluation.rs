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
          // log_K 决定了 Sumcheck 多项式有几个变量 (x_0 ... x_{logK-1})。
          let log_K = one_hot_params.ram_k.log_2();

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
        params: RafEvaluationSumcheckParams<F>,
        trace: &[Cycle],
        memory_layout: &MemoryLayout,
    ) -> Self {
        let T = trace.len();
        let K = 1 << params.log_K;

        // Split-Eq (分块 Eq) 优化：
        // EqPolynomial::evals 使用大端序：r_cycle[0] 是最高位 (MSB)。
        // 为了在循环中获得连续的 Cycle 索引访问模式，我们将 r_cycle 拆分为高位和低位两部分。
        // 后缀（低位）作为内层循环 E_lo，前缀（高位）作为外层并行循环 E_hi。
        let r_cycle = &params.r_cycle.r;
        let log_T = r_cycle.len();
        let lo_bits = log_T / 2;
        let hi_bits = log_T - lo_bits;
        // split_at: 将 challenge 向量切分为 r_hi 和 r_lo
        let (r_hi, r_lo) = r_cycle.split_at(hi_bits);

        // 并行预计算 Eq 表：
        // E_hi: 大小为 2^hi_bits，对应时间步的高位部分
        // E_lo: 大小为 2^lo_bits，对应时间步的低位部分
        let (E_hi, E_lo) = rayon::join(
            || EqPolynomial::<F>::evals(r_hi),
            || EqPolynomial::<F>::evals(r_lo),
        );

        let in_len = E_lo.len(); // 内层循环长度 = 2^lo_bits

        // 确定并行度：将 E_hi 分块给不同的线程处理
        let num_threads = rayon::current_num_threads();
        let chunk_size = E_hi.len().div_ceil(num_threads);

        // 并行计算 partial ra_evals (局部 ra 统计向量)：
        // 每个线程独立分配一个大小为 K 的零向量，避免锁竞争。
        let ra_evals: Vec<F> = E_hi
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(chunk_idx, chunk)| {
                // 分配本地累加器，大小为 K (地址空间大小)
                let mut partial: Vec<F> = unsafe_allocate_zero_vec(K);

                let chunk_start = chunk_idx * chunk_size;
                // 外层循环：遍历分配给当前线程的 E_hi 块
                for (local_idx, &e_hi) in chunk.iter().enumerate() {
                    let c_hi = chunk_start + local_idx;
                    // c_hi 是高位索引，对应的实际 cycle 索引基数是 c_hi * 2^lo_bits
                    let c_hi_base = c_hi * in_len;

                    // 内层循环：遍历所有低位组合
                    for c_lo in 0..in_len {
                        let j = c_hi_base + c_lo;
                        // 边界检查：防止 j 超出 Trace 实际长度
                        if j >= T {
                            break;
                        }

                        // 获取第 j 个 cycle 访问的物理地址，并映射到密集索引 k
                        if let Some(k) =
                            remap_address(trace[j].ram_access().address() as u64, memory_layout)
                        {
                            // 累加权重：eq(r_cycle, j) = E_hi[c_hi] * E_lo[c_lo]
                            // 将结果加到对应的地址桶 k 中
                            partial[k as usize] += e_hi * E_lo[c_lo];
                        }
                    }
                }

                partial
            })
            // Reduce 阶段：将所有线程产生的 partial 向量对应位置相加
            .reduce(
                || unsafe_allocate_zero_vec(K),
                |mut running, new| {
                    running
                        .par_iter_mut()
                        .zip(new.par_iter())
                        .for_each(|(x, y)| *x += *y);
                    running
                },
            );

        // 构造密集的多线性多项式 ra
        let ra = MultilinearPolynomial::from(ra_evals);
        
        // 构造 Unmap 多项式
        // 这是一个简单的多项式，用于表示从重映射地址 k 恢复到物理地址的映射关系。
        // 它在 Sumcheck 验证中用于确保 proving 的地址是一致的。
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
        let evals = (0..self.ra.len() / 2)
            .into_par_iter()
            .map(|i| {
                let ra_evals = self
                    .ra
                    .sumcheck_evals_array::<DEGREE_BOUND>(i, BindingOrder::LowToHigh);
                let unmap_evals =
                    self.unmap
                        .sumcheck_evals(i, DEGREE_BOUND, BindingOrder::LowToHigh);

                // Compute the product evaluations
                [
                    ra_evals[0].mul_unreduced::<9>(unmap_evals[0]),
                    ra_evals[1].mul_unreduced::<9>(unmap_evals[1]),
                ]
            })
            .reduce(
                || [F::Unreduced::zero(); DEGREE_BOUND],
                |running, new| [running[0] + new[0], running[1] + new[1]],
            )
            .map(F::from_montgomery_reduce);

        UniPoly::from_evals_and_hint(previous_claim, &evals)
    }

    #[tracing::instrument(skip_all, name = "RamRafEvaluationSumcheckProver::ingest_challenge")]
    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.ra.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.unmap.bind_parallel(r_j, BindingOrder::LowToHigh);
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
