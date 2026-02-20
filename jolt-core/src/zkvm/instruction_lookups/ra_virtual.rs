use std::sync::Arc;

use crate::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, PolynomialBinding},
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
        },
        ra_poly::RaPolynomial,
        split_eq_poly::GruenSplitEqPolynomial,
        unipoly::UniPoly,
    },
    subprotocols::{
        mles_product_sum::{
            compute_mles_product_sum_evals_sum_of_products_d16,
            compute_mles_product_sum_evals_sum_of_products_d4,
            compute_mles_product_sum_evals_sum_of_products_d8, finish_mles_product_sum_from_evals,
        },
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
    zkvm::{
        config::OneHotParams,
        instruction::LookupQuery,
        instruction_lookups::LOG_K,
        witness::{CommittedPolynomial, VirtualPolynomial},
    },
};
use allocative::Allocative;
use common::constants::XLEN;
use rayon::prelude::*;
use tracer::instruction::Cycle;

#[derive(Allocative, Clone)]
pub struct InstructionRaSumcheckParams<F: JoltField> {
    pub r_cycle: OpeningPoint<BIG_ENDIAN, F>,
    pub r_address: OpeningPoint<BIG_ENDIAN, F>,
    pub one_hot_params: OneHotParams,
    pub gamma_powers: Vec<F>,
    pub n_virtual_ra_polys: usize,
    pub n_committed_ra_polys: usize,
    /// Number of committed ra polynomials that multiply together to
    /// form a single virtual ra polynomial.
    pub n_committed_per_virtual: usize,
}

impl<F: JoltField> InstructionRaSumcheckParams<F> {
    /// 初始化上下文，负责重建完整的地址随机挑战点 `r_address` 并提取周期随机点 `r_cycle`。
    ///
    /// 在 Jolt 中，由于内存地址空间 (M) 可能很大（例如 2^64），直接对整个地址空间进行多项式承诺可能不切实际。
    /// 因此，Read Access (RA) 多项式被切分成了多个“虚拟多项式” (`InstructionRa(i)`)，每个负责地址的一部分比特。
    ///
    /// 此函数的任务是将这些分散的随机点片段（chunks）重新拼接，以还原出 Verifier 在整个地址空间上进行一致性检查所需的完整随机向量。
    ///
    /// # 举例说明
    ///
    /// 假设系统的参数如下：
    /// - `LOG_K` (总地址位数) = 64
    /// - `ra_virtual_log_k_chunk` (每个虚拟多项式负责的地址位数) = 16
    ///
    /// 则 `n_virtual_ra_polys` = 64 / 16 = 4。我们需要 4 个虚拟多项式来覆盖整个地址空间。
    ///
    /// 在 Sumcheck 过程中，Prover 针对这 4 个多项式分别提供了 Opening，对应的随机点结构如下：
    /// - `InstructionRa(0)` 打开点: `P0 = [r_addr_bits_0..16,  r_cycle_bits]`
    /// - `InstructionRa(1)` 打开点: `P1 = [r_addr_bits_16..32, r_cycle_bits]`
    /// - `InstructionRa(2)` 打开点: `P2 = [r_addr_bits_32..48, r_cycle_bits]`
    /// - `InstructionRa(3)` 打开点: `P3 = [r_addr_bits_48..64, r_cycle_bits]`
    ///
    /// 此函数会遍历这 4 个 Opening：
    /// 1. 调用 `split_at_r` 将 `r_cycle` 分离。
    /// 2. 提取前半部分的地址片段 `r_address_chunk`。
    /// 3. 将这些片段依次追加到 `r_address` Vec 中。
    ///
    /// 最终得到的 `r_address` 包含了完整的 64 个比特的随机性：`[r_addr_bits_0..64]`。
    pub fn new(
        one_hot_params: &OneHotParams,
        opening_accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        // 用于存放重建后的完整地址随机挑战点 (长度应为 LOG_K)
        let mut r_address = Vec::new();

        let ra_virtual_log_k_chunk = one_hot_params.lookups_ra_virtual_log_k_chunk;
        let ra_committed_log_k_chunk = one_hot_params.log_k_chunk;

        // 计算每个虚拟多项式包含多少个已提交的多项式块
        let n_committed_per_virtual = ra_virtual_log_k_chunk / ra_committed_log_k_chunk;

        // 计算总共需要多少个虚拟多项式来覆盖整个 LOG_K 地址空间
        let n_virtual_ra_polys = LOG_K / ra_virtual_log_k_chunk;
        let n_committed_ra_polys = LOG_K / ra_committed_log_k_chunk;

        // 遍历所有虚拟多项式，提取地址片段
        for i in 0..n_virtual_ra_polys {
            // 从累加器中获取第 i 个 InstructionRa 多项式的 Opening Point (点 r)
            let (r, _) = opening_accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::InstructionRa(i),
                SumcheckId::InstructionReadRaf,
            );

            // r 的结构是 [address_chunk_randomness, cycle_randomness]
            // split_at_r 将其切分，我们只需要前半部分 (地址片段)
            let (r_address_chunk, _) = r.split_at_r(ra_virtual_log_k_chunk);

            // 将片段拼接到总向量中
            r_address.extend_from_slice(r_address_chunk);
        }

        // 提取 r_cycle (所有 chunk 共享同一个 cycle 随机点，取第 0 个即可)
        let (r, _) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::InstructionRa(0),
            SumcheckId::InstructionReadRaf,
        );
        // split_at 将 r 切分为 [地址部分 (len=chunk), 周期部分 (其余)]
        let (_, r_cycle) = r.split_at(ra_virtual_log_k_chunk);

        // 生成 gamma 的幂次，用于后续在此上下文中线性组合多个多项式项
        let gamma_powers = transcript.challenge_scalar_powers(n_virtual_ra_polys);

        Self {
            r_cycle,
            one_hot_params: one_hot_params.clone(),
            r_address: OpeningPoint::new(r_address),
            gamma_powers,
            n_virtual_ra_polys,
            n_committed_ra_polys,
            n_committed_per_virtual,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for InstructionRaSumcheckParams<F> {
    fn num_rounds(&self) -> usize {
        self.r_cycle.len()
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let mut res = F::zero();

        for i in 0..self.n_virtual_ra_polys {
            let (_, ra_i_claim) = accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::InstructionRa(i),
                SumcheckId::InstructionReadRaf,
            );
            res += self.gamma_powers[i] * ra_i_claim;
        }

        res
    }

    fn degree(&self) -> usize {
        self.n_committed_per_virtual + 1
    }

    fn normalize_opening_point(
        &self,
        challenges: &[<F as JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }
}

#[derive(Allocative)]
pub struct InstructionRaSumcheckProver<F: JoltField> {
    ra_i_polys: Vec<RaPolynomial<u8, F>>,
    eq_poly: GruenSplitEqPolynomial<F>,
    pub params: InstructionRaSumcheckParams<F>,
}

impl<F: JoltField> InstructionRaSumcheckProver<F> {
    #[tracing::instrument(skip_all, name = "InstructionRaSumcheckProver::initialize")]
    pub fn initialize(params: InstructionRaSumcheckParams<F>, trace: &[Cycle]) -> Self {
        // =========================================================
        // 1. 计算拆分后的 r_address
        // Verifier 发来一个针对整个查表空间的随机挑战点 r_address。
        // 将其拆分为对应每个 chunk 维度的小随机点数组 r_address_chunks。
        // =========================================================
        let r_address_chunks = params
            .one_hot_params
            .compute_r_address_chunks::<F>(&params.r_address.r);

        // =========================================================
        // 2. 提取 Trace 中的切片 (Indices)
        // 遍历整个执行痕迹，把操作数大整数转化为查表用的 chunk 数组。
        // H_indices[i][t] 表示：在维度 i，时刻 t 的 chunk 值 (如 0~255)。
        // =========================================================
        let H_indices: Vec<Vec<Option<u8>>> = (0..params.one_hot_params.instruction_d)
            .map(|i| {
                trace
                    .par_iter()
                    .map(|cycle| {
                        // to_lookup_index: 取出 rs1, rs2 等组合成大整数
                        let lookup_index = LookupQuery::<XLEN>::to_lookup_index(cycle);
                        // 提取出第 i 个 byte 切片
                        Some(params.one_hot_params.lookup_index_chunk(lookup_index, i))
                    })
                    .collect()
            })
            .collect();

        // c: 每个虚拟查找(比如一次 ADD)需要多少个物理多项式切片组合
        let n_committed_per_virtual = params.n_committed_per_virtual;
        // gamma 幂次数组: [1, gamma, gamma^2, ...]
        let gamma_powers = &params.gamma_powers;

        // =========================================================
        // 3. 构建惰性查表多项式 (RaPolynomials) 及 Gamma 预缩放优化
        // =========================================================
        let ra_i_polys = H_indices
            .into_par_iter() // 并行处理每一列
            .enumerate()
            .map(|(i, lookup_indices)| {
                // 核心优化：预缩放。
                // 如果当前维度 i 是一组切片 (batch) 的第一个 (i % c == 0)
                let scaling_factor = if i % n_committed_per_virtual == 0 {
                    let batch = i / n_committed_per_virtual; // 计算属于第几个 batch
                    let gamma = gamma_powers[batch];         // 取出对应的权重 gamma^b
                    if gamma != F::one() {
                        Some(gamma)
                    } else {
                        None
                    }
                } else {
                    None // 组内的其他切片不缩放
                };

                // evals_with_scaling: 计算 eq(r, x) 的查表字典 (大小 256)。
                // 如果 scaling_factor 存在，字典里的每个值预先乘以 gamma。
                let eq_evals =
                    EqPolynomial::evals_with_scaling(&r_address_chunks[i], scaling_factor);

                // RaPolynomial 不存具体的域元素，只存索引 `lookup_indices` 和字典 `eq_evals`。
                // 每次需要求值时，执行：eq_evals[ lookup_indices[t] ]
                RaPolynomial::new(Arc::new(lookup_indices), eq_evals)
            })
            .collect();

        // 4. 返回初始化好的 Prover 实例
        Self {
            ra_i_polys,
            // 绑定到时间维度的快速 Eq 多项式 (Gruen 优化)，用于 O(N) 验证
            eq_poly: GruenSplitEqPolynomial::new(&params.r_cycle.r, BindingOrder::LowToHigh),
            params,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for InstructionRaSumcheckProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(skip_all, name = "InstructionRaSumcheckProver::compute_message")]
    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let eq_poly = &self.eq_poly;

        // Compute q(X) = Σ_i ∏_j ra_{i,j}(X,·) on the U_D grid using a *single*
        // split-eq fold. The per-batch γ^i weights have already been absorbed by
        // pre-scaling the first polynomial in each batch (see `initialize`).
        let evals = match self.params.n_committed_per_virtual {
            4 => compute_mles_product_sum_evals_sum_of_products_d4(
                &self.ra_i_polys,
                self.params.n_virtual_ra_polys,
                eq_poly,
            ),
            8 => compute_mles_product_sum_evals_sum_of_products_d8(
                &self.ra_i_polys,
                self.params.n_virtual_ra_polys,
                eq_poly,
            ),
            16 => compute_mles_product_sum_evals_sum_of_products_d16(
                &self.ra_i_polys,
                self.params.n_virtual_ra_polys,
                eq_poly,
            ),
            n => unimplemented!("{n}"),
        };

        finish_mles_product_sum_from_evals(&evals, previous_claim, eq_poly)
    }

    #[tracing::instrument(skip_all, name = "InstructionRaSumcheckProver::ingest_challenge")]
    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.ra_i_polys
            .iter_mut()
            .for_each(|p| p.bind_parallel(r_j, BindingOrder::LowToHigh));
        self.eq_poly.bind(r_j);
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let r_cycle = self.params.normalize_opening_point(sumcheck_challenges);

        // Compute r_address_chunks with proper padding
        let r_address_chunks = self
            .params
            .one_hot_params
            .compute_r_address_chunks::<F>(&self.params.r_address.r);

        for (i, r_address) in r_address_chunks.into_iter().enumerate() {
            // Undo the per-batch γ scaling applied in `initialize` before caching openings,
            // so the claimed openings match the *actual* committed polynomials.
            let mut claim = self.ra_i_polys[i].final_sumcheck_claim();
            if i % self.params.n_committed_per_virtual == 0 {
                let batch = i / self.params.n_committed_per_virtual;
                let gamma = self.params.gamma_powers[batch];
                if gamma != F::one() {
                    claim = claim / gamma;
                }
            }
            accumulator.append_sparse(
                transcript,
                vec![CommittedPolynomial::InstructionRa(i)],
                SumcheckId::InstructionRaVirtualization,
                r_address,
                r_cycle.r.clone(),
                vec![claim],
            );
        }
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

/// Instruction read-access (RA) virtualization sumcheck.
///
/// A sumcheck instance for:
///
/// ```text
/// sum_x eq(r_cycle, x) * sum_{i=0}^{N-1} [ gamma^i * VirtualRa_i(x) ]
/// ```
///
/// Where each `VirtualRa_i` corresponds to a chunk of the address space and is composed
/// of the product of `M` committed polynomials:
///
/// ```text
/// VirtualRa_i(x) = prod_{j=0}^{M-1} CommittedRa_{i*M+j}(x)
/// ```
///
/// Here:
/// - `N` is the number of virtual polynomials.
/// - `M` is the fan-in of committed polynomials required to reconstruct one virtual polynomial.
pub struct RaSumcheckVerifier<F: JoltField> {
    params: InstructionRaSumcheckParams<F>,
}

impl<F: JoltField> RaSumcheckVerifier<F> {
    pub fn new(
        one_hot_params: &OneHotParams,
        opening_accumulator: &VerifierOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let params =
            InstructionRaSumcheckParams::new(one_hot_params, opening_accumulator, transcript);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for RaSumcheckVerifier<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let r = self.params.normalize_opening_point(sumcheck_challenges);
        let eq_eval = EqPolynomial::mle_endian(&self.params.r_cycle, &r);

        // Claims of the committed ra polynomials.
        let mut committed_ra_claims = (0..self.params.n_committed_ra_polys).map(|i| {
            let (_, ra_i_claim) = accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::InstructionRa(i),
                SumcheckId::InstructionRaVirtualization,
            );
            ra_i_claim
        });

        // Compute sum_i VirtualRa_i(r)
        let mut ra_acc = F::zero();
        for i in 0..self.params.n_virtual_ra_polys {
            let committed_ra_prod = (&mut committed_ra_claims)
                .take(self.params.n_committed_per_virtual)
                .product::<F>();
            ra_acc += self.params.gamma_powers[i] * committed_ra_prod;
        }

        eq_eval * ra_acc
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let r_cycle = self.params.normalize_opening_point(sumcheck_challenges);

        // Compute r_address_chunks with proper padding
        let r_address_chunks = self
            .params
            .one_hot_params
            .compute_r_address_chunks::<F>(&self.params.r_address.r);

        for (i, r_address) in r_address_chunks.iter().enumerate() {
            let opening_point = [r_address.as_slice(), r_cycle.r.as_slice()].concat();

            accumulator.append_sparse(
                transcript,
                vec![CommittedPolynomial::InstructionRa(i)],
                SumcheckId::InstructionRaVirtualization,
                opening_point,
            );
        }
    }
}
