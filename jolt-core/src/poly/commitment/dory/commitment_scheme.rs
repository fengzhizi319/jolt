//! Dory polynomial commitment scheme implementation

use super::dory_globals::DoryGlobals;
use super::jolt_dory_routines::{JoltG1Routines, JoltG2Routines};
use super::wrappers::{
    jolt_to_ark, ArkDoryProof, ArkFr, ArkG1, ArkGT, ArkworksProverSetup, ArkworksVerifierSetup,
    JoltToDoryTranscript, BN254,
};
use crate::{
    field::JoltField,
    poly::commitment::commitment_scheme::{CommitmentScheme, StreamingCommitmentScheme},
    poly::multilinear_polynomial::MultilinearPolynomial,
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, math::Math, small_scalar::SmallScalar},
};
use ark_bn254::{G1Affine, G1Projective};
use ark_ec::CurveGroup;
use ark_ff::Zero;
use dory::primitives::{
    arithmetic::{Group, PairingCurve},
    poly::Polynomial,
};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;
use rayon::prelude::*;
use sha3::{Digest, Sha3_256};
use std::borrow::Borrow;
use tracing::trace_span;

#[derive(Clone)]
pub struct DoryCommitmentScheme;

impl CommitmentScheme for DoryCommitmentScheme {
    type Field = ark_bn254::Fr;
    type ProverSetup = ArkworksProverSetup;
    type VerifierSetup = ArkworksVerifierSetup;
    type Commitment = ArkGT;
    type Proof = ArkDoryProof;
    type BatchedProof = Vec<ArkDoryProof>;
    type OpeningProofHint = Vec<ArkG1>;

    fn setup_prover(max_num_vars: usize) -> Self::ProverSetup {
        let _span = trace_span!("DoryCommitmentScheme::setup_prover").entered();
        let mut hasher = Sha3_256::new();
        hasher.update(b"Jolt Dory URS seed");
        let hash_result = hasher.finalize();
        let seed: [u8; 32] = hash_result.into();
        let mut rng = ChaCha20Rng::from_seed(seed);
        let setup = ArkworksProverSetup::new_from_urs(&mut rng, max_num_vars);

        // The prepared-point cache in dory-pcs is global and can only be initialized once.
        // In unit tests, multiple setups with different sizes are created, so initializing the
        // cache with a small setup can break later tests that need more generators.
        // We therefore disable cache initialization in `cfg(test)` builds.
        #[cfg(not(test))]
        DoryGlobals::init_prepared_cache(&setup.g1_vec, &setup.g2_vec);

        setup
    }

    fn setup_verifier(setup: &Self::ProverSetup) -> Self::VerifierSetup {
        let _span = trace_span!("DoryCommitmentScheme::setup_verifier").entered();
        setup.to_verifier_setup()
    }

    fn commit(
        poly: &MultilinearPolynomial<ark_bn254::Fr>,
        setup: &Self::ProverSetup,
    ) -> (Self::Commitment, Self::OpeningProofHint) {
        let _span = trace_span!("DoryCommitmentScheme::commit").entered();

        let num_cols = DoryGlobals::get_num_columns();
        let num_rows = DoryGlobals::get_max_num_rows();
        let sigma = num_cols.log_2();
        let nu = num_rows.log_2();

        let (tier_2, row_commitments) = <MultilinearPolynomial<ark_bn254::Fr> as Polynomial<
            ArkFr,
        >>::commit::<BN254, JoltG1Routines>(
            poly, nu, sigma, setup
        )
            .expect("commitment should succeed");

        (tier_2, row_commitments)
    }

    fn batch_commit<U>(
        polys: &[U],
        gens: &Self::ProverSetup,
    ) -> Vec<(Self::Commitment, Self::OpeningProofHint)>
    where
        U: std::borrow::Borrow<MultilinearPolynomial<ark_bn254::Fr>> + Sync,
    {
        let _span = trace_span!("DoryCommitmentScheme::batch_commit").entered();

        polys
            .par_iter()
            .map(|poly| Self::commit(poly.borrow(), gens))
            .collect()
    }

    fn prove<ProofTranscript: Transcript>(
        setup: &Self::ProverSetup,
        poly: &MultilinearPolynomial<ark_bn254::Fr>,
        opening_point: &[<ark_bn254::Fr as JoltField>::Challenge],
        hint: Option<Self::OpeningProofHint>,
        transcript: &mut ProofTranscript,
    ) -> Self::Proof {
        let _span = trace_span!("DoryCommitmentScheme::prove").entered();

        let row_commitments = hint.unwrap_or_else(|| {
            let (_commitment, row_commitments) = Self::commit(poly, setup);
            row_commitments
        });

        let num_cols = DoryGlobals::get_num_columns();
        let num_rows = DoryGlobals::get_max_num_rows();
        let sigma = num_cols.log_2();
        let nu = num_rows.log_2();

        // Dory uses the opposite endian-ness as Jolt
        let ark_point: Vec<ArkFr> = opening_point
            .iter()
            .rev()  // Reverse the order for Dory
            .map(|p| {
                let f_val: ark_bn254::Fr = (*p).into();
                jolt_to_ark(&f_val)
            })
            .collect();

        let mut dory_transcript = JoltToDoryTranscript::<ProofTranscript>::new(transcript);

        dory::prove::<ArkFr, BN254, JoltG1Routines, JoltG2Routines, _, _>(
            poly,
            &ark_point,
            row_commitments,
            nu,
            sigma,
            setup,
            &mut dory_transcript,
        )
            .expect("proof generation should succeed")
    }

    fn verify<ProofTranscript: Transcript>(
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut ProofTranscript,
        opening_point: &[<ark_bn254::Fr as JoltField>::Challenge],
        opening: &ark_bn254::Fr,
        commitment: &Self::Commitment,
    ) -> Result<(), ProofVerifyError> {
        let _span = trace_span!("DoryCommitmentScheme::verify").entered();

        // Dory uses the opposite endian-ness as Jolt
        let ark_point: Vec<ArkFr> = opening_point
            .iter()
            .rev()  // Reverse the order for Dory
            .map(|p| {
                let f_val: ark_bn254::Fr = (*p).into();
                jolt_to_ark(&f_val)
            })
            .collect();
        let ark_eval: ArkFr = jolt_to_ark(opening);

        let mut dory_transcript = JoltToDoryTranscript::<ProofTranscript>::new(transcript);

        dory::verify::<ArkFr, BN254, JoltG1Routines, JoltG2Routines, _>(
            *commitment,
            ark_eval,
            &ark_point,
            proof,
            setup.clone().into_inner(),
            &mut dory_transcript,
        )
            .map_err(|_| ProofVerifyError::InternalError)?;

        Ok(())
    }

    fn protocol_name() -> &'static [u8] {
        b"Dory"
    }

    /// In Dory, the opening proof hint consists of the Pedersen commitments to the rows
    /// of the polynomial coefficient matrix. In the context of a batch opening proof, we
    /// can homomorphically combine the row commitments for multiple polynomials into the
    /// row commitments for the RLC of those polynomials. This is more efficient than computing
    /// the row commitments for the RLC from scratch.
    #[tracing::instrument(skip_all, name = "DoryCommitmentScheme::combine_hints")]
    fn combine_hints(
        hints: Vec<Self::OpeningProofHint>,
        coeffs: &[Self::Field],
    ) -> Self::OpeningProofHint {
        let num_rows = DoryGlobals::get_max_num_rows();

        let mut rlc_hint = vec![ArkG1(G1Projective::zero()); num_rows];
        for (coeff, mut hint) in coeffs.iter().zip(hints.into_iter()) {
            hint.resize(num_rows, ArkG1(G1Projective::zero()));

            let row_commitments: &mut [G1Projective] = unsafe {
                std::slice::from_raw_parts_mut(hint.as_mut_ptr() as *mut G1Projective, hint.len())
            };

            let rlc_row_commitments: &[G1Projective] = unsafe {
                std::slice::from_raw_parts(rlc_hint.as_ptr() as *const G1Projective, rlc_hint.len())
            };

            let _span = trace_span!("vector_scalar_mul_add_gamma_g1_online");
            let _enter = _span.enter();

            // Scales the row commitments for the current polynomial by
            // its coefficient
            jolt_optimizations::vector_scalar_mul_add_gamma_g1_online(
                row_commitments,
                *coeff,
                rlc_row_commitments,
            );

            let _ = std::mem::replace(&mut rlc_hint, hint);
        }

        rlc_hint
    }

    /// Homomorphically combines multiple commitments using a random linear combination.
    /// Computes: sum_i(coeff_i * commitment_i) for the GT elements.
    #[tracing::instrument(skip_all, name = "DoryCommitmentScheme::combine_commitments")]
    fn combine_commitments<C: Borrow<Self::Commitment>>(
        commitments: &[C],
        coeffs: &[Self::Field],
    ) -> Self::Commitment {
        let _span = trace_span!("DoryCommitmentScheme::combine_commitments").entered();

        // Combine GT elements using parallel RLC
        let commitments_vec: Vec<&ArkGT> = commitments.iter().map(|c| c.borrow()).collect();
        coeffs
            .par_iter()
            .zip(commitments_vec.par_iter())
            .map(|(coeff, commitment)| {
                let ark_coeff = jolt_to_ark(coeff);
                ark_coeff * **commitment
            })
            .reduce(ArkGT::identity, |a, b| a + b)
    }
}

impl StreamingCommitmentScheme for DoryCommitmentScheme {
    type ChunkState = Vec<ArkG1>; // Tier 1 commitment chunks

    #[tracing::instrument(skip_all, name = "DoryCommitmentScheme::compute_tier1_commitment")]
    fn process_chunk<T: SmallScalar>(setup: &Self::ProverSetup, chunk: &[T]) -> Self::ChunkState {
        debug_assert_eq!(chunk.len(), DoryGlobals::get_num_columns());

        let row_len = DoryGlobals::get_num_columns();
        let g1_slice =
            unsafe { std::slice::from_raw_parts(setup.g1_vec.as_ptr(), setup.g1_vec.len()) };

        let g1_bases: Vec<G1Affine> = g1_slice[..row_len]
            .iter()
            .map(|g| g.0.into_affine())
            .collect();

        let row_commitment =
            ArkG1(T::msm(&g1_bases[..chunk.len()], chunk).expect("MSM calculation failed."));
        vec![row_commitment]
    }
    /// 处理 One-Hot 编码数据的单个块（Chunk），计算第一层（Tier 1）承诺。
    ///
    /// 在 Jolt/Lasso 证明系统中，经常需要对 One-Hot 形式的数据进行承诺。
    /// 如果将 One-Hot 逻辑展开成巨大的稀疏矩阵计算非常低效，此函数利用其特性进行了优化。
    ///
    /// # 参数
    /// * `setup`: 证明者的公共参数（包含 G1 基点）。
    /// * `onehot_k`: One-Hot 向量的维度（即值的取值范围大小，例如 256）。
    ///   这表示我们实际上是在并行计算 `K` 个多项式的承诺。
    /// * `chunk`: 紧凑表示的输入数据。`chunk[i] = Some(v)` 表示第 `i` 列的值为 `v`。
    ///   这意味着在展开的 One-Hot 矩阵中，第 `v` 行在第 `i` 列为 1，其余行为 0。
    ///
    /// # 优化原理
    /// 将 MSM（系数 * 基点）简化为特定的基点子集求和（Sum of Bases），因为系数非 0 即 1。
    #[tracing::instrument(
        skip_all,
        name = "DoryCommitmentScheme::compute_tier1_commitment_onehot"
    )]
    fn process_chunk_onehot(
        setup: &Self::ProverSetup,
        onehot_k: usize,
        chunk: &[Option<usize>],
    ) -> Self::ChunkState {
        let K = onehot_k; // One-Hot 的大小（并行处理的多项式数量）

        let row_len = DoryGlobals::get_num_columns();
        // 获取 G1 基点切片，用于后续的线性组合
        let g1_slice =
            unsafe { std::slice::from_raw_parts(setup.g1_vec.as_ptr(), setup.g1_vec.len()) };

        // 将基点转换为仿射坐标（Affine），通常在此类批量加法运算中性能更好
        let g1_bases: Vec<G1Affine> = g1_slice[..row_len]
            .iter()
            .map(|g| g.0.into_affine())
            .collect();

        // 1. 索引分组 (Bucket Sort style)
        // 遍历输入块，将每个列索引（col_index）分配给它对应的数值（k）。
        // indices_per_k[k] 存储了所有值为 k 的列位置。
        // 这意味着第 k 个多项式的承诺由这些位置对应的基点组成。
        let mut indices_per_k: Vec<Vec<usize>> = vec![Vec::new(); K];
        for (col_index, k) in chunk.iter().enumerate() {
            if let Some(k) = k {
                indices_per_k[*k].push(col_index);
            }
        }

        // 2. 批量基点加法优化
        // 计算每个 k 对应的基点之和。相当于 coefficient=1 的 MSM。
        // 调用底层优化库进行并行计算。
        let results = jolt_optimizations::batch_g1_additions_multi(&g1_bases, &indices_per_k);

        // 3. 封装结果
        // 将计算得到的 Projective 点转换回 ArkG1 封装类型。
        let mut row_commitments = vec![ArkG1(G1Projective::zero()); K];
        for (k, result) in results.into_iter().enumerate() {
            // 只有当该数值 k 在 chunk 中出现过，才会有非零承诺，否则为零点
            if !indices_per_k[k].is_empty() {
                row_commitments[k] = ArkG1(G1Projective::from(result));
            }
        }
        row_commitments
    }

    /// 计算第二层（Tier 2）的全局承诺。
    ///
    /// 在 Jolt 的流式计算中，Tier 1 阶段已经计算出了每一小块（Chunk）的中间承诺。
    /// 此函数负责将这些分散的 Tier 1 承诺（`chunks`）聚合在一起，形成最终的多项式承诺。
    /// - P1 的中间承诺: [Result_Chunk1_P1, Result_Chunk2_P1],需要聚合
    /// - P2 的中间承诺: [Result_Chunk1_P2, Result_Chunk2_P2]，需要聚合
    /// 核心任务是：
    /// 1. 整理数据布局：如果是 One-Hot 多项式，需要进行矩阵转置；如果是普通多项式，则直接平铺。
    /// 2. 密码学聚合：使用双线性配对（Pairing）将整理后的行承诺与 SRS 中的 G2 元素结合。
    #[tracing::instrument(skip_all, name = "DoryCommitmentScheme::compute_tier2_commitment")]
    fn aggregate_chunks(
        setup: &Self::ProverSetup,
        onehot_k: Option<usize>,
        chunks: &[Self::ChunkState],
    ) -> (Self::Commitment, Self::OpeningProofHint) {
        if let Some(K) = onehot_k {
            // === 分支 1: 处理 One-Hot (Lasso) 多项式 ===
            // 这种情况下，虽然我们在逻辑上是并行处理 K 个多项式，
            // 但输入 `chunks` 是按时间顺序（Time-ordered）流式产生的。
            // chunks[t] 包含第 t 个时间片内所有 K 个多项式的承诺：[C(P0, t), C(P1, t), ..., C(PK, t)]。
            // Dory 协议要求数据按多项式顺序（Poly-ordered）排列：
            // [C(P0, all_t), C(P1, all_t), ..., C(PK, all_t)]。
            // 因此，我们需要执行数据转置（Transpose）。

            let row_len = DoryGlobals::get_num_columns(); // 每个 Chunk 的列数
            let T = DoryGlobals::get_T(); // Trace 总长度
            // 计算每个多项式被分成了多少个 Chunk (即行数)。
            // 例如 trace 长度 1024，每块长度 16，则每个多项式有 64 行。
            let rows_per_k = T / row_len;
            // 计算总共涉及的行承诺数量 (K 个多项式 * 每个多项式的行数)
            let num_rows = K * T / row_len;

            // 初始化结果向量，准备存放重排后的行承诺。
            // 这是一个巨大的扁平化向量，逻辑上分为 K 段，每段长度为 rows_per_k。
            let mut row_commitments = vec![ArkG1(G1Projective::zero()); num_rows];

            // 遍历每个时间切片 chunk_index (t)，模拟流式数据到达的过程
            for (chunk_index, commitments) in chunks.iter().enumerate() {
                // commitments 是当前时间片 t 内所有 K 个多项式的承诺向量：
                // [C(P0, t), C(P1, t), ..., C(PK, t)]

                // 使用 Rayon 并行迭代器将数据分散写入到 row_commitments 的正确位置。
                // 逻辑解释：
                // 我们要把 commitments[k] (第 k 个多项式在时间 t 的值) 放入 row_commitments。
                // 目标位置应该是在第 k 个多项式的区域内的第 t 个偏移处。
                // 即 index = (k * rows_per_k) + chunk_index。
                //
                // 代码实现技巧：
                // 1. skip(chunk_index): 从偏移量 t 开始。
                // 2. step_by(rows_per_k): 每次跳跃一个多项式的长度。
                // 这样迭代产生的索引序列正是：
                // dest[0]: 0*rows_per_k + t  (属于 Poly 0 的第 t 行)
                // dest[1]: 1*rows_per_k + t  (属于 Poly 1 的第 t 行)
                // ...
                // 正好对应 commitments 中的 P0, P1... 的顺序。
                row_commitments
                    .par_iter_mut()
                    .skip(chunk_index)
                    .step_by(rows_per_k)
                    .zip(commitments.par_iter())
                    .for_each(|(dest, src)| *dest = *src);
            }

            // 获取所需的 G2 基点（数量等于行承诺总数）
            let g2_bases = &setup.g2_vec[..row_commitments.len()];
            // 计算最终承诺：Product of Pairings e(RowCommitment_i, G2_i)
            // 这一步将巨大的行承诺向量压缩为一个恒定大小的 GT 元素。
            let tier_2 = <BN254 as PairingCurve>::multi_pair_g2_setup(&row_commitments, g2_bases);

            // 返回最终承诺和整理好的中间行承诺（作为 Hint，用于后续 Open 证明）
            (tier_2, row_commitments)
        } else {
            // === 分支 2: 处理普通（密集）多项式 ===
            // 对于非 One-Hot 的普通多项式（如寄存器读写），数据已经是顺序排列的。
            // chunks 直接包含了 [Chunk0, Chunk1, Chunk2 ...]，不需要转置。

            // 将 `chunks` (Vec<Vec<G1>>) 扁平化为一维向量 `Vec<G1>`。
            let row_commitments: Vec<ArkG1> =
                chunks.iter().flat_map(|chunk| chunk.clone()).collect();

            let g2_bases = &setup.g2_vec[..row_commitments.len()];
            // 同样进行双线性配对计算
            let tier_2 = <BN254 as PairingCurve>::multi_pair_g2_setup(&row_commitments, g2_bases);

            (tier_2, row_commitments)
        }
    }
}
