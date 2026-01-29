#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use crate::field::JoltField;
use crate::poly::opening_proof::{ProverOpeningAccumulator, VerifierOpeningAccumulator};
use crate::poly::unipoly::{CompressedUniPoly, UniPoly};
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use crate::subprotocols::sumcheck_verifier::SumcheckInstanceVerifier;
use crate::transcripts::{AppendToTranscript, Transcript};
use crate::utils::errors::ProofVerifyError;
#[cfg(not(target_arch = "wasm32"))]
use crate::utils::profiling::print_current_memory_usage;

use ark_serialize::*;
use std::marker::PhantomData;

/// Implements the standard technique for batching parallel sumchecks to reduce
/// verifier cost and proof size.
///
/// For details, refer to Jim Posen's ["Perspectives on Sumcheck Batching"](https://hackmd.io/s/HyxaupAAA).
/// We do what they describe as "front-loaded" batch sumcheck.
pub enum BatchedSumcheck {}
impl BatchedSumcheck {
    pub fn prove<F: JoltField, ProofTranscript: Transcript>(
        // sumcheck_instances: 包含多个多项式实例的列表，每个可能有不同的变量数（轮数）
        mut sumcheck_instances: Vec<&mut dyn SumcheckInstanceProver<F, ProofTranscript>>,
        opening_accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut ProofTranscript,
    ) -> (SumcheckInstanceProof<F, ProofTranscript>, Vec<F::Challenge>) {

        // 1. 确定最大轮数 N。这将是主要的 Sumcheck 循环次数。
        let max_num_rounds = sumcheck_instances
            .iter()
            .map(|sumcheck| sumcheck.num_rounds())
            .max()
            .unwrap();

        // 2. 将每个实例声称的求和结果（Claim）写入 Transcript，用于生成随机数
        sumcheck_instances.iter().for_each(|sumcheck| {
            let input_claim = sumcheck.input_claim(opening_accumulator);
            transcript.append_scalar(&input_claim);
        });

        // 3. 生成批量系数 alpha_i。
        // 数学对应：Transcript -> [α_0, α_1, ..., α_k]
        let batching_coeffs: Vec<F> = transcript.challenge_vector(sumcheck_instances.len());

        // 4. 对齐 Claims（数学原理 C）。
        // 如果一个实例只有 M 轮，而最大轮数是 N，我们需要将它视为 N 轮。
        // 这意味着前 N-M 个变量是“虚拟”的。
        // 对虚拟变量求和会导致总和乘以 2^(N-M)。
        // 公式：Claim_aligned = Claim_original * 2^(max_rounds - my_rounds)
        let mut individual_claims: Vec<F> = sumcheck_instances
            .iter()
            .map(|sumcheck| {
                let num_rounds = sumcheck.num_rounds();
                let input_claim = sumcheck.input_claim(opening_accumulator);
                input_claim.mul_pow_2(max_num_rounds - num_rounds)
            })
            .collect();

        // 仅用于测试：计算预期的合并 Claim
        #[cfg(test)]
        let mut batched_claim: F = individual_claims
            .iter()
            .zip(batching_coeffs.iter())
            .map(|(claim, coeff)| *claim * coeff)
            .sum();

        let mut r_sumcheck: Vec<F::Challenge> = Vec::with_capacity(max_num_rounds);
        let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(max_num_rounds);
        // 计算 1/2，用于处理虚拟轮次的常数多项式
        let two_inv = F::from_u64(2).inverse().unwrap();

        // ================== Sumcheck 主循环 (0 到 N-1) ==================
        for round in 0..max_num_rounds {
            #[cfg(not(target_arch = "wasm32"))]
            {
                let label = format!("Sumcheck round {round}");
                print_current_memory_usage(label.as_str());
            }

            // 5. 计算当前轮次每个实例的单变量多项式 g_j(X)
            let univariate_polys: Vec<UniPoly<F>> = sumcheck_instances
                .iter_mut()
                .zip(individual_claims.iter())
                .map(|(sumcheck, previous_claim)| {
                    let num_rounds = sumcheck.num_rounds();
                    // 计算该实例在全局轮次中的“激活”起始点。
                    // 假如 max=5, num=3，则 offset=2。意味着前2轮是虚拟的，后3轮是真实的。
                    let offset = sumcheck.round_offset(max_num_rounds);

                    // 判断当前轮次是否属于该实例的真实变量范围
                    let active = round >= offset && round < offset + num_rounds;

                    if active {
                        // 情况 A：真实轮次。
                        // 调用实例的 compute_message 计算真实的单变量多项式 g(X)
                        // 参数 round - offset 是传给实例的本地轮次索引 (0 到 num-1)
                        sumcheck.compute_message(round - offset, *previous_claim)
                    } else {
                        // 情况 B：虚拟轮次 (Dummy Round)。
                        // 多项式不依赖于当前变量 X。
                        // 此时 g(X) 必须满足 g(0) + g(1) = previous_claim。
                        // 且 g(0) == g(1)。
                        // 所以 g(X) = previous_claim / 2。
                        // 这是一个常数多项式（度为0）。
                        UniPoly::from_coeff(vec![*previous_claim * two_inv])
                    }
                })
                .collect();

            // 6. 合并多项式。
            // S(X) = \sum alpha_j * g_j(X)
            let batched_univariate_poly: UniPoly<F> =
                univariate_polys.iter().zip(&batching_coeffs).fold(
                    UniPoly::from_coeff(vec![]),
                    |mut batched_poly, (poly, &coeff)| {
                        // 累加：Poly += Poly_j * alpha_j
                        batched_poly += &(poly * coeff);
                        batched_poly
                    },
                );

            // 压缩多项式（只保留必要的系数，例如二次多项式只需发2个点或系数）
            let compressed_poly = batched_univariate_poly.compress();

            // 7. 发送证明消息并获取挑战 r。
            // Fiat-Shamir: Transcript <- S(X), Transcript -> r_j
            compressed_poly.append_to_transcript(transcript);
            let r_j = transcript.challenge_scalar_optimized::<F>();
            r_sumcheck.push(r_j);

            // 8. 更新每个实例的 Claim。
            // NewClaim_j = g_j(r_j)
            // 这一步对于下一轮是必须的，因为下一轮需要满足 sum(g_{next}) = NewClaim_j
            individual_claims
                .iter_mut()
                .zip(univariate_polys.into_iter())
                .for_each(|(claim, poly)| *claim = poly.evaluate(&r_j));

            #[cfg(test)]
            {
                // 完整性检查：确保 g(0) + g(1) = Claim
                let h0 = batched_univariate_poly.evaluate::<F>(&F::zero());
                let h1 = batched_univariate_poly.evaluate::<F>(&F::one());
                assert_eq!(
                    h0 + h1,
                    batched_claim,
                    "round {round}: H(0) + H(1) = {h0} + {h1} != {batched_claim}"
                );
                batched_claim = batched_univariate_poly.evaluate(&r_j);
            }

            // 9. 让活跃的实例“消化”挑战 r_j。
            // 这通常意味着实例内部会进行部分求值 (Partial Evaluation)，折叠多项式。
            for sumcheck in sumcheck_instances.iter_mut() {
                let num_rounds = sumcheck.num_rounds();
                let offset = sumcheck.round_offset(max_num_rounds);
                let active = round >= offset && round < offset + num_rounds;
                if active {
                    // 注意：只将挑战传给正处于活跃状态的实例
                    sumcheck.ingest_challenge(r_j, round - offset);
                }
            }

            compressed_polys.push(compressed_poly);
        }

        // 10. 收尾工作
        for sumcheck in sumcheck_instances.iter_mut() {
            sumcheck.finalize();
        }

        // 11. 缓存最终的 Openings (Evaluate at r1, ..., rN)
        // 这些 Openings 稍后需要通过 PCS (Polynomial Commitment Scheme) 或其他方式证明。
        let max_num_rounds = sumcheck_instances
            .iter()
            .map(|sumcheck| sumcheck.num_rounds())
            .max()
            .unwrap();

        for sumcheck in sumcheck_instances.iter() {
            let offset = sumcheck.round_offset(max_num_rounds);
            // 提取该实例对应的随机数切片。
            // 如果实例只有后3轮是活跃的，它只取后3个随机数。
            let r_slice = &r_sumcheck[offset..offset + sumcheck.num_rounds()];

            sumcheck.cache_openings(opening_accumulator, transcript, r_slice);
        }

        (SumcheckInstanceProof::new(compressed_polys), r_sumcheck)
    }

    pub fn verify<F: JoltField, ProofTranscript: Transcript>(
        proof: &SumcheckInstanceProof<F, ProofTranscript>,
        sumcheck_instances: Vec<&dyn SumcheckInstanceVerifier<F, ProofTranscript>>,
        opening_accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut ProofTranscript,
    ) -> Result<Vec<F::Challenge>, ProofVerifyError> {
        let max_degree = sumcheck_instances
            .iter()
            .map(|sumcheck| sumcheck.degree())
            .max()
            .unwrap();
        let max_num_rounds = sumcheck_instances
            .iter()
            .map(|sumcheck| sumcheck.num_rounds())
            .max()
            .unwrap();

        // Append input claims to transcript
        sumcheck_instances.iter().for_each(|sumcheck| {
            let input_claim = sumcheck.input_claim(opening_accumulator);
            transcript.append_scalar(&input_claim);
        });

        let batching_coeffs: Vec<F> = transcript.challenge_vector(sumcheck_instances.len());

        // To see why we may need to scale by a power of two, consider a batch of
        // two sumchecks:
        //   claim_a = \sum_x P(x)             where x \in {0, 1}^M
        //   claim_b = \sum_{x, y} Q(x, y)     where x \in {0, 1}^M, y \in {0, 1}^N
        // Then the batched sumcheck is:
        //   \sum_{x, y} A * P(x) + B * Q(x, y)  where A and B are batching coefficients
        //   = A * \sum_y \sum_x P(x) + B * \sum_{x, y} Q(x, y)
        //   = A * \sum_y claim_a + B * claim_b
        //   = A * 2^N * claim_a + B * claim_b
        let claim: F = sumcheck_instances
            .iter()
            .zip(batching_coeffs.iter())
            .map(|(sumcheck, coeff)| {
                let num_rounds = sumcheck.num_rounds();
                let input_claim = sumcheck.input_claim(opening_accumulator);
                input_claim.mul_pow_2(max_num_rounds - num_rounds) * coeff
            })
            .sum();

        let (output_claim, r_sumcheck) =
            proof.verify(claim, max_num_rounds, max_degree, transcript)?;

        let expected_output_claim = sumcheck_instances
            .iter()
            .zip(batching_coeffs.iter())
            .map(|(sumcheck, coeff)| {
                let offset = sumcheck.round_offset(max_num_rounds);
                let r_slice = &r_sumcheck[offset..offset + sumcheck.num_rounds()];

                // Cache polynomial opening claims, to be proven using either an
                // opening proof or sumcheck (in the case of virtual polynomials).
                sumcheck.cache_openings(opening_accumulator, transcript, r_slice);
                let claim = sumcheck.expected_output_claim(opening_accumulator, r_slice);

                claim * coeff
            })
            .sum();

        if output_claim != expected_output_claim {
            return Err(ProofVerifyError::SumcheckVerificationError);
        }

        Ok(r_sumcheck)
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct SumcheckInstanceProof<F: JoltField, ProofTranscript: Transcript> {
    pub compressed_polys: Vec<CompressedUniPoly<F>>,
    _marker: PhantomData<ProofTranscript>,
}

impl<F: JoltField, ProofTranscript: Transcript> SumcheckInstanceProof<F, ProofTranscript> {
    pub fn new(
        compressed_polys: Vec<CompressedUniPoly<F>>,
    ) -> SumcheckInstanceProof<F, ProofTranscript> {
        SumcheckInstanceProof {
            compressed_polys,
            _marker: PhantomData,
        }
    }

    /// Verify this sumcheck proof.
    /// Note: Verification does not execute the final check of sumcheck protocol: g_v(r_v) = oracle_g(r),
    /// as the oracle is not passed in. Expected that the caller will implement.
    ///
    /// Params
    /// - `claim`: Claimed evaluation
    /// - `num_rounds`: Number of rounds of sumcheck, or number of variables to bind
    /// - `degree_bound`: Maximum allowed degree of the combined univariate polynomial
    /// - `transcript`: Fiat-shamir transcript
    ///
    /// Returns (e, r)
    /// - `e`: Claimed evaluation at random point
    /// - `r`: Evaluation point
    pub fn verify(
        &self,
        claim: F,
        num_rounds: usize,
        degree_bound: usize,
        transcript: &mut ProofTranscript,
    ) -> Result<(F, Vec<F::Challenge>), ProofVerifyError> {
        let mut e = claim;
        let mut r: Vec<F::Challenge> = Vec::new();

        // verify that there is a univariate polynomial for each round
        assert_eq!(self.compressed_polys.len(), num_rounds);
        for i in 0..self.compressed_polys.len() {
            // verify degree bound
            if self.compressed_polys[i].degree() > degree_bound {
                return Err(ProofVerifyError::InvalidInputLength(
                    degree_bound,
                    self.compressed_polys[i].degree(),
                ));
            }

            // append the prover's message to the transcript
            self.compressed_polys[i].append_to_transcript(transcript);

            //derive the verifier's challenge for the next round
            let r_i: F::Challenge = transcript.challenge_scalar_optimized::<F>();
            r.push(r_i);

            // evaluate the claimed degree-ell polynomial at r_i using the hint
            e = self.compressed_polys[i].eval_from_hint(&e, &r_i);
        }

        Ok((e, r))
    }
}
