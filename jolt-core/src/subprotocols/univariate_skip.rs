use std::marker::PhantomData;

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

use crate::field::JoltField;
use crate::poly::lagrange_poly::LagrangePolynomial;
use crate::poly::opening_proof::{ProverOpeningAccumulator, VerifierOpeningAccumulator};
use crate::poly::unipoly::UniPoly;
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use crate::subprotocols::sumcheck_verifier::SumcheckInstanceVerifier;
use crate::transcripts::{AppendToTranscript, Transcript};
use crate::utils::errors::ProofVerifyError;

/// Returns the interleaved symmetric univariate-skip target indices outside the base window.
///
/// Domain is assumed to be the canonical symmetric window of size DOMAIN_SIZE with
/// base indices from start = -((DOMAIN_SIZE-1)/2) to end = start + DOMAIN_SIZE - 1.
///
/// Targets are the extended points z ∈ {−DEGREE..−1} ∪ {1..DEGREE}, interleaved as
/// [start-1, end+1, start-2, end+2, ...] until DEGREE points are produced.
#[inline]
pub const fn uniskip_targets<const DOMAIN_SIZE: usize, const DEGREE: usize>() -> [i64; DEGREE] {
    let d: i64 = DEGREE as i64;
    let ext_left: i64 = -d;
    let ext_right: i64 = d;
    let base_left: i64 = -((DOMAIN_SIZE as i64 - 1) / 2);
    let base_right: i64 = base_left + (DOMAIN_SIZE as i64) - 1;

    let mut targets: [i64; DEGREE] = [0; DEGREE];
    let mut idx = 0usize;
    let mut n = base_left - 1;
    let mut p = base_right + 1;

    while n >= ext_left && p <= ext_right && idx < DEGREE {
        targets[idx] = n;
        idx += 1;
        if idx >= DEGREE {
            break;
        }
        targets[idx] = p;
        idx += 1;
        n -= 1;
        p += 1;
    }

    while idx < DEGREE && n >= ext_left {
        targets[idx] = n;
        idx += 1;
        n -= 1;
    }

    while idx < DEGREE && p <= ext_right {
        targets[idx] = p;
        idx += 1;
        p += 1;
    }

    targets
}

/// Builds the uni-skip first-round polynomial s1 from base and extended evaluations of t1.
///
/// SPECIFIC: This helper targets the setting where s1(Y) = L(τ_high, Y) · t1(Y), with L the
/// degree-(DOMAIN_SIZE-1) Lagrange kernel over the base window and t1 a univariate of degree
/// at most 2·DEGREE (extended symmetric window size EXTENDED_SIZE = 2·DEGREE + 1).
/// Consequently, the resulting s1 has degree at most 3·DEGREE (NUM_COEFFS = 3·DEGREE + 1).
///
/// Inputs:
/// - base_evals: optional t1 evaluations on the base window (symmetric grid of size DOMAIN_SIZE).
///   When `None`, base evaluations are treated as all zeros.
/// - extended_evals: t1 evaluated on the extended symmetric grid outside the base window,
///   in the order given by `uniskip_targets::<DOMAIN_SIZE, DEGREE>()`.
/// - tau_high: the challenge used in the Lagrange kernel L(τ_high, ·) over the base window.
///
/// Returns: UniPoly s1 with exactly NUM_COEFFS coefficients.
#[inline]
pub fn build_uniskip_first_round_poly<
    F: JoltField,
    const DOMAIN_SIZE: usize,
    const DEGREE: usize,
    const EXTENDED_SIZE: usize,
    const NUM_COEFFS: usize,
>(
    base_evals: Option<&[F; DOMAIN_SIZE]>,
    extended_evals: &[F; DEGREE],
    tau_high: F::Challenge,
) -> UniPoly<F> {
    debug_assert_eq!(EXTENDED_SIZE, 2 * DEGREE + 1);
    debug_assert_eq!(NUM_COEFFS, 3 * DEGREE + 1);

    // Rebuild t1 on the full extended symmetric window
    let targets: [i64; DEGREE] = uniskip_targets::<DOMAIN_SIZE, DEGREE>();
    let mut t1_vals: [F; EXTENDED_SIZE] = [F::zero(); EXTENDED_SIZE];

    // Fill in base window evaluations when provided (otherwise treated as zeros)
    if let Some(base) = base_evals {
        let base_left: i64 = -((DOMAIN_SIZE as i64 - 1) / 2);
        for (i, &val) in base.iter().enumerate() {
            let z = base_left + (i as i64);
            let pos = (z + (DEGREE as i64)) as usize;
            t1_vals[pos] = val;
        }
    }

    // Fill in extended evaluations (outside base window)
    for (idx, &val) in extended_evals.iter().enumerate() {
        let z = targets[idx];
        let pos = (z + (DEGREE as i64)) as usize;
        t1_vals[pos] = val;
    }

    let t1_coeffs = LagrangePolynomial::<F>::interpolate_coeffs::<EXTENDED_SIZE>(&t1_vals);
    let lagrange_values = LagrangePolynomial::<F>::evals::<F::Challenge, DOMAIN_SIZE>(&tau_high);
    let lagrange_coeffs =
        LagrangePolynomial::<F>::interpolate_coeffs::<DOMAIN_SIZE>(&lagrange_values);

    let mut s1_coeffs: [F; NUM_COEFFS] = [F::zero(); NUM_COEFFS];
    for (i, &a) in lagrange_coeffs.iter().enumerate() {
        for (j, &b) in t1_coeffs.iter().enumerate() {
            s1_coeffs[i + j] += a * b;
        }
    }

    UniPoly::from_coeff(s1_coeffs.to_vec())
}

/// Univariate Skip 协议第一轮的 Prover 辅助函数。
///
/// 该函数执行 Sumcheck 协议第一轮（针对 Uni-skip 场景）的核心逻辑：
/// 1. 获取输入声明（Input Claim）。
/// 2. 计算第一轮的单变量多项式。
/// 3. 将多项式提交到 Transcript 以进行 Fiat-Shamir 变换。
/// 4. 获取验证者的随机挑战点 $r_0$。
/// 5. 更新实例状态，为后续轮次做准备。
pub fn prove_uniskip_round<F: JoltField, T: Transcript, I: SumcheckInstanceProver<F, T>>(
    instance: &mut I,
    opening_accumulator: &mut ProverOpeningAccumulator<F>,
    transcript: &mut T,
) -> UniSkipFirstRoundProof<F, T> {
    // 1. 获取实例当前的输入声明（Input Claim）。
    // 在 Sumcheck 开始时，这通常是 Prover 声称的计算总和（Sum）。
    let input_claim = instance.input_claim(opening_accumulator);

    // 2. 计算第 0 轮（即第一轮）的消息。
    // Prover 需要在所有布尔超立方体顶点上对多项式进行求和，将其规约为一个单变量多项式 S(X)。
    // 在 Uni-skip 场景中，这个多项式可能具有特定的度数结构。
    let uni_poly = instance.compute_message(0, input_claim);

    // 3. 将完整的多项式（系数形式）附加到 Transcript 中。
    // 这一步模拟了 Prover 将多项式发送给 Verifier 的过程。
    // Transcript 会吸收这些数据以作为生成后续随机挑战的熵源。
    uni_poly.append_to_transcript(transcript);

    // 4. 从 Transcript 中确定性地派生出随机挑战点 r_0。
    // 这是 Fiat-Shamir 启发式应用，代替了交互式协议中 Verifier 发送随机数的一步。
    let r0: F::Challenge = transcript.challenge_scalar_optimized::<F>();

    // 5. 将挑战点 r_0 缓存到实例中，并利用它更新内部状态。
    // 实际上，这里会执行 Partial Evaluation（部分评估）：
    // 将多变量多项式的第一个变量固定为 r_0，从而将问题规模从 n 个变量减少到 n-1 个变量，
    // 为下一轮 Sumcheck 做准备。同时也会更新 accumulator 中的声明值。
    instance.cache_openings(opening_accumulator, transcript, &[r0]);

    // 6. 将生成的单变量多项式封装在证明结构体中返回。
    UniSkipFirstRoundProof::new(uni_poly)
}

/// The sumcheck proof for a univariate skip round
/// Consists of the (single) univariate polynomial sent in that round, no omission of any coefficient
#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct UniSkipFirstRoundProof<F: JoltField, T: Transcript> {
    pub uni_poly: UniPoly<F>,
    _marker: PhantomData<T>,
}

impl<F: JoltField, T: Transcript> UniSkipFirstRoundProof<F, T> {
    pub fn new(uni_poly: UniPoly<F>) -> Self {
        Self {
            uni_poly,
            _marker: PhantomData,
        }
    }

    /// Verify only the univariate-skip first round.
    ///
    /// Params
    /// - `const N`: the first degree plus one (e.g. the size of the first evaluation domain)
    /// - `const FIRST_ROUND_POLY_NUM_COEFFS`: number of coefficients in the first-round polynomial
    /// - `degree_bound_first`: Maximum allowed degree of the first univariate polynomial
    /// - `transcript`: Fiat-Shamir transcript
    pub fn verify<const N: usize, const FIRST_ROUND_POLY_NUM_COEFFS: usize>(
        proof: &Self,
        sumcheck_instance: &dyn SumcheckInstanceVerifier<F, T>,
        opening_accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
    ) -> Result<(), ProofVerifyError> {
        let degree_bound = sumcheck_instance.degree();
        // Degree check for the high-degree first polynomial
        if proof.uni_poly.degree() > degree_bound {
            return Err(ProofVerifyError::InvalidInputLength(
                degree_bound,
                proof.uni_poly.degree(),
            ));
        }

        // Append full polynomial and derive r0
        proof.uni_poly.append_to_transcript(transcript);
        let r0 = transcript.challenge_scalar_optimized::<F>();

        // Check symmetric-domain sum equals zero (initial claim), and compute next claim s1(r0)
        let input_claim = sumcheck_instance.input_claim(opening_accumulator);
        let ok = proof
            .uni_poly
            .check_sum_evals::<N, FIRST_ROUND_POLY_NUM_COEFFS>(input_claim);
        sumcheck_instance.cache_openings(opening_accumulator, transcript, &[r0]);

        if !ok {
            Err(ProofVerifyError::UniSkipVerificationError)
        } else {
            Ok(())
        }
    }
}
