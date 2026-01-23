use std::marker::PhantomData;

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use tracing::info;
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
/// 构建 Uniskip 协议第一轮的单变量多项式。
///
/// 该函数主要执行以下步骤：
/// 1.  **重构 T1 多项式的值域**：将基础评估值 (`base_evals`) 和扩展评估值 (`extended_evals`)
///     合并到一个对称的扩展窗口中。这个窗口覆盖了 $[-DEGREE, DEGREE]$ 的整数点。
/// 2.  **插值 T1**：将上述点值表示转换为系数表示，得到 $T_1(X)$ 的系数。
/// 3.  **构建拉格朗日多项式**：计算在挑战点 `tau_high` 处的拉格朗日基函数值，并插值得到其系数表示。
/// 4.  **多项式乘法（卷积）**：计算 $S_1(X) = \text{Lagrange}(X) \cdot T_1(X)$。
///     在系数表示下，这通过系数的卷积来实现。
///
/// # 泛型参数
/// * `F`: 域元素的类型 (Field Type)。
/// * `DOMAIN_SIZE`: 基础求值域的大小。
/// * `DEGREE`: 扩展窗口的一半大小，即窗口范围大约是 $[-DEGREE, DEGREE]$。
/// * `EXTENDED_SIZE`: 扩展后用于插值 T1 的总点数，通常为 `2 * DEGREE + 1`。
/// * `NUM_COEFFS`: 最终多项式的系数个数，等于两个多项式度数之和加 1。
#[inline]
pub fn build_uniskip_first_round_poly<
    F: JoltField,
    const DOMAIN_SIZE: usize,
    const DEGREE: usize,
    const EXTENDED_SIZE: usize,
    const NUM_COEFFS: usize,
>(
    // 基础窗口内的评估值（可选）。如果为 None，则视为全 0。
    // 这些值对应于以 0 为中心的对称小区间内的点。
    base_evals: Option<&[F; DOMAIN_SIZE]>,
    // 扩展窗口的评估值。
    // 这些是基础窗口之外的点，用于支持更高次数的插值。
    extended_evals: &[F; DEGREE],
    // 来自验证者的随机挑战点 (High bits challenge)。
    tau_high: F::Challenge,
) -> UniPoly<F> {
    // 编译时断言，确保泛型参数满足几何关系
    // EXTENDED_SIZE 是 T1 多项式的点数，覆盖 [-DEGREE, DEGREE] 共 2*DEGREE + 1 个点
    debug_assert_eq!(EXTENDED_SIZE, 2 * DEGREE + 1);
    // NUM_COEFFS 是最终多项式 S1 的系数个数。
    // 它是 (DOMAIN_SIZE - 1) + (EXTENDED_SIZE - 1) + 1 的某种组合，这里简化为 3*DEGREE + 1
    debug_assert_eq!(NUM_COEFFS, 3 * DEGREE + 1);

    // 1. 准备 T1 多项式的点值数据
    // uniskip_targets 返回需要填充的扩展点的坐标索引
    let targets: [i64; DEGREE] = uniskip_targets::<DOMAIN_SIZE, DEGREE>();
    // 初始化 T1 的值数组，大小为 EXTENDED_SIZE
    let mut t1_vals: [F; EXTENDED_SIZE] = [F::zero(); EXTENDED_SIZE];

    // 填充基础窗口的评估值 (Base evaluations)
    // 基础窗口通常是以 0 为中心对于称的，例如 [-1, 0, 1] 对应 DOMAIN_SIZE=3
    if let Some(base) = base_evals {
        // 计算基础窗口最左侧的坐标值
        let base_left: i64 = -((DOMAIN_SIZE as i64 - 1) / 2);
        for (i, &val) in base.iter().enumerate() {
            // z 是实际的整数坐标
            let z = base_left + (i as i64);
            // pos 是将坐标映射到数组索引 [0, EXTENDED_SIZE)。
            // 映射方式是：index = coordinate + DEGREE。
            // 因为 coordinate 最小是 -DEGREE，所以最小 index 为 0。
            let pos = (z + (DEGREE as i64)) as usize;
            t1_vals[pos] = val;
        }
    }

    // 填充扩展评估值 (Extended evaluations)
    // 这些点位于基础窗口之外
    for (idx, &val) in extended_evals.iter().enumerate() {
        let z = targets[idx];
        let pos = (z + (DEGREE as i64)) as usize;
        t1_vals[pos] = val;
    }

    // 2. T1 插值：从点值形式转换到系数形式
    // 这里将填充好的 t1_vals 视为在点 [-DEGREE, ..., DEGREE] 上的值进行插值
    let t1_coeffs = LagrangePolynomial::<F>::interpolate_coeffs::<EXTENDED_SIZE>(&t1_vals);

    // 3. 构建与挑战点相关的拉格朗日多项式
    // 计算拉格朗日基函数在 tau_high 处的值
    let lagrange_values = LagrangePolynomial::<F>::evals::<F::Challenge, DOMAIN_SIZE>(&tau_high);
    // 将这些值插值为系数形式
    let lagrange_coeffs =
        LagrangePolynomial::<F>::interpolate_coeffs::<DOMAIN_SIZE>(&lagrange_values);

    // 4. 计算多项式乘积 (卷积)
    // S1(X) = Lagrange(X) * T1(X)
    // 多项式乘法对应其系数向量的卷积
    let mut s1_coeffs: [F; NUM_COEFFS] = [F::zero(); NUM_COEFFS];
    for (i, &a) in lagrange_coeffs.iter().enumerate() {
        for (j, &b) in t1_coeffs.iter().enumerate() {
            // 卷积公式：c[k] = sum(a[i] * b[j]) where i + j = k
            s1_coeffs[i + j] += a * b;
        }
    }

    // 返回构建好的单变量多项式
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
    // 1. 获取当前轮次的输入声明 (Input Claim)。
    // 在 Sumcheck 协议开始时，这就是 Prover 声称的整个计算的总和 (Total Sum)。
    // 如果这是递归步骤的一部分，它可能是上一层协议产生的 claim。
    // opening_accumulator 负责管理这些声明的累加和验证状态。
    let input_claim = instance.input_claim(opening_accumulator);
    info!("input_claim: {:?}", input_claim);

    // 2. 计算当前轮次（UniSkip 轮）的单变量多项式 g(x)。
    // 核心逻辑：
    // Sumcheck 协议要求 Prover 将多变量多项式 P(x_1, ..., x_n) 在除 x_1 外的所有变量上求和，
    // 得到一个单变量多项式 g(x_1) = \sum_{x_2...x_n} P(x_1, ..., x_n)。
    //
    // 在 Jolt 的 UniSkip 上下文中：
    // 这个 g(x) 通常是一个高次多项式（Degree > 1）。
    // 这里的 `compute_message` 会执行昂贵的“全量扫描”操作：
    // 它会在特定的点（通常是域外的点，即 Extrapolated Points，如 -1, -2...）上评估 g(x)，
    // 而不仅仅是计算系数。这利用了 Jolt 的高性能评估算法。
    // 参数 `0` 通常指示这是 Sumcheck 的第一阶段或特定的 Skip 轮次索引。
    let uni_poly = instance.compute_message(0, input_claim);

    // 3. 将计算出的单变量多项式 g(x) 提交到 Transcript。
    // 这一步对应 Sumcheck 协议中 Prover 将 g(x) 发送给 Verifier。
    // 实际上发送的是多项式在特定点上的评估值（Evaluations），或者是压缩后的系数。
    // 这一步是 Fiat-Shamir 变换的一部分，用于让 Verifier (或 Transcript) 稍后生成随机挑战。
    uni_poly.append_to_transcript(transcript);

    // 4. 从 Transcript 中生成随机挑战点 r。
    // 这是 Fiat-Shamir 启发式：模拟 Verifier 发送一个随机数 r (F::Challenge)。
    // 这个 r 将被选为多项式第一个变量（或 UniSkip 这一批变量）的固定值。
    let r0: F::Challenge = transcript.challenge_scalar_optimized::<F>();
    info!("r0: {:?}", r0);

    // 5. 更新实例状态：绑定变量并计算新的 Claim。
    // 这一步非常关键，它完成了 Sumcheck 的状态转移：
    // 1. Binding (绑定): 将多变量多项式中的对应变量固定为 r0。
    //    (注意：在 UniSkip 中，这可能意味着一次性绑定了多个原始布尔变量，或者在一个扩展域上进行绑定)。
    // 2. Evaluate (求值): 计算 g(r0)。
    //    根据 Sumcheck 协议，g(r0) 将成为下一轮 Sumcheck 的目标声明 (Next Claim)。
    // 3. Update (更新): 将 accumulator 更新为这个新值，准备进入下一阶段（例如常规的 Sumcheck 轮次或结束）。
    //
    // 或者处理高维度的折叠。本质是将问题规模缩小，并将验证责任转移到 g(r0) 上。
    instance.cache_openings(opening_accumulator, transcript, &[r0]);

    // 6. 构造并返回证明结构体。
    // 这个 Proof 包含了一开始计算出的多项式 g(x)（通常是压缩形式或评估值形式）。
    // Verifier 随后会利用这个 Proof 来执行自己的检查：
    // 验证 g(0) + g(1) == input_claim，并计算 g(r0) 以推进验证。
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
