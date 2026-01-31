use allocative::Allocative;
use rayon::prelude::*;

use crate::field::JoltField;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::opening_proof::{OpeningPoint, BIG_ENDIAN};
use crate::utils::math::Math;
use crate::utils::thread::unsafe_allocate_zero_vec;

/// Polynomial evaluating to eq+1(x, y) for x in [0, 2^l - 2]
pub struct EqPlusOnePolynomial<F: JoltField> {
    pub x: Vec<F::Challenge>,
}

impl<F: JoltField> EqPlusOnePolynomial<F> {
    pub fn new(x: Vec<F::Challenge>) -> Self {
        EqPlusOnePolynomial { x }
    }

    /* This MLE is 1 if y = x + 1 for x in the range [0... 2^l-2].
    That is, it ignores the case where x is all 1s, outputting 0.
    Assumes x and y are provided big-endian. */
    pub fn evaluate(&self, y: &[F::Challenge]) -> F {
        let l = self.x.len();
        let x = &self.x;
        assert!(y.len() == l);
        let one = F::from_u64(1_u64);

        /* If y+1 = x, then the two bit vectors are of the following form.
            Let k be the longest suffix of 1s in x.
            In y, those k bits are 0.
            Then, the next bit in x is 0 and the next bit in y is 1.
            The remaining higher bits are the same in x and y.
        */
        (0..l)
            .into_par_iter()
            .map(|k| {
                let lower_bits_product = (0..k)
                    .map(|i| x[l - 1 - i] * (F::one() - y[l - 1 - i]))
                    .product::<F>();
                let kth_bit_product = (F::one() - x[l - 1 - k]) * y[l - 1 - k];
                let higher_bits_product = ((k + 1)..l)
                    .map(|i| {
                        x[l - 1 - i] * y[l - 1 - i] + (one - x[l - 1 - i]) * (one - y[l - 1 - i])
                    })
                    .product::<F>();
                lower_bits_product * kth_bit_product * higher_bits_product
            })
            .sum()
    }

    #[tracing::instrument(skip_all, "EqPlusOnePolynomial::evals")]
    pub fn evals(r: &[F::Challenge], scaling_factor: Option<F>) -> (Vec<F>, Vec<F>) {
        let ell = r.len();
        // 分配内存。ell.pow2() 即 2^ell
        let mut eq_evals: Vec<F> = unsafe_allocate_zero_vec(ell.pow2());
        eq_evals[0] = scaling_factor.unwrap_or(F::one());
        let mut eq_plus_one_evals: Vec<F> = unsafe_allocate_zero_vec(ell.pow2());

        // [算法: 标准 Eq 向量的增量更新]
        // 这是一个辅助闭包，它将现有的 eq_evals 按照新的随机点维度 r[i] 进行“分裂”。
        // 原本的每个值 v 会变成 v*(1-r[i]) 和 v*r[i]。
        let eq_evals_helper = |eq_evals: &mut Vec<F>, r: &[F::Challenge], i: usize| {
            debug_assert!(i != 0);
            let step = 1 << (ell - i);

            // 使用 Rayon 进行并行处理，提高在大规模 Trace 下的速度
            let mut selected: Vec<_> = eq_evals.par_iter_mut().step_by(step).collect();

            selected.par_chunks_mut(2).for_each(|chunk| {
                // chunk[0] 存储 v，chunk[1] 存储 0
                *chunk[1] = *chunk[0] * r[i - 1]; // 变成 v * r[i]
                *chunk[0] -= *chunk[1];          // 变成 v - v * r[i] = v * (1 - r[i])
            });
        };

        // [算法: 核心循环 - 同时计算 Eq 和 Eq+1]
        for i in 0..ell {
            let step = 1 << (ell - i);
            let half_step = step / 2;

            // [算法: 预计算后缀乘积]
            // 为了计算 Eq(x+1, r)，当第 i 位发生进位时，更高位 (i+1...) 会被激活。
            // 这里计算 (1 - r[i]) * r[i+1] * r[i+2] * ...
            let mut r_lower_product = F::one();
            for &x in r.iter().skip(i + 1) {
                r_lower_product = r_lower_product * x;
            }
            r_lower_product *= F::one() - r[i];

            // [算法原理: 捕获进位贡献]
            // EqPlusOne 的逻辑：如果在第 i 位发生了进位，那么低位部分必然是全 1。
            // 我们利用当前已经算好的部分 Eq 评估值，乘以预计算的后缀乘积，
            // 直接填入 eq_plus_one_evals 对应的索引位置。
            eq_plus_one_evals
                .par_iter_mut()
                .enumerate()
                .skip(half_step)
                .step_by(step)
                .for_each(|(index, v)| {
                    *v = eq_evals[index - half_step] * r_lower_product;
                });

            // 更新标准 Eq 向量，为下一轮迭代做准备
            eq_evals_helper(&mut eq_evals, r, i + 1);
        }

        (eq_evals, eq_plus_one_evals)
    }

    /// Generate prefix-suffix decomposition of eq+1 polynomial
    /// for use in prefix-suffix sumcheck optimization.
    ///
    /// Returns EqPlusOnePrefixSuffixPoly with prefix and suffix evaluations.
    pub fn prefix_suffix(r: &OpeningPoint<BIG_ENDIAN, F>) -> EqPlusOnePrefixSuffixPoly<F> {
        EqPlusOnePrefixSuffixPoly::new(r)
    }

    /// Evaluate the MLE of eq+1 at the point (r, s)
    pub fn mle(r: &[F::Challenge], s: &[F::Challenge]) -> F {
        Self::new(r.to_vec()).evaluate(s)
    }
}

/// Prefix-suffix decomposition of eq+1 polynomial for sumcheck optimization.
///
/// Decomposes eq+1((r_hi, r_lo), (y_hi, y_lo)) as:
///   prefix_0(r_lo, y_lo) * suffix_0(r_hi, y_hi) +
///   prefix_1(r_lo, y_lo) * suffix_1(r_hi, y_hi)
#[derive(Allocative)]
pub struct EqPlusOnePrefixSuffixPoly<F: JoltField> {
    /// Evals of `eq+1(r_lo, j)` for all j in the hypercube.
    pub prefix_0: Vec<F>,
    /// Evals of `eq(r_hi, j)` for all j in the hypercube.
    pub suffix_0: Vec<F>,
    /// Evals of `is_max(r_lo) * is_min(j)` for all j in the hypercube.
    /// Where `is_max(x) = eq((1)^n, x)`, `is_min(x) = eq((0)^n, x)`.
    /// Note: This is non-zero in 1 position but doesn't matter for perf.
    pub prefix_1: Vec<F>,
    /// Evals of `eq+1(r_hi, j)` for all j in the hypercube.
    pub suffix_1: Vec<F>,
}

impl<F: JoltField> EqPlusOnePrefixSuffixPoly<F> {
    pub fn new(r: &OpeningPoint<BIG_ENDIAN, F>) -> Self {
        // [算法原理: Variable Splitting (变量拆分)]
        // 将随机点 r (长度为 n) 对半拆分为高位 r_hi 和低位 r_lo。
        // 这对应了 Tensor Product 结构：Eq(x, r) = Eq(x_hi, r_hi) * Eq(x_lo, r_lo)
        let (r_hi, r_lo) = r.split_at(r.len() / 2);

        // [算法原理: Boundary Condition (边界检查)]
        // is_max_eval 用于处理进位逻辑。
        // 当低位索引 x_lo 全为 1 时（例如 0111），x_lo + 1 会产生进位，影响高位 x_hi。
        // EqPolynomial::mle 会计算：当变量全为 1 时，Eq(x_lo, r_lo) 的评估值。
        let is_max_eval = EqPolynomial::mle(&vec![F::one(); r_lo.len()], &r_lo.r);

        // 初始化前缀进位多项式 (Prefix 1)
        // 这个向量代表了：如果 x_lo 发生了进位，低位部分变成了全 0 (即索引 0)。
        let mut prefix_1_evals = vec![F::zero(); 1 << r_lo.len()];
        prefix_1_evals[0] = is_max_eval;

        // [算法原理: EqPlusOne 递归评估]
        // 核心算法：同时计算 Eq(x, r) 和 Eq(x+1, r) 的所有评估值。
        // 对于高位部分 r_hi：
        // - suffix_0 对应没有从低位进位过来的情况：Eq(x_hi, r_hi)
        // - suffix_1 对应低位进位过来的情况：Eq(x_hi + 1, r_hi)
        let (suffix_0, suffix_1) = EqPlusOnePolynomial::<F>::evals(&r_hi.r, None);

        Self {
            // [算法原理: Tensor Product 组装]

            // prefix_0: 正常情况下的低位部分。
            // 这里取 .1 是因为 EqPlusOnePolynomial 返回的是 (Eq, Eq+1)，
            // 在 Jolt 的特定实现中，这里映射的是 Eq(x_lo + 1, r_lo)。
            prefix_0: EqPlusOnePolynomial::<F>::evals(&r_lo.r, None).1,

            // suffix_0: 对应 Eq(x_hi, r_hi)
            suffix_0,

            // prefix_1: 只有当 x_lo 为全 1 且进位时才非零的特殊项
            prefix_1: prefix_1_evals,

            // suffix_1: 对应 Eq(x_hi + 1, r_hi)
            suffix_1,
        }
    }
}

#[cfg(test)]
mod tests {
    use ark_bn254::Fr;

    use crate::poly::{
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
        opening_proof::{OpeningPoint, BIG_ENDIAN},
    };

    use super::{EqPlusOnePolynomial, EqPlusOnePrefixSuffixPoly};

    #[test]
    fn test_eq_prefix_suffix() {
        let r = OpeningPoint::<BIG_ENDIAN, Fr>::new([9, 2, 3, 7].map(<_>::into).to_vec());
        let eq_plus_one_gt = EqPlusOnePolynomial::new(r.r.clone());
        let r_prime = OpeningPoint::<BIG_ENDIAN, Fr>::new([4, 3, 2, 8].map(<_>::into).to_vec());
        let (r_prime_hi, r_prime_lo) = r_prime.split_at(2);

        let EqPlusOnePrefixSuffixPoly {
            prefix_0,
            suffix_0,
            prefix_1,
            suffix_1,
        } = EqPlusOnePrefixSuffixPoly::new(&r);

        assert_eq!(
            MultilinearPolynomial::from(prefix_0).evaluate(&r_prime_lo.r)
                * MultilinearPolynomial::from(suffix_0).evaluate(&r_prime_hi.r)
                + MultilinearPolynomial::from(prefix_1).evaluate(&r_prime_lo.r)
                    * MultilinearPolynomial::from(suffix_1).evaluate(&r_prime_hi.r),
            eq_plus_one_gt.evaluate(&r_prime.r)
        );
    }
}
