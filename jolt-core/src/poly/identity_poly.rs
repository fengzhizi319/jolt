use std::sync::{Arc, RwLock};

use allocative::Allocative;
use num::Integer;

use crate::field::{ChallengeFieldOps, FieldChallengeOps, JoltField};
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::poly::prefix_suffix::{
    CachedPolynomial, Prefix, PrefixCheckpoints, PrefixPolynomial, PrefixRegistry,
    PrefixSuffixPolynomial, SuffixPolynomial,
};
use crate::utils::lookup_bits::LookupBits;
use crate::utils::math::Math;
use crate::utils::uninterleave_bits;

use super::multilinear_polynomial::{BindingOrder, PolynomialBinding, PolynomialEvaluation};

#[derive(Clone, Debug, Allocative)]
pub struct IdentityPolynomial<F: JoltField> {
    num_vars: usize,
    num_bound_vars: usize,
    bound_value: F,
}

impl<F: JoltField> IdentityPolynomial<F> {
    pub fn new(num_vars: usize) -> Self {
        IdentityPolynomial {
            num_vars,
            num_bound_vars: 0,
            bound_value: F::zero(),
        }
    }
}

impl<F: JoltField> PolynomialBinding<F> for IdentityPolynomial<F> {
    fn is_bound(&self) -> bool {
        self.num_bound_vars != 0
    }

    fn bind(&mut self, r: F::Challenge, order: BindingOrder) {
        debug_assert!(self.num_bound_vars < self.num_vars);

        match order {
            BindingOrder::LowToHigh => {
                self.bound_value += F::from_u128(1 << self.num_bound_vars) * r;
            }
            BindingOrder::HighToLow => {
                self.bound_value += self.bound_value;
                self.bound_value = self.bound_value + r;
            }
        }
        self.num_bound_vars += 1;
    }

    fn bind_parallel(&mut self, r: F::Challenge, order: BindingOrder) {
        // Binding is constant time, no parallelism necessary
        self.bind(r, order);
    }

    fn final_sumcheck_claim(&self) -> F {
        debug_assert_eq!(self.num_vars, self.num_bound_vars);
        self.bound_value
    }
}

impl<F: JoltField> PolynomialEvaluation<F> for IdentityPolynomial<F> {
    fn evaluate<C>(&self, r: &[C]) -> F
    where
        C: Copy + Send + Sync + Into<F>,
        F: std::ops::Mul<C, Output = F> + std::ops::SubAssign<F>,
    {
        let len = r.len();
        debug_assert_eq!(len, self.num_vars);
        (0..len)
            .map(|i| r[i].into().mul_u128(1 << (len - 1 - i)))
            .sum()
    }

    fn batch_evaluate<C>(_polys: &[&Self], _r: &[C]) -> Vec<F>
    where
        C: Copy + Send + Sync + Into<F> + ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        unimplemented!("Currently unused")
    }

    fn sumcheck_evals(&self, index: usize, degree: usize, order: BindingOrder) -> Vec<F> {
        let mut evals = vec![F::zero(); degree];
        let m = match order {
            BindingOrder::LowToHigh => {
                let m = F::from_u128(1 << self.num_bound_vars);
                evals[0] = self.bound_value + (m + m).mul_u64(index as u64);
                m
            }
            BindingOrder::HighToLow => {
                let m = F::from_u128(1 << (self.num_vars - 1 - self.num_bound_vars));
                evals[0] = self.bound_value * (m + m) + F::from_u64(index as u64);
                m
            }
        };

        let mut eval = evals[0] + m;
        for i in 1..degree {
            eval += m;
            evals[i] = eval;
        }
        evals
    }
}

impl<F: JoltField> PrefixSuffixPolynomial<F, 2> for IdentityPolynomial<F> {
    fn suffixes(&self) -> [Box<dyn SuffixPolynomial<F> + Sync>; 2] {
        [Box::new(ShiftSuffixPolynomial), Box::new(self.clone())]
    }

    fn prefixes(
        &self,
        chunk_len: usize,
        phase: usize,
        registry: &mut PrefixRegistry<F>,
    ) -> [Option<Arc<RwLock<CachedPolynomial<F>>>>; 2] {
        if registry[Prefix::Identity].is_none() {
            registry[Prefix::Identity] = Some(Arc::new(RwLock::new(self.prefix_polynomial(
                &registry.checkpoints,
                chunk_len,
                phase,
            ))));
        }
        [registry[Prefix::Identity].clone(), None]
    }
}

impl<F: JoltField> PrefixPolynomial<F> for IdentityPolynomial<F> {
    fn prefix_polynomial(
        &self,
        checkpoints: &PrefixCheckpoints<F>,
        chunk_len: usize,
        _phase: usize,
    ) -> CachedPolynomial<F> {
        debug_assert!(chunk_len.is_even());
        let bound_value = checkpoints[Prefix::Identity].unwrap_or(F::zero());

        let evals: Vec<F> = (0..chunk_len.pow2())
            .map(|i| bound_value.mul_u128(1 << chunk_len) + F::from_u64(i as u64))
            .collect();

        CachedPolynomial::new(MultilinearPolynomial::from(evals), (chunk_len - 1).pow2())
    }
}

impl<F: JoltField> SuffixPolynomial<F> for IdentityPolynomial<F> {
    fn suffix_mle(&self, b: LookupBits) -> u128 {
        debug_assert!(b.len().is_even());
        u128::from(b)
    }
}

pub struct ShiftSuffixPolynomial;
impl<F: JoltField> SuffixPolynomial<F> for ShiftSuffixPolynomial {
    fn suffix_mle(&self, b: LookupBits) -> u128 {
        debug_assert!(b.len().is_even());
        1u128 << b.len()
    }
}

pub struct ShiftHalfSuffixPolynomial;
impl<F: JoltField> SuffixPolynomial<F> for ShiftHalfSuffixPolynomial {
    fn suffix_mle(&self, b: LookupBits) -> u128 {
        debug_assert!(b.len().is_even());
        1u128 << (b.len() / 2)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperandSide {
    Left,
    Right,
}

/// OperandPolynomial evaluates to either the right or left operand value from uninterleaved bits
/// For Right: sum_{i=0}^{num_vars/2-1} r[2i] * 2^(num_vars/2 - 1 - i)
/// For Left: sum_{i=0}^{num_vars/2-1} r[2i + 1] * 2^(num_vars/2 - 1 - i)
#[derive(Clone)]
pub struct OperandPolynomial<F: JoltField> {
    num_vars: usize,
    num_bound_vars: usize,
    bound_value: F,
    side: OperandSide,
}

impl<F: JoltField> OperandPolynomial<F> {
    pub fn new(num_vars: usize, side: OperandSide) -> Self {
        // 1. 断言检查：变量数必须是偶数
        debug_assert!(num_vars.is_even(), "num_vars must be divisible by 2");

        // 2. 初始化结构体
        OperandPolynomial {
            num_vars,               // 多项式的变量总数 (对应地址空间的位数，即 LOG_K)
            num_bound_vars: 0,      // 当前已绑定的变量数 (Sumcheck 还没开始，所以是 0)
            bound_value: F::zero(), // 当前的评估累加值 (初始为 0)
            side,                   // 指示它是左操作数 (RS1) 还是右操作数 (RS2)
        }
    }
}

impl<F: JoltField> PolynomialBinding<F> for OperandPolynomial<F> {
    fn is_bound(&self) -> bool {
        self.num_bound_vars != 0
    }

    fn bind(&mut self, r: F::Challenge, order: BindingOrder) {
        debug_assert!(self.num_bound_vars < self.num_vars);
        debug_assert_eq!(
            order,
            BindingOrder::HighToLow,
            "OperandPolynomial only supports high-to-low binding"
        );

        if (self.num_bound_vars.is_even() && self.side == OperandSide::Left)
            || (self.num_bound_vars.is_odd() && self.side == OperandSide::Right)
        {
            self.bound_value += self.bound_value;
            self.bound_value += r.into();
        }
        self.num_bound_vars += 1;
    }

    fn bind_parallel(&mut self, r: F::Challenge, order: BindingOrder) {
        // Binding is constant time, no parallelism necessary
        self.bind(r, order);
    }

    fn final_sumcheck_claim(&self) -> F {
        debug_assert_eq!(self.num_vars, self.num_bound_vars);
        self.bound_value
    }
}

impl<F: JoltField> PolynomialEvaluation<F> for OperandPolynomial<F> {
    fn evaluate<C>(&self, r: &[C]) -> F
    where
        C: Copy + Send + Sync + Into<F>,
        F: std::ops::Mul<C, Output = F> + std::ops::SubAssign<F>,
    {
        let len = r.len();
        debug_assert_eq!(len, self.num_vars);
        debug_assert!(len.is_even());

        match self.side {
            OperandSide::Left => (0..len / 2)
                .map(|i| r[2 * i].into().mul_u128(1 << (self.num_vars / 2 - 1 - i)))
                .sum(),
            OperandSide::Right => (0..len / 2)
                .map(|i| {
                    r[2 * i + 1]
                        .into()
                        .mul_u128(1 << (self.num_vars / 2 - 1 - i))
                })
                .sum(),
        }
    }

    fn batch_evaluate<C>(_polys: &[&Self], _r: &[C]) -> Vec<F>
    where
        C: Copy + Send + Sync + Into<F> + ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        unimplemented!("Currently unused")
    }

    fn sumcheck_evals(&self, index: usize, degree: usize, order: BindingOrder) -> Vec<F> {
        debug_assert_eq!(
            order,
            BindingOrder::HighToLow,
            "OperandPolynomial only supports high-to-low binding"
        );

        let mut evals = vec![F::zero(); degree];
        let (left, right) = uninterleave_bits(index as u128);

        let index = match self.side {
            OperandSide::Left => F::from_u64(left),
            OperandSide::Right => F::from_u64(right),
        };

        if self.num_bound_vars.is_even() {
            let unbound_pairs = (self.num_vars - self.num_bound_vars) / 2;
            debug_assert!(unbound_pairs > 0);
            evals[0] = self.bound_value.mul_u128(1 << unbound_pairs) + index;
            if self.side == OperandSide::Left {
                let m = F::from_u128(1 << (unbound_pairs - 1));
                let mut eval = evals[0] + m;
                for i in 1..degree {
                    eval += m;
                    evals[i] = eval;
                }
            } else {
                for i in 1..degree {
                    evals[i] = evals[0];
                }
            }
        } else if self.side == OperandSide::Right {
            // We are currently bindinng the right operand variable
            let unbound_pairs = (self.num_vars - self.num_bound_vars).div_ceil(2);
            evals[0] = self.bound_value.mul_u128(1 << unbound_pairs) + index;
            let m = F::from_u128(1 << (unbound_pairs - 1));
            let mut eval = evals[0] + m;
            for i in 1..degree {
                eval += m;
                evals[i] = eval;
            }
        } else {
            // We are currently binding right operand variable, but polynoimal is LeftOperand
            let unbound_pairs = (self.num_vars - self.num_bound_vars) / 2;
            evals[0] = self.bound_value.mul_u128(1 << unbound_pairs) + index;
            for i in 1..degree {
                evals[i] = evals[0];
            }
        }

        evals
    }
}

impl<F: JoltField> PrefixSuffixPolynomial<F, 2> for OperandPolynomial<F> {
    fn suffixes(&self) -> [Box<dyn SuffixPolynomial<F> + Sync>; 2] {
        [Box::new(ShiftHalfSuffixPolynomial), Box::new(self.clone())]
    }

    fn prefixes(
        &self,
        chunk_len: usize,
        phase: usize,
        prefix_registry: &mut PrefixRegistry<F>,
    ) -> [Option<Arc<RwLock<CachedPolynomial<F>>>>; 2] {
        match self.side {
            OperandSide::Left => {
                if prefix_registry[Prefix::LeftOperand].is_none() {
                    let lo_poly = OperandPolynomial::new(self.num_vars, OperandSide::Left);
                    prefix_registry[Prefix::LeftOperand] = Some(Arc::new(RwLock::new(
                        lo_poly.prefix_polynomial(&prefix_registry.checkpoints, chunk_len, phase),
                    )));
                }
                [prefix_registry[Prefix::LeftOperand].clone(), None]
            }
            OperandSide::Right => {
                if prefix_registry[Prefix::RightOperand].is_none() {
                    let ro_poly = OperandPolynomial::new(self.num_vars, OperandSide::Right);
                    prefix_registry[Prefix::RightOperand] = Some(Arc::new(RwLock::new(
                        ro_poly.prefix_polynomial(&prefix_registry.checkpoints, chunk_len, phase),
                    )));
                }
                [prefix_registry[Prefix::RightOperand].clone(), None]
            }
        }
    }
}

impl<F: JoltField> SuffixPolynomial<F> for OperandPolynomial<F> {
    fn suffix_mle(&self, b: LookupBits) -> u128 {
        debug_assert!(b.len().is_even());
        debug_assert!(self.num_bound_vars.is_even());
        let (left, right) = b.uninterleave();
        match self.side {
            OperandSide::Left => u128::from(left),
            OperandSide::Right => u128::from(right),
        }
    }
}

impl<F: JoltField> PrefixPolynomial<F> for OperandPolynomial<F> {
    fn prefix_polynomial(
        &self,
        checkpoints: &PrefixCheckpoints<F>,
        chunk_len: usize,
        _phase: usize,
    ) -> CachedPolynomial<F> {
        debug_assert!(chunk_len.is_even());
        debug_assert!(self.num_bound_vars.is_even());

        let bound_value = match self.side {
            OperandSide::Left => checkpoints[Prefix::LeftOperand].unwrap_or(F::zero()),
            OperandSide::Right => checkpoints[Prefix::RightOperand].unwrap_or(F::zero()),
        };

        let evals: Vec<F> = (0..chunk_len.pow2())
            .map(|i| {
                let bits = LookupBits::new(i as u128, chunk_len);
                let (left, right) = bits.uninterleave();
                let operand_value = match self.side {
                    OperandSide::Left => F::from_u64(u64::from(left)),
                    OperandSide::Right => F::from_u64(u64::from(right)),
                };
                bound_value.mul_u128(1 << (chunk_len / 2)) + operand_value
            })
            .collect();

        CachedPolynomial::new(MultilinearPolynomial::from(evals), (chunk_len - 1).pow2())
    }
}

/// Polynomial that unmaps RAM addresses: k -> k*8 + start_address
#[derive(Allocative)]
pub struct UnmapRamAddressPolynomial<F: JoltField> {
    pub start_address: u64,
    int_poly: IdentityPolynomial<F>,
}

impl<F: JoltField> UnmapRamAddressPolynomial<F> {
    pub fn new(num_vars: usize, start_address: u64) -> Self {
        assert!(start_address > 8);
        UnmapRamAddressPolynomial {
            start_address,
            int_poly: IdentityPolynomial::new(num_vars),
        }
    }
}

impl<F: JoltField> PolynomialBinding<F> for UnmapRamAddressPolynomial<F> {
    fn is_bound(&self) -> bool {
        self.int_poly.is_bound()
    }

    fn bind(&mut self, r: F::Challenge, order: BindingOrder) {
        self.int_poly.bind(r, order);
    }

    fn bind_parallel(&mut self, r: F::Challenge, order: BindingOrder) {
        // Binding is constant time, no parallelism necessary
        self.bind(r, order);
    }

    fn final_sumcheck_claim(&self) -> F {
        self.int_poly.final_sumcheck_claim().mul_u64(8) + F::from_u64(self.start_address)
    }
}

impl<F: JoltField> PolynomialEvaluation<F> for UnmapRamAddressPolynomial<F> {
    fn evaluate<C>(&self, r: &[C]) -> F
    where
        C: Copy + Send + Sync + Into<F> + ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        self.int_poly.evaluate(r).mul_u64(8) + F::from_u64(self.start_address)
    }

    fn batch_evaluate<C>(_polys: &[&Self], _r: &[C]) -> Vec<F>
    where
        C: Copy + Send + Sync + Into<F> + ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        unimplemented!("Unused")
    }

    fn sumcheck_evals(&self, index: usize, degree: usize, order: BindingOrder) -> Vec<F> {
        let evals = self.int_poly.sumcheck_evals(index, degree, order);
        evals
            .into_iter()
            .map(|l| l.mul_u64(8) + F::from_u64(self.start_address))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::poly::multilinear_polynomial::MultilinearPolynomial;
    use crate::poly::prefix_suffix::tests::prefix_suffix_decomposition_test;

    use super::*;
    use ark_bn254::Fr;
    use ark_ec::AdditiveGroup;
    use ark_ff::Field;
    use ark_std::test_rng;

    #[test]
    fn identity_poly() {
        const NUM_VARS: usize = 10;
        const DEGREE: usize = 3;

        let mut rng = test_rng();
        let mut identity_poly: IdentityPolynomial<Fr> = IdentityPolynomial::new(NUM_VARS);
        let mut reference_poly: MultilinearPolynomial<Fr> =
            MultilinearPolynomial::from((0..(1 << NUM_VARS)).map(|i| i as u32).collect::<Vec<_>>());

        for j in 0..reference_poly.len() / 2 {
            let identity_poly_evals =
                identity_poly.sumcheck_evals(j, DEGREE, BindingOrder::LowToHigh);
            let reference_poly_evals =
                reference_poly.sumcheck_evals(j, DEGREE, BindingOrder::LowToHigh);
            assert_eq!(identity_poly_evals, reference_poly_evals);
        }

        for _ in 0..NUM_VARS {
            let r = <Fr as JoltField>::Challenge::random(&mut rng);
            identity_poly.bind(r, BindingOrder::LowToHigh);
            reference_poly.bind(r, BindingOrder::LowToHigh);
            for j in 0..reference_poly.len() / 2 {
                let identity_poly_evals =
                    identity_poly.sumcheck_evals(j, DEGREE, BindingOrder::LowToHigh);
                let reference_poly_evals =
                    reference_poly.sumcheck_evals(j, DEGREE, BindingOrder::LowToHigh);
                assert_eq!(identity_poly_evals, reference_poly_evals);
            }
        }

        assert_eq!(
            identity_poly.final_sumcheck_claim(),
            reference_poly.final_sumcheck_claim()
        );
    }

    #[test]
    fn identity_poly_prefix_suffix_decomposition() {
        prefix_suffix_decomposition_test::<8, 2, 2, _>(
            IdentityPolynomial::new(8),
            Prefix::Identity,
        );
    }

    #[test]
    fn operand_poly_prefix_suffix_decomposition() {
        prefix_suffix_decomposition_test::<8, 2, 2, _>(
            OperandPolynomial::new(8, OperandSide::Left),
            Prefix::LeftOperand,
        );
        prefix_suffix_decomposition_test::<8, 2, 2, _>(
            OperandPolynomial::new(8, OperandSide::Right),
            Prefix::RightOperand,
        );
    }

    #[test]
    fn unmap_address_poly_evaluate() {
        const NUM_VARS: usize = 10;
        const START_ADDRESS: u64 = 0x80000000;

        // Test that UnmapAddressPolynomial evaluates correctly on boolean hypercube
        let unmap_poly = UnmapRamAddressPolynomial::<Fr>::new(NUM_VARS, START_ADDRESS);

        // Test a few specific points
        // k=0 should map to start_address
        let point_0 = vec![Fr::ZERO; NUM_VARS];
        assert_eq!(unmap_poly.evaluate(&point_0), Fr::from(START_ADDRESS));

        // k=1 should map to start_address + 8
        let mut point_1 = vec![Fr::ZERO; NUM_VARS];
        point_1[NUM_VARS - 1] = Fr::ONE;
        assert_eq!(unmap_poly.evaluate(&point_1), Fr::from(START_ADDRESS + 8));

        // k=2 should map to start_address + 16
        let mut point_2 = vec![Fr::ZERO; NUM_VARS];
        point_2[NUM_VARS - 2] = Fr::ONE;
        assert_eq!(unmap_poly.evaluate(&point_2), Fr::from(START_ADDRESS + 16));

        // k=3 should map to start_address + 24
        let mut point_3 = vec![Fr::ZERO; NUM_VARS];
        point_3[NUM_VARS - 1] = Fr::ONE;
        point_3[NUM_VARS - 2] = Fr::ONE;
        assert_eq!(unmap_poly.evaluate(&point_3), Fr::from(START_ADDRESS + 24));
    }

    /// 测试 OperandPolynomial 在布尔超立方体上的正确性
    ///
    /// # 测试目标
    ///
    /// 验证 `OperandPolynomial` (左、右操作数多项式) 能够在布尔超立方体
    /// (Boolean Hypercube) 的所有顶点上正确计算出对应的操作数值。
    ///
    /// # 测试逻辑
    ///
    /// 1. **布尔超立方体**: 对于 `NUM_VARS = 8`,布尔超立方体有 2^8 = 256 个顶点,
    ///    每个顶点对应一个 8 位的二进制索引 (i)。
    /// 2. **索引拆分**: 使用 `uninterleave_bits(i)` 将索引拆分为左、右操作数:
    ///    - 如索引 `0b10110011`,拆分为 `left = 0b1101`, `right = 0b0011`
    /// 3. **验证**: 在每个顶点 eval_point 上,验证多项式的评估值是否等于预期的操作数值。
    #[test]
    fn operand_poly_boolean_hypercube() {
        const NUM_VARS: usize = 8;

        // 创建右操作数和左操作数多项式
        let ro_poly = OperandPolynomial::<Fr>::new(NUM_VARS, OperandSide::Right);
        let lo_poly = OperandPolynomial::<Fr>::new(NUM_VARS, OperandSide::Left);

        // 遍历布尔超立方体的所有 2^8 = 256 个顶点
        for i in 0..(1 << NUM_VARS) {
            // 1. 将索引 i 转换为布尔超立方体的评估点 (eval_point)
            //    eval_point[j] = (i 的第 j 位是否为 1)
            let mut eval_point = vec![Fr::ZERO; NUM_VARS];
            for j in 0..NUM_VARS {
                if (i >> j) & 1 == 1 {
                    eval_point[j] = Fr::ONE;
                }
            }
            // 注意: 反转是因为 Jolt 的比特顺序约定
            eval_point.reverse();

            // 2. 使用 uninterleave_bits 将索引 i 拆分为左右操作数
            //    例如: i = 0b10110011 -> left = 0b1101, right = 0b0011
            let (left, right) = uninterleave_bits(i as u128);
            let expected_r = Fr::from_u64(right);
            let expected_l = Fr::from_u64(left);

            // 3. 验证多项式在该顶点的评估值是否等于预期的操作数值
            assert_eq!(
                ro_poly.evaluate(&eval_point),
                expected_r,
                "Boolean hypercube RIGHT OPERAND evaluation failed at index {i}"
            );
            assert_eq!(
                lo_poly.evaluate(&eval_point),
                expected_l,
                "Boolean hypercube LEFT OPERAND evaluation failed at index {i}"
            );
        }
    }

    /// 测试 OperandPolynomial 在 Sumcheck 协议中的正确性
    ///
    /// # 测试目标
    ///
    /// 验证 `OperandPolynomial` 在 Sumcheck 协议的各个轮次 (Rounds) 中的行为是否
    /// 与参考多项式 (Reference Polynomial, 即完全展开的多线性多项式) 一致。
    ///
    /// # 测试流程
    ///
    /// 1. **初始化**: 创建操作数多项式和参考多项式 (通过完全展开的评估向量)。
    /// 2. **布尔超立方体验证**: 在所有 256 个顶点上验证两者评估值一致。
    /// 3. **Sumcheck 模拟**: 模拟 Sumcheck 的 8 轮绑定 (Binding) 过程:
    ///    - 每轮开始前,验证 `sumcheck_evals` (单变量多项式的评估值) 与参考多项式一致。
    ///    - 使用随机数 `r` 对当前变量进行绑定,将多项式维度降低 1。
    /// 4. **最终声明**: 验证 Sumcheck 结束后的最终声明值 (Final Claim) 一致。
    #[test]
    fn operand_poly() {
        const NUM_VARS: usize = 8;

        let mut rng = test_rng();
        let mut ro_poly = OperandPolynomial::<Fr>::new(NUM_VARS, OperandSide::Right);
        let mut lo_poly = OperandPolynomial::<Fr>::new(NUM_VARS, OperandSide::Left);

        // 1. 创建参考多项式 (Reference Polynomial)
        //    通过显式计算所有 2^8 = 256 个评估点的值来构造完整的多线性多项式
        let (reference_evals_l, reference_evals_r): (Vec<Fr>, Vec<Fr>) = (0u128..(1 << NUM_VARS))
            .map(|i| {
                let (left, right) = uninterleave_bits(i);
                (Fr::from_u64(left), Fr::from_u64(right))
            })
            .collect();
        let mut reference_poly_r = MultilinearPolynomial::from(reference_evals_r);
        let mut reference_poly_l = MultilinearPolynomial::from(reference_evals_l);

        // 2. 验证两种多项式在整个布尔超立方体上的评估值一致
        for i in 0..(1 << NUM_VARS) {
            let mut eval_point = vec![Fr::ZERO; NUM_VARS];
            for j in 0..NUM_VARS {
                if (i >> j) & 1 == 1 {
                    eval_point[j] = Fr::ONE;
                }
            }
            let ro_poly = ro_poly.evaluate(&eval_point);
            let lo_poly = lo_poly.evaluate(&eval_point);
            let reference_r = reference_poly_r.evaluate(&eval_point);
            let reference_l = reference_poly_l.evaluate(&eval_point);
            assert_eq!(
                (ro_poly, lo_poly), (reference_r, reference_l),
                "Evaluation mismatch at index {i}:, operand_poly={ro_poly}, {lo_poly}, reference={reference_r}, {reference_l}"
            );
        }

        // 3. 模拟 Sumcheck 协议的 NUM_VARS 轮绑定过程
        for round in 0..NUM_VARS {
            // 当前轮的评估点数量 = 2^(剩余变量数)
            let num_evals = 1 << (NUM_VARS - round - 1);

            // 对每个评估点,验证单变量多项式的评估值 (degree=3 表示评估 g(0), g(1), g(2))
            for i in 0..num_evals {
                let ro_poly = ro_poly.sumcheck_evals(i, 3, BindingOrder::HighToLow);
                let lo_poly = lo_poly.sumcheck_evals(i, 3, BindingOrder::HighToLow);
                let reference_r = reference_poly_r.sumcheck_evals(i, 3, BindingOrder::HighToLow);
                let reference_l = reference_poly_l.sumcheck_evals(i, 3, BindingOrder::HighToLow);

                assert_eq!(
                    (ro_poly, lo_poly),
                    (reference_r, reference_l),
                    "Round {round}, index {i}: sumcheck_evals mismatch"
                );
            }

            // 使用 Verifier 发送的随机数 r 对当前变量进行绑定
            // 绑定后,多项式的变量数减少 1
            let r = <Fr as JoltField>::Challenge::random(&mut rng);
            ro_poly.bind(r, BindingOrder::HighToLow);
            lo_poly.bind(r, BindingOrder::HighToLow);
            reference_poly_r.bind(r, BindingOrder::HighToLow);
            reference_poly_l.bind(r, BindingOrder::HighToLow);
        }

        // 4. 验证 Sumcheck 结束后的最终声明值 (Final Claim)
        //    此时所有变量都已绑定,多项式变为一个标量
        assert_eq!(
            ro_poly.final_sumcheck_claim(),
            reference_poly_r.final_sumcheck_claim(),
            "Final sumcheck claims don't match"
        );
        assert_eq!(
            lo_poly.final_sumcheck_claim(),
            reference_poly_l.final_sumcheck_claim(),
            "Final sumcheck claims don't match"
        );
    }

    /// 测试 UnmapRamAddressPolynomial 的正确性
    ///
    /// # 功能说明
    ///
    /// `UnmapRamAddressPolynomial` 用于将规范化的 RAM 索引 `k` 映射回物理内存地址:
    /// `physical_address = k * 8 + START_ADDRESS`
    ///
    /// 这在 RAM 访问证明中很关键,因为我们需要证明:
    /// "我访问的地址 `addr` 是通过合法的索引 `k` 计算得到的"。
    ///
    /// # 测试流程
    ///
    /// 1. **初始化**: 创建地址反映射多项式和参考多项式。
    /// 2. **Sumcheck 模拟**: 模拟 Sumcheck 的 4 轮绑定过程。
    /// 3. **最终声明**: 验证最终声明值一致。
    ///
    /// # 举例
    ///
    /// 假设 `START_ADDRESS = 0x80000000`, `NUM_VARS = 4`:
    /// - `k = 0` -> `address = 0 * 8 + 0x80000000 = 0x80000000`
    /// - `k = 1` -> `address = 1 * 8 + 0x80000000 = 0x80000008`
    /// - `k = 2` -> `address = 2 * 8 + 0x80000000 = 0x80000010`
    #[test]
    fn unmap_address_poly() {
        const NUM_VARS: usize = 4;
        const START_ADDRESS: u64 = 0x80000000; // RAM 起始地址 (2GB)

        let mut unmap_poly = UnmapRamAddressPolynomial::<Fr>::new(NUM_VARS, START_ADDRESS);

        // 1. 创建参考多项式
        //    unmap_evals[k] = k * 8 + START_ADDRESS
        let K = 1 << NUM_VARS;
        let unmap_evals: Vec<Fr> = (0..K)
            .map(|k| Fr::from((k as u64).wrapping_mul(8).wrapping_add(START_ADDRESS)))
            .collect();
        let mut reference_poly = MultilinearPolynomial::from(unmap_evals.clone());

        // 2. 模拟 Sumcheck 的 NUM_VARS 轮绑定过程
        for round in 0..NUM_VARS {
            let num_evals = 1 << (NUM_VARS - round - 1);

            // 验证每个索引 i 处的单变量多项式评估值
            for i in 0..num_evals {
                let unmap_evals = unmap_poly.sumcheck_evals(i, 3, BindingOrder::LowToHigh);
                let reference_evals = reference_poly.sumcheck_evals(i, 3, BindingOrder::LowToHigh);

                assert_eq!(
                    unmap_evals, reference_evals,
                    "Round {round}, index {i}: sumcheck_evals mismatch",
                );
            }

            // 使用确定性随机数进行绑定 (为了测试的可重复性)
            let r = <Fr as JoltField>::Challenge::from(0x12345678 + round as u128);
            unmap_poly.bind(r, BindingOrder::LowToHigh);
            reference_poly.bind(r, BindingOrder::LowToHigh);
        }

        // 3. 验证最终声明值
        assert_eq!(
            unmap_poly.final_sumcheck_claim(),
            reference_poly.final_sumcheck_claim(),
            "Final sumcheck claims don't match"
        );
    }
}
