use crate::field::JoltField;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding};
use crate::poly::opening_proof::{
    OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
    VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
};
use crate::poly::split_eq_poly::GruenSplitEqPolynomial;
use crate::poly::unipoly::UniPoly;
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use crate::subprotocols::sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier};
use crate::transcripts::Transcript;
use crate::zkvm::witness::VirtualPolynomial;
use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use rayon::prelude::*;
use tracer::instruction::Cycle;

// RAM Hamming booleanity sumcheck
//
// Proves a zero-check of the form
//   0 = Σ_j eq(r_cycle, j) · (H(j)^2 − H(j))
// where:
// - r_cycle are the time/cycle variables bound in this sumcheck
// - H(j) is an indicator of whether a RAM access occurred at cycle j (1 if address != 0, 0 otherwise)

/// Degree bound of the sumcheck round polynomials in [`HammingBooleanitySumcheckVerifier`].
const DEGREE_BOUND: usize = 3;

#[derive(Allocative, Clone)]
pub struct HammingBooleanitySumcheckParams<F: JoltField> {
    pub r_cycle: OpeningPoint<BIG_ENDIAN, F>,
}

impl<F: JoltField> HammingBooleanitySumcheckParams<F> {
    pub fn new(opening_accumulator: &dyn OpeningAccumulator<F>) -> Self {
        let (r_cycle, _) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::LookupOutput,
            SumcheckId::SpartanOuter,
        );

        Self { r_cycle }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for HammingBooleanitySumcheckParams<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.r_cycle.len()
    }

    fn input_claim(&self, _: &dyn OpeningAccumulator<F>) -> F {
        F::zero()
    }

    fn normalize_opening_point(
        &self,
        challenges: &[<F as JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }
}

#[derive(Allocative)]
pub struct HammingBooleanitySumcheckProver<F: JoltField> {
    eq_r_cycle: GruenSplitEqPolynomial<F>,
    H: MultilinearPolynomial<F>,
    pub params: HammingBooleanitySumcheckParams<F>,
}

impl<F: JoltField> HammingBooleanitySumcheckProver<F> {
       #[tracing::instrument(skip_all, name = "RamHammingBooleanitySumcheckProver::initialize")]
       /// 该 Sumcheck 的主要目的是强制约束 RAM 访问指示信号必须是布尔值（0 或 1）。
       /// 如果没有这个约束，恶意 Prover 可能会在计算“发生了多少次 RAM 访问”时使用非 0/1 的值（例如 2 或 -1）进行欺骗，从而破坏内存一致性检查。
       pub fn initialize(params: HammingBooleanitySumcheckParams<F>, trace: &[Cycle]) -> Self {
           // =========================================================
           // 1. 构建 RAM 访问指示向量 H (Indicator Vector)
           // ---------------------------------------------------------
           // 这里的 H(t) 代表：在第 t 个 CPU 周期，是否发生了有效的 RAM 访问。
           // - 如果地址 != 0，视为有效访问 (1/True)。
           // - 如果地址 == 0 (通常表示空操作或无访问)，视为无访问 (0/False)。
           //
           // 解决的核心问题 (Booleanity Check)：
           // 我们必须向 Verifier 证明这个 H 向量中的每一个元素都严格属于 {0, 1}。
           // 只有这样，后续累加 H 来计算“总访问次数”或“汉明重量”才是合法的。
           // 证明方法是验证恒等式：H^2 - H = 0 (即 x(x-1)=0 => x=0 or 1)。
           // =========================================================
           let H = trace
               .par_iter()
               .map(|cycle| cycle.ram_access().address() != 0)
               .collect::<Vec<bool>>();

           // 将布尔向量转换为多线性多项式 (MLE)，作为 Sumcheck 的 Witness
           let H = MultilinearPolynomial::from(H);

           // 初始化 Eq(r_cycle, t) 多项式
           // 用于在 Sumcheck 协议中对所有时间步 t 进行加权求和
           let eq_r_cycle = GruenSplitEqPolynomial::new(&params.r_cycle.r, BindingOrder::LowToHigh);

           Self {
               eq_r_cycle,
               H,
               params,
           }
       }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T>
    for HammingBooleanitySumcheckProver<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(skip_all, name = "RamHammingBooleanitySumcheckProver::compute_message")]
    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let eq = &self.eq_r_cycle;
        let H = &self.H;

        // Accumulate constant (c0) and quadratic (e) coefficients via generic split-eq fold.
        let [c0, e] = eq.par_fold_out_in_unreduced::<9, 2>(&|g| {
            let h0 = H.get_bound_coeff(2 * g);
            let h1 = H.get_bound_coeff(2 * g + 1);
            let delta = h1 - h0;
            [h0.square() - h0, delta.square()]
        });
        eq.gruen_poly_deg_3(c0, e, previous_claim)
    }

    #[tracing::instrument(
        skip_all,
        name = "RamHammingBooleanitySumcheckProver::ingest_challenge"
    )]
    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.eq_r_cycle.bind(r_j);
        self.H.bind_parallel(r_j, BindingOrder::LowToHigh);
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RamHammingWeight,
            SumcheckId::RamHammingBooleanity,
            self.params.normalize_opening_point(sumcheck_challenges),
            self.H.final_sumcheck_claim(),
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

pub struct HammingBooleanitySumcheckVerifier<F: JoltField> {
    params: HammingBooleanitySumcheckParams<F>,
}

impl<F: JoltField> HammingBooleanitySumcheckVerifier<F> {
    pub fn new(opening_accumulator: &dyn OpeningAccumulator<F>) -> Self {
        Self {
            params: HammingBooleanitySumcheckParams::new(opening_accumulator),
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
    for HammingBooleanitySumcheckVerifier<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let H_claim = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RamHammingWeight,
                SumcheckId::RamHammingBooleanity,
            )
            .1;

        let (r_cycle, _) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::LookupOutput,
            SumcheckId::SpartanOuter,
        );

        let eq = EqPolynomial::<F>::mle(
            sumcheck_challenges,
            &r_cycle
                .r
                .iter()
                .cloned()
                .rev()
                .collect::<Vec<F::Challenge>>(),
        );

        (H_claim.square() - H_claim) * eq
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RamHammingWeight,
            SumcheckId::RamHammingBooleanity,
            self.params.normalize_opening_point(sumcheck_challenges),
        );
    }
}
