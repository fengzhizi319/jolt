use std::{array, iter::once, sync::Arc};

use num_traits::Zero;

use crate::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        identity_poly::IdentityPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN,
        },
        ra_poly::RaPolynomial,
        split_eq_poly::GruenSplitEqPolynomial,
        unipoly::UniPoly,
    },
    subprotocols::{
        mles_product_sum::eval_linear_prod_assign,
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
    utils::{math::Math, small_scalar::SmallScalar, thread::unsafe_allocate_zero_vec},
    zkvm::{
        bytecode::BytecodePreprocessing,
        config::OneHotParams,
        instruction::{
            CircuitFlags, Flags, InstructionFlags, InstructionLookup, InterleavedBitsMarker,
            NUM_CIRCUIT_FLAGS,
        },
        lookup_table::{LookupTables, NUM_LOOKUP_TABLES},
        witness::{CommittedPolynomial, VirtualPolynomial},
    },
};
use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use common::constants::{REGISTER_COUNT, XLEN};
use itertools::{zip_eq, Itertools};
use rayon::prelude::*;
use strum::{EnumCount, IntoEnumIterator};
use tracing::info;
use tracer::instruction::{Cycle, Instruction};

/// Number of batched read-checking sumchecks bespokely
const N_STAGES: usize = 5;

/// Bytecode instruction: multi-stage Read + RAF sumcheck (N_STAGES = 5).
///
/// Stages virtualize different claim families (Stage1: Spartan outer; Stage2: product-virtualized
/// flags; Stage3: Shift; Stage4: Registers RW; Stage5: Registers val-eval + Instruction lookups).
/// The input claim is a γ-weighted RLC of stage rv_claims plus RAF contributions folded into
/// stages 1 and 3 via the identity polynomial. Address vars are bound in `d` chunks; cycle vars
/// are bound with per-stage `GruenSplitEqPolynomial` (low-to-high binding), producing degree-3
/// univariates.
///
/// Mathematical claim:
/// - Let K = 2^{log_K} and T = 2^{log_T}.
/// - For stage s ∈ {1,2,3,4,5}, let r_s ∈ F^{log_T} and define eq_s(j) = EqPolynomial(j; r_s).
/// - Let r_addr ∈ F^{log_K}. Let ra(k, j) ∈ {0,1} be the indicator that cycle j has program
///   counter/address k (implemented as ∏_{i=0}^{d-1} ra_i(k_i, j)).
/// - Int(k) = 1 for all k (evaluation of the IdentityPolynomial over address variables).
/// - Define per-stage Val_s(k) (address-only) as implemented by `compute_val_*`:
///   * Stage1: Val_1(k) = unexpanded_pc(k) + γ·imm(k) + Σ_t γ^{2+t}·circuit_flag_t(k).
///   * Stage2: Val_2(k) = 1_{jump}(k) + γ·1_{branch}(k) + γ^2·rd_addr(k) + γ^3·1_{write_lookup_to_rd}(k).
///   * Stage3: Val_3(k) = imm(k) + γ·unexpanded_pc(k) + γ^2·1_{L_is_rs1}(k) + γ^3·1_{L_is_pc}(k)
///   + γ^4·1_{R_is_rs2}(k) + γ^5·1_{R_is_imm}(k) + γ^6·1_{IsNoop}(k)
///   + γ^7·1_{VirtualInstruction}(k) + γ^8·1_{IsFirstInSequence}(k).
///   * Stage4: Val_4(k) = 1_{rd=r}(k) + γ·1_{rs1=r}(k) + γ^2·1_{rs2=r}(k), where r is fixed by opening.
///   * Stage5: Val_5(k) = 1_{rd=r}(k) + γ·1_{¬interleaved}(k) + Σ_i γ^{2+i}·1_{table=i}(k).
///
/// Accumulator-provided LHS (RLC of stage claims with RAF):
///   rv_1(r_1) + γ·rv_2(r_2) + γ^2·rv_3(r_3) + γ^3·rv_4(r_4) + γ^4·rv_5(r_5)
///   + γ^5·raf_1(r_1) + γ^6·raf_3(r_3).
///
/// Sumcheck RHS proved (double sum over cycles and addresses):
///   Σ_{j=0}^{T-1} Σ_{k=0}^{K-1} ra(k, j) · [
///       γ^0·eq_1(j)·Val_1(k) + γ^1·eq_2(j)·Val_2(k) + γ^2·eq_3(j)·Val_3(k)
///     + γ^3·eq_4(j)·Val_4(k) + γ^4·eq_5(j)·Val_5(k)
///     + γ^5·eq_1(j)·Int(k)   + γ^6·eq_3(j)·Int(k)
///   ].
///
/// Thus the identity established by this sumcheck is:
///   rv_1(r_1) + γ·rv_2(r_2) + γ^2·rv_3(r_3) + γ^3·rv_4(r_4) + γ^4·rv_5(r_5)
///   + γ^5·raf_1(r_1) + γ^6·raf_3(r_3)
///     = Σ_{j,k} ra(k, j) · [ Σ_{s=1}^{5} γ^{s-1}·eq_s(j)·Val_s(k) + γ^5·eq_1(j)·Int(k) + γ^6·eq_3(j)·Int(k) ].
///
/// Binding/implementation notes:
/// - Address variables are bound first (high→low) in `d` chunks, accumulating `F_i` and `v` tables;
///   this materializes the address-only Val_s(k) evaluations and sets up `ra_i` polynomials.
/// - Cycle variables are then bound (low→high) per stage with `GruenSplitEqPolynomial`, using
///   previous-round claims to recover the cubic univariate each round.
///   Prover state for the bytecode Read+RAF multi-stage sumcheck.
///
/// First log(K) rounds bind address variables in chunks, aggregating per-stage address-only
/// contributions; last log(T) rounds bind cycle variables via per-stage `GruenSplitEqPolynomial`s.
#[derive(Allocative)]
pub struct BytecodeReadRafSumcheckProver<F: JoltField> {
    /// Per-stage address MLEs F_i(k) built from eq(r_cycle_stage_i, (chunk_index, j)),
    /// bound high-to-low during the address-binding phase.
    F: [MultilinearPolynomial<F>; N_STAGES],
    /// Chunked RA polynomials over address variables (one per dimension `d`), used to form
    /// the product ∏_i ra_i during the cycle-binding phase.
    ra: Vec<RaPolynomial<u8, F>>,
    /// Binding challenges for the first log_K variables of the sumcheck
    r_address_prime: Vec<F::Challenge>,
    /// Per-stage Gruen-split eq polynomials over cycle vars (low-to-high binding order).
    gruen_eq_polys: [GruenSplitEqPolynomial<F>; N_STAGES],
    /// Previous-round claims s_i(0)+s_i(1) per stage, needed for degree-3 univariate recovery.
    prev_round_claims: [F; N_STAGES],
    /// Round polynomials per stage for advancing to the next claim at r_j.
    prev_round_polys: Option<[UniPoly<F>; N_STAGES]>,
    /// Final sumcheck claims of stage Val polynomials (with RAF Int folded where applicable).
    bound_val_evals: Option<[F; N_STAGES]>,
    /// Trace for computing PCs on the fly in init_log_t_rounds.
    #[allocative(skip)]
    trace: Arc<Vec<Cycle>>,
    /// Bytecode preprocessing for computing PCs.
    #[allocative(skip)]
    bytecode_preprocessing: Arc<BytecodePreprocessing>,
    pub params: BytecodeReadRafSumcheckParams<F>,
}

impl<F: JoltField> BytecodeReadRafSumcheckProver<F> {
    #[tracing::instrument(skip_all, name = "BytecodeReadRafSumcheckProver::initialize")]
    pub fn initialize(
        params: BytecodeReadRafSumcheckParams<F>,
        trace: Arc<Vec<Cycle>>,
        bytecode_preprocessing: Arc<BytecodePreprocessing>,
    ) -> Self {
        let claim_per_stage = [
            params.rv_claims[0] + params.gamma_powers[5] * params.raf_claim,
            params.rv_claims[1],
            params.rv_claims[2] + params.gamma_powers[4] * params.raf_shift_claim,
            params.rv_claims[3],
            params.rv_claims[4],
        ];

        // Two-table split-eq optimization for computing F[stage][k] = Σ_{c: PC(c)=k} eq(r_cycle, c).
        //
        // Double summation pattern:
        //   F[stage][k] = Σ_{c_hi} E_hi[c_hi] × ( Σ_{c_lo : PC(c)=k} E_lo[c_lo] )
        //
        // Inner sum (over c_lo): ADDITIONS ONLY - accumulate E_lo contributions by PC
        // Outer sum (over c_hi): ONE multiplication per touched PC, not per cycle
        //
        // This reduces multiplications from O(T × N_STAGES) to O(touched_PCs × out_len × N_STAGES)
        let T = trace.len();
        let K = params.K;
        let log_T = params.log_T;

        // Optimal split: sqrt(T) for balanced tables
        let lo_bits = log_T / 2;
        let hi_bits = log_T - lo_bits;
        let in_len: usize = 1 << lo_bits; // E_lo size (inner loop)
        let out_len: usize = 1 << hi_bits; // E_hi size (outer loop)

        // Pre-compute E_hi[stage][c_hi] and E_lo[stage][c_lo] for all stages in parallel
        let (E_hi, E_lo): ([Vec<F>; N_STAGES], [Vec<F>; N_STAGES]) = rayon::join(
            || {
                params
                    .r_cycles
                    .each_ref()
                    .map(|r_cycle| EqPolynomial::evals(&r_cycle[..hi_bits]))
            },
            || {
                params
                    .r_cycles
                    .each_ref()
                    .map(|r_cycle| EqPolynomial::evals(&r_cycle[hi_bits..]))
            },
        );

        // Process by c_hi blocks, distributing work evenly among threads
        let num_threads = rayon::current_num_threads();
        let chunk_size = out_len.div_ceil(num_threads);

        // Double summation: outer sum over c_hi, inner sum over c_lo
        let F: [Vec<F>; N_STAGES] = E_hi[0]
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(chunk_idx, chunk)| {
                // Per-thread accumulators for final F
                let mut partial: [Vec<F>; N_STAGES] =
                    array::from_fn(|_| unsafe_allocate_zero_vec(K));

                // Per-c_hi inner accumulators (reused across c_hi iterations)
                let mut inner: [Vec<F>; N_STAGES] = array::from_fn(|_| unsafe_allocate_zero_vec(K));

                // Track which PCs were touched in this c_hi block
                let mut touched = Vec::with_capacity(in_len);

                let chunk_start = chunk_idx * chunk_size;
                for (local_idx, _) in chunk.iter().enumerate() {
                    let c_hi = chunk_start + local_idx;
                    let c_hi_base = c_hi * in_len;

                    // Clear inner accumulators for touched PCs only
                    for &k in &touched {
                        for stage in 0..N_STAGES {
                            inner[stage][k] = F::zero();
                        }
                    }
                    touched.clear();

                    // INNER SUM: accumulate E_lo by PC (ADDITIONS ONLY, no multiplications)
                    for c_lo in 0..in_len {
                        let c = c_hi_base + c_lo;
                        if c >= T {
                            break;
                        }

                        let pc = bytecode_preprocessing.get_pc(&trace[c]);

                        // Track touched PCs (avoid duplicates with a simple check)
                        if inner[0][pc].is_zero() {
                            touched.push(pc);
                        }

                        // Accumulate E_lo contributions (addition only!)
                        for stage in 0..N_STAGES {
                            inner[stage][pc] += E_lo[stage][c_lo];
                        }
                    }

                    // OUTER SUM: multiply by E_hi and add to partial (sparse)
                    for &k in &touched {
                        for stage in 0..N_STAGES {
                            partial[stage][k] += E_hi[stage][c_hi] * inner[stage][k];
                        }
                    }
                }

                partial
            })
            .reduce(
                || array::from_fn(|_| unsafe_allocate_zero_vec(K)),
                |mut a, b| {
                    for stage in 0..N_STAGES {
                        a[stage]
                            .par_iter_mut()
                            .zip(b[stage].par_iter())
                            .for_each(|(a, b)| *a += *b);
                    }
                    a
                },
            );

        #[cfg(test)]
        {
            // Verify that for each stage i: sum(val_i[k] * F_i[k] * eq_i[k]) = rv_claim_i
            for i in 0..N_STAGES {
                let computed_claim: F = (0..params.K)
                    .into_par_iter()
                    .map(|k| {
                        let val_k = params.val_polys[i].get_bound_coeff(k);
                        let F_k = F[i][k];
                        val_k * F_k
                    })
                    .sum();
                assert_eq!(
                    computed_claim,
                    params.rv_claims[i],
                    "Stage {} mismatch: computed {} != expected {}",
                    i + 1,
                    computed_claim,
                    params.rv_claims[i]
                );
            }
        }

        let F = F.map(MultilinearPolynomial::from);

        let gruen_eq_polys = params
            .r_cycles
            .each_ref()
            .map(|r_cycle| GruenSplitEqPolynomial::new(r_cycle, BindingOrder::LowToHigh));

        Self {
            F,
            ra: Vec::with_capacity(params.d),
            r_address_prime: Vec::with_capacity(params.log_K),
            gruen_eq_polys,
            prev_round_claims: claim_per_stage,
            prev_round_polys: None,
            bound_val_evals: None,
            trace,
            bytecode_preprocessing,
            params,
        }
    }

    fn init_log_t_rounds(&mut self) {
        let int_poly = self.params.int_poly.final_sumcheck_claim();

        // We have a separate Val polynomial for each stage
        // Additionally, for stages 1 and 3 we have an Int polynomial for RAF
        // So we would have:
        // Stage 1: gamma^0 * (Val_1 + gamma^5 * Int)
        // Stage 2: gamma^1 * (Val_2)
        // Stage 3: gamma^2 * (Val_3 + gamma^4 * Int)
        // Stage 4: gamma^3 * (Val_4)
        // Stage 5: gamma^4 * (Val_5)
        // Which matches with the input claim:
        // rv_1 + gamma * rv_2 + gamma^2 * rv_3 + gamma^3 * rv_4 + gamma^4 * rv_5 + gamma^5 * raf_1 + gamma^6 * raf_3
        self.bound_val_evals = Some(
            self.params
                .val_polys
                .iter()
                .zip([
                    int_poly * self.params.gamma_powers[5],
                    F::zero(),
                    int_poly * self.params.gamma_powers[4],
                    F::zero(),
                    F::zero(),
                ])
                .map(|(poly, int_poly)| poly.final_sumcheck_claim() + int_poly)
                .collect::<Vec<F>>()
                .try_into()
                .unwrap(),
        );

        // Reverse r_address_prime to get the correct order (it was built low-to-high)
        let mut r_address = std::mem::take(&mut self.r_address_prime);
        r_address.reverse();

        // Drop log_K phase data that's no longer needed (val_polys reduced to bound_val_evals)
        // F polynomials are fully bound and can be dropped
        self.F = array::from_fn(|_| MultilinearPolynomial::default());
        // val_polys are reduced to scalars in bound_val_evals
        self.params.val_polys = array::from_fn(|_| MultilinearPolynomial::default());
        // int_poly is reduced to a scalar
        self.params.int_poly = IdentityPolynomial::new(0);

        let r_address_chunks = self
            .params
            .one_hot_params
            .compute_r_address_chunks::<F>(&r_address);

        // Build RA polynomials by iterating over trace and computing PCs on the fly
        self.ra = r_address_chunks
            .iter()
            .enumerate()
            .map(|(i, r_address_chunk)| {
                let ra_i: Vec<Option<u8>> = self
                    .trace
                    .par_iter()
                    .map(|cycle| {
                        let pc = self.bytecode_preprocessing.get_pc(cycle);
                        Some(self.params.one_hot_params.bytecode_pc_chunk(pc, i))
                    })
                    .collect();
                RaPolynomial::new(Arc::new(ra_i), EqPolynomial::evals(r_address_chunk))
            })
            .collect();

        // Drop trace and preprocessing - no longer needed after this
        self.trace = Arc::new(Vec::new());
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T>
for BytecodeReadRafSumcheckProver<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(skip_all, name = "BytecodeReadRafSumcheckProver::compute_message")]
    fn compute_message(&mut self, round: usize, _previous_claim: F) -> UniPoly<F> {
        if round < self.params.log_K {
            const DEGREE: usize = 2;

            // Evaluation at [0, 2] for each stage.
            let eval_per_stage: [[F; DEGREE]; N_STAGES] = (0..self.params.val_polys[0].len() / 2)
                .into_par_iter()
                .map(|i| {
                    let ra_evals = self.F.each_ref().map(|poly| {
                        poly.sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh)
                    });

                    let int_evals =
                        self.params.int_poly
                            .sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);

                    // We have a separate Val polynomial for each stage
                    // Additionally, for stages 1 and 3 we have an Int polynomial for RAF
                    // So we would have:
                    // Stage 1: Val_1 + gamma^5 * Int
                    // Stage 2: Val_2
                    // Stage 3: Val_3 + gamma^4 * Int
                    // Stage 4: Val_4
                    // Stage 5: Val_5
                    // Which matches with the input claim:
                    // rv_1 + gamma * rv_2 + gamma^2 * rv_3 + gamma^3 * rv_4 + gamma^4 * rv_5 + gamma^5 * raf_1 + gamma^6 * raf_3
                    let mut val_evals = self
                        .params.val_polys
                        .iter()
                        // Val polynomials
                        .map(|val| val.sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh))
                        // Here are the RAF polynomials and their powers
                        .zip([Some(&int_evals), None, Some(&int_evals), None, None])
                        .zip([Some(self.params.gamma_powers[5]), None, Some(self.params.gamma_powers[4]), None, None])
                        .map(|((val_evals, int_evals), gamma)| {
                            std::array::from_fn::<F, DEGREE, _>(|j| {
                                val_evals[j]
                                    + int_evals.map_or(F::zero(), |int_evals| {
                                    int_evals[j] * gamma.unwrap()
                                })
                            })
                        });

                    array::from_fn(|stage| {
                        let [ra_at_0, ra_at_2] = ra_evals[stage];
                        let [val_at_0, val_at_2] = val_evals.next().unwrap();
                        [ra_at_0 * val_at_0, ra_at_2 * val_at_2]
                    })
                })
                .reduce(
                    || [[F::zero(); DEGREE]; N_STAGES],
                    |a, b| array::from_fn(|i| array::from_fn(|j| a[i][j] + b[i][j])),
                );

            let mut round_polys: [_; N_STAGES] = array::from_fn(|_| UniPoly::zero());
            let mut agg_round_poly = UniPoly::zero();

            for (stage, evals) in eval_per_stage.into_iter().enumerate() {
                let [eval_at_0, eval_at_2] = evals;
                let eval_at_1 = self.prev_round_claims[stage] - eval_at_0;
                let round_poly = UniPoly::from_evals(&[eval_at_0, eval_at_1, eval_at_2]);
                agg_round_poly += &(&round_poly * self.params.gamma_powers[stage]);
                round_polys[stage] = round_poly;
            }

            self.prev_round_polys = Some(round_polys);

            agg_round_poly
        } else {
            let degree = <Self as SumcheckInstanceProver<F, T>>::degree(self);

            let out_len = self.gruen_eq_polys[0].E_out_current().len();
            let in_len = self.gruen_eq_polys[0].E_in_current().len();
            let in_n_vars = in_len.log_2();

            // Evaluations on [1, ..., degree - 2, inf] (for each stage).
            let mut evals_per_stage: [Vec<F>; N_STAGES] = (0..out_len)
                .into_par_iter()
                .map(|j_hi| {
                    let mut ra_eval_pairs = vec![(F::zero(), F::zero()); self.ra.len()];
                    let mut ra_prod_evals = vec![F::zero(); degree - 1];
                    let mut evals_per_stage: [_; N_STAGES] =
                        array::from_fn(|_| vec![F::Unreduced::zero(); degree - 1]);

                    for j_lo in 0..in_len {
                        let j = j_lo + (j_hi << in_n_vars);

                        for (i, ra_i) in self.ra.iter().enumerate() {
                            let ra_i_eval_at_j_0 = ra_i.get_bound_coeff(j * 2);
                            let ra_i_eval_at_j_1 = ra_i.get_bound_coeff(j * 2 + 1);
                            ra_eval_pairs[i] = (ra_i_eval_at_j_0, ra_i_eval_at_j_1);
                        }
                        // Eval prod_i ra_i(x).
                        eval_linear_prod_assign(&ra_eval_pairs, &mut ra_prod_evals);

                        for stage in 0..N_STAGES {
                            let eq_in_eval = self.gruen_eq_polys[stage].E_in_current()[j_lo];
                            for i in 0..degree - 1 {
                                evals_per_stage[stage][i] +=
                                    eq_in_eval.mul_unreduced::<9>(ra_prod_evals[i]);
                            }
                        }
                    }

                    array::from_fn(|stage| {
                        let eq_out_eval = self.gruen_eq_polys[stage].E_out_current()[j_hi];
                        evals_per_stage[stage]
                            .iter()
                            .map(|v| eq_out_eval * F::from_montgomery_reduce(*v))
                            .collect()
                    })
                })
                .reduce(
                    || array::from_fn(|_| vec![F::zero(); degree - 1]),
                    |a, b| array::from_fn(|i| zip_eq(&a[i], &b[i]).map(|(a, b)| *a + *b).collect()),
                );
            // Multiply by bound values.
            let bound_val_evals = self.bound_val_evals.as_ref().unwrap();
            for (stage, evals) in evals_per_stage.iter_mut().enumerate() {
                evals.iter_mut().for_each(|v| *v *= bound_val_evals[stage]);
            }

            let mut round_polys: [_; N_STAGES] = array::from_fn(|_| UniPoly::zero());
            let mut agg_round_poly = UniPoly::zero();

            // Obtain round poly for each stage and perform RLC.
            for (stage, evals) in evals_per_stage.iter().enumerate() {
                let claim = self.prev_round_claims[stage];
                let round_poly = self.gruen_eq_polys[stage].gruen_poly_from_evals(evals, claim);
                agg_round_poly += &(&round_poly * self.params.gamma_powers[stage]);
                round_polys[stage] = round_poly;
            }

            self.prev_round_polys = Some(round_polys);

            agg_round_poly
        }
    }

    #[tracing::instrument(skip_all, name = "BytecodeReadRafSumcheckProver::ingest_challenge")]
    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        if let Some(prev_round_polys) = self.prev_round_polys.take() {
            self.prev_round_claims = prev_round_polys.map(|poly| poly.evaluate(&r_j));
        }

        if round < self.params.log_K {
            self.params
                .val_polys
                .iter_mut()
                .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::LowToHigh));
            self.params
                .int_poly
                .bind_parallel(r_j, BindingOrder::LowToHigh);
            self.F
                .iter_mut()
                .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::LowToHigh));
            self.r_address_prime.push(r_j);
            if round == self.params.log_K - 1 {
                self.init_log_t_rounds();
            }
        } else {
            self.ra
                .iter_mut()
                .for_each(|ra| ra.bind_parallel(r_j, BindingOrder::LowToHigh));
            self.gruen_eq_polys
                .iter_mut()
                .for_each(|poly| poly.bind(r_j));
        }
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        let (r_address, r_cycle) = opening_point.split_at(self.params.log_K);

        // Compute r_address_chunks with proper padding
        let r_address_chunks = self
            .params
            .one_hot_params
            .compute_r_address_chunks::<F>(&r_address.r);

        for i in 0..self.params.d {
            accumulator.append_sparse(
                transcript,
                vec![CommittedPolynomial::BytecodeRa(i)],
                SumcheckId::BytecodeReadRaf,
                r_address_chunks[i].clone(),
                r_cycle.clone().into(),
                vec![self.ra[i].final_sumcheck_claim()],
            );
        }
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

pub struct BytecodeReadRafSumcheckVerifier<F: JoltField> {
    params: BytecodeReadRafSumcheckParams<F>,
}

impl<F: JoltField> BytecodeReadRafSumcheckVerifier<F> {
    pub fn gen(
        bytecode_preprocessing: &BytecodePreprocessing,
        n_cycle_vars: usize,
        one_hot_params: &OneHotParams,
        opening_accumulator: &VerifierOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        Self {
            params: BytecodeReadRafSumcheckParams::gen(
                bytecode_preprocessing,
                n_cycle_vars,
                one_hot_params,
                opening_accumulator,
                transcript,
            ),
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
for BytecodeReadRafSumcheckVerifier<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        let (r_address_prime, r_cycle_prime) = opening_point.split_at(self.params.log_K);
        // r_cycle is bound LowToHigh, so reverse

        let int_poly = self.params.int_poly.evaluate(&r_address_prime.r);

        let ra_claims = (0..self.params.d).map(|i| {
            accumulator
                .get_committed_polynomial_opening(
                    CommittedPolynomial::BytecodeRa(i),
                    SumcheckId::BytecodeReadRaf,
                )
                .1
        });

        // We have a separate Val polynomial for each stage
        // Additionally, for stages 1 and 3 we have an Int polynomial for RAF
        // So we would have:
        // Stage 1: gamma^0 * (Val_1 + gamma^5 * Int)
        // Stage 2: gamma^1 * (Val_2)
        // Stage 3: gamma^2 * (Val_3 + gamma^4 * Int)
        // Stage 4: gamma^3 * (Val_4)
        // Stage 5: gamma^4 * (Val_5)
        // Which matches with the input claim:
        // rv_1 + gamma * rv_2 + gamma^2 * rv_3 + gamma^3 * rv_4 + gamma^4 * rv_5 + gamma^5 * raf_1 + gamma^6 * raf_3
        let val = self
            .params
            .val_polys
            .iter()
            .zip(&self.params.r_cycles)
            .zip(&self.params.gamma_powers)
            .zip([
                int_poly * self.params.gamma_powers[5], // RAF for Stage1
                F::zero(),                              // There's no raf for Stage2
                int_poly * self.params.gamma_powers[4], // RAF for Stage3
                F::zero(),                              // There's no raf for Stage4
                F::zero(),                              // There's no raf for Stage5
            ])
            .map(|(((val, r_cycle), gamma), int_poly)| {
                (val.evaluate(&r_address_prime.r) + int_poly)
                    * EqPolynomial::<F>::mle(r_cycle, &r_cycle_prime.r)
                    * gamma
            })
            .sum::<F>();

        ra_claims.fold(val, |running, ra_claim| running * ra_claim)
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[<F as JoltField>::Challenge],
    ) {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        let (r_address, r_cycle) = opening_point.split_at(self.params.log_K);

        // Compute r_address_chunks with proper padding
        let r_address_chunks = self
            .params
            .one_hot_params
            .compute_r_address_chunks::<F>(&r_address.r);

        (0..self.params.d).for_each(|i| {
            let opening_point = [&r_address_chunks[i][..], &r_cycle.r].concat();
            accumulator.append_sparse(
                transcript,
                vec![CommittedPolynomial::BytecodeRa(i)],
                SumcheckId::BytecodeReadRaf,
                opening_point,
            );
        });
    }
}

#[derive(Allocative, Clone)]
pub struct BytecodeReadRafSumcheckParams<F: JoltField> {
    /// Index `i` stores `gamma^i`.
    pub gamma_powers: Vec<F>,
    /// RLC of stage rv_claims and RAF claims (per Stage1/Stage3) used as the sumcheck LHS.
    pub input_claim: F,
    /// RaParams
    pub one_hot_params: OneHotParams,
    /// Bytecode length.
    pub K: usize,
    /// log2(K) and log2(T) used to determine round counts.
    pub log_K: usize,
    pub log_T: usize,
    /// Number of address chunks (and RA polynomials in the product).
    pub d: usize,
    /// Stage Val polynomials evaluated over address vars.
    pub val_polys: [MultilinearPolynomial<F>; N_STAGES],
    /// Stage rv claims.
    pub rv_claims: [F; N_STAGES],
    pub raf_claim: F,
    pub raf_shift_claim: F,
    /// Identity polynomial over address vars used to inject RAF contributions.
    pub int_poly: IdentityPolynomial<F>,
    pub r_cycles: [Vec<F::Challenge>; N_STAGES],
}

impl<F: JoltField> BytecodeReadRafSumcheckParams<F> {
    #[tracing::instrument(skip_all, name = "BytecodeReadRafSumcheckParams::gen")]
    /// 生成 BytecodeReadRafSumcheckParams 实例。
    ///具体来说，它负责：
    /// 收集挑战数：从 Transcript 中获取随机挑战数 <span>\gamma</span>（Gamma）。
    /// 计算目标值：从之前的证明步骤（Opening Accumulator）中提取多项式的评估值，并通过随机线性组合（RLC）计算出当前 Sumcheck 需要证明的总和（LHS）。
    /// 计算 Val 多项式：为了验证效率，通过一次并行扫描（Fused Pass）计算所有 5 个阶段所需的 Val(k) 多项式，而不是扫描 5 次。
    /// 此函数是协议设置阶段的核心，负责：
    /// 1. 从 Transcript 获取随机挑战数 (Gamma)。
    /// 2. 从 OpeningAccumulator 获取各个子协议生成的 Claims (声称值)。
    /// 3. 计算最终的 input_claim (Verifier 期望 Prover 证明的总和)。
    /// 4. 预计算所有阶段的 Val 多项式。
    /// 生成 BytecodeReadRafSumcheck 实例的参数
    /// 主要任务：收集前 5 个 Stage 关于“指令读取”的承诺，并结合静态字节码构建验证多项式。
    pub fn gen(
        bytecode_preprocessing: &BytecodePreprocessing, // 预处理的静态字节码 (ROM)
        n_cycle_vars: usize,                            // 对应 trace 长度的变量数 (log_2(Time))
        one_hot_params: &OneHotParams,                  // One-hot 编码参数
        opening_accumulator: &dyn OpeningAccumulator<F>, // 存储前序 Stage 计算结果的累加器
        transcript: &mut impl Transcript,               // Fiat-Shamir 挑战生成器
    ) -> Self {
        // =========================================================
        // 1. 生成全局随机权重 Gamma
        // 用于将 5 个不同的 Stage 的 Check + 2 个 RAF Check 合并为一个大的 Sumcheck
        // =========================================================
        // Compute powers of scalar q : (1, q, q^2, ..., q^(len-1))
        let gamma_powers = transcript.challenge_scalar_powers(7);

        let bytecode = &bytecode_preprocessing.bytecode;

        // =========================================================
        // 2. 为每个子 Stage 生成内部混合系数
        // 前面的每个 Stage 可能包含多个子多项式（例如 Stage 1 有多个 Circuit Flags），
        // 这里需要生成对应的权重将它们压缩。
        // =========================================================

        // Stage 1: Spartan Outer (验证操作码 Opcode 是否正确)
        let stage1_gammas: Vec<F> = transcript.challenge_scalar_powers(2 + NUM_CIRCUIT_FLAGS);
        // Stage 2: Product Virtualization (验证乘法相关逻辑)
        let stage2_gammas: Vec<F> = transcript.challenge_scalar_powers(4);
        // Stage 3: Shift Sumcheck (验证位移操作)
        let stage3_gammas: Vec<F> = transcript.challenge_scalar_powers(9);
        // Stage 4: Register R/W Checking (验证寄存器读写索引 rd, rs1, rs2)
        let stage4_gammas: Vec<F> = transcript.challenge_scalar_powers(3);
        // Stage 5: Instruction Lookups (验证指令查找表的结果)
        let stage5_gammas: Vec<F> = transcript.challenge_scalar_powers(2 + NUM_LOOKUP_TABLES);

        // =========================================================
        // 3. 计算各个 Stage 的 "声明值" (RV Claims)
        // 这些值代表了：“在前几个 Stage 中，Prover 声称从 Bytecode 中读到了什么值”。
        // opening_accumulator 里存的是 MLE(r) 的求值结果。即p(r)
        // =========================================================
        let rv_claim_1 = Self::compute_rv_claim_1(opening_accumulator, &stage1_gammas);
        let rv_claim_2 = Self::compute_rv_claim_2(opening_accumulator, &stage2_gammas);
        let rv_claim_3 = Self::compute_rv_claim_3(opening_accumulator, &stage3_gammas);
        let rv_claim_4 = Self::compute_rv_claim_4(opening_accumulator, &stage4_gammas);
        let rv_claim_5 = Self::compute_rv_claim_5(opening_accumulator, &stage5_gammas);
        let rv_claims = [rv_claim_1, rv_claim_2, rv_claim_3, rv_claim_4, rv_claim_5];

        // =========================================================
        // 4. 预计算 Eq 多项式 (用于过滤寄存器)
        // Stage 4 和 5 涉及到寄存器索引的检查。
        // 我们需要构建 eq(r_register, index) 来验证 Trace 里的寄存器索引是否匹配。
        // =========================================================

        // 获取 Stage 4 中用于校验寄存器读写的随机点 r
        let r_register_4 = opening_accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RdWa, // 目标是 Rd/Wa (Write Address)
                SumcheckId::RegistersReadWriteChecking,
            )
            .0
            .r; // 取出随机向量 r
        // 计算 eq(r, x) 在所有可能的寄存器索引上的值 (0..31)
        let eq_r_register_4 =
            EqPolynomial::<F>::evals(&r_register_4[..(REGISTER_COUNT as usize).log_2()]);

        // 同上，处理 Stage 5
        let r_register_5 = opening_accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RdWa,
                SumcheckId::RegistersValEvaluation,
            )
            .0
            .r;
        let eq_r_register_5 =
            EqPolynomial::<F>::evals(&r_register_5[..(REGISTER_COUNT as usize).log_2()]);

        // =========================================================
        // 5. [核心优化] 融合计算 (Fused Computation)
        // 这是为了性能。我们需要遍历静态的 Bytecode 来计算“理论上的多项式值”。
        // 与其对 Bytecode 遍历 5 次（为每个 Stage 算一次），
        // 不如遍历 1 次，同时计算出 5 个 Stage 所需的系数 (val_polys)。
        // 这将生成这一轮 Sumcheck 所需的基础多项式。
        // =========================================================
        let val_polys = Self::compute_val_polys(
            bytecode,
            &eq_r_register_4,
            &eq_r_register_5,
            &stage1_gammas,
            &stage2_gammas,
            &stage3_gammas,
            &stage4_gammas,
            &stage5_gammas,
        );

        // =========================================================
        // 身份多项式 (Identity Polynomial)
        // ---------------------------------------------------------
        // 目标：生成静态的 PC 地址序列，作为“位置指纹”绑定到每一行指令上。
        // 它代表了向量 v = [0, 1, 2, ..., K-1] 的多线性扩张 (MLE)。
        //
        // 作用说明：
        // 1. **PC 值绑定 (Location Binding)**：
        //    它确保了指令的内容（如 Opcode, Imm）是与它在 ROM 中的地址（PC）强绑定的。
        //    例如在 Stage 3 中，Val 公式包含 `γ * PC` 项。Verifier 使用此多项式计算
        //    "当前行理论上应有的 PC 值"，防止 Prover 将位于 PC=100 的指令“剪切粘贴”到
        //    PC=200 处去执行。
        //
        // 2. **注入 RAF (Read Access Frequency) 贡献**：
        //    RAF 协议计算了运行过程中每个 PC 被访问的频次。
        //    为了将 RAF 的结果（动态执行流统计）与 Bytecode Sumcheck（静态 ROM 内容检查）
        //    连接起来，我们需要在 Sumcheck 的 Claim 中包含 PC 的贡献。
        //    int_poly 提供了这个 "Base PC" 的值，使得 Val_1(k) = PC(k) + ... 成立。
        //
        // 举例：
        // 假设 Bytecode 长度 K=4 (log_K=2)。
        // int_poly 对应的就是向量 I = [0, 1, 2, 3]。
        // 当 Sumcheck 进行到随机点 r=(r0, r1) 时，int_poly.evaluate(r) 会计算出
        // 该点对应的“广义地址值”，用于验证上述逻辑。
        // =========================================================
        let int_poly = IdentityPolynomial::new(one_hot_params.bytecode_k.log_2());
        // =========================================================
        // 6. 获取 RAF Claims (读访问频次声明)
        // 这里从 accumulator 获取之前的 Sumcheck 结果，这些结果隐含了
        // "PC 这一列在之前的检查中被访问了多少次"。
        // =========================================================
        let (_, raf_claim) = opening_accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::PC, SumcheckId::SpartanOuter);
        let (_, raf_shift_claim) = opening_accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::PC, SumcheckId::SpartanShift);

        // =========================================================
        // 7. 组合最终的 Input Claim (总目标值)
        // Sumcheck 的目标是证明：Sum(Poly(x)) = Input_Claim
        // input_claim = (Claim1 * γ^0) + ... + (Claim5 * γ^4) + (RAF * γ^5) + (Shift * γ^6)
        // =========================================================
        let input_claim = [
            rv_claim_1,
            rv_claim_2,
            rv_claim_3,
            rv_claim_4,
            rv_claim_5,
            raf_claim,
            raf_shift_claim,
        ]
            .iter()
            .zip(&gamma_powers)
            .map(|(claim, g)| *claim * g) // 线性组合
            .sum();

        // =========================================================
        // 8. 收集时间绑定随机点 (Time Binding / r_cycle)
        // ---------------------------------------------------------
        // 目标：从 OpeningAccumulator 中提取出用于“时间维度”加权的随机挑战向量 r。
        //
        // 意义：这个 Sumcheck 是要证明：
        //      Sum_t( Eq(r_cycle, t) * Val(RAM[PC_t]) ) == Claim
        // 既然 Claim 是之前的动态检查产生的，那么这里的 r_cycle 必须严格等于
        // 之前动态检查时 Verifier 下发的那个随机挑战，否则等式无法成立。
        // =========================================================

        // --- 简单情况：纯时间维度的子协议 ---
        // Stage 1 (Spartan Outer): 变量仅包含 log(T) 维度的 Cycle
        let (r_cycle_1, _) = opening_accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::Imm, SumcheckId::SpartanOuter);

        // Stage 2: 变量仅包含 log(T) 维度的 Cycle
        let (r_cycle_2, _) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::OpFlags(CircuitFlags::Jump),
            SumcheckId::SpartanProductVirtualization,
        );

        // Stage 3: 变量仅包含 log(T) 维度的 Cycle
        let (r_cycle_3, _) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::UnexpandedPC,
            SumcheckId::SpartanShift,
        );

        // --- 复杂情况：复合维度 (寄存器 x 时间) 的子协议 ---
        // ... (重复获取其他 Stage 的 r_cycle)

        // Stage 4 (寄存器读写检查 Register RAM):
        // 这里的多项式定义在 (寄存器索引 || 时间周期) 上。
        // Challenge r 的结构通常是 [ r_register_bits ... | r_cycle_bits ... ]。
        // 我们需要把前 log(RegisterCount) 个变量剥离（它们用于之前的寄存器指纹生成），
        // 剩下的才是代表“时间”的随机数。
        let (r, _) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::Rs1Ra,
            SumcheckId::RegistersReadWriteChecking,
        );
        // split_at 参数为 5 (32个寄存器 log2=5)。
        // 结果：丢弃前5个，保留剩下的作为 r_cycle_4。
        let (_, r_cycle_4) = r.split_at((REGISTER_COUNT as usize).log_2());

        // Stage 5 (指令查表检查):
        // 原理同 Stage 4，也是复合维度。
        let (r, _) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RdWa,
            SumcheckId::RegistersValEvaluation,
        );
        let (_, r_cycle_5) = r.split_at((REGISTER_COUNT as usize).log_2());

        // 将 5 个 Stage 提取出的时间随机向量打包，
        // 用于后续构造 GruenSplitEqPolynomial (分段 Eq 多项式)，
        // 以便在 Sumcheck 过程中快速计算 Eq(r_cycle, t)。
        let r_cycles = [
            r_cycle_1.r,
            r_cycle_2.r,
            r_cycle_3.r,
            r_cycle_4.r,
            r_cycle_5.r,
        ];

        // 返回构造好的参数结构体
        Self {
            gamma_powers,
            input_claim,
            one_hot_params: one_hot_params.clone(),
            K: one_hot_params.bytecode_k,
            log_K: one_hot_params.bytecode_k.log_2(),
            d: one_hot_params.bytecode_d,
            log_T: n_cycle_vars,
            val_polys,
            rv_claims,
            raf_claim,
            raf_shift_claim,
            int_poly,
            r_cycles,
        }
    }

    /// Fused computation of all Val polynomials in a single parallel pass over bytecode.
    ///
    /// This computes all 5 stage-specific Val(k) polynomials simultaneously, avoiding
    /// 5 separate passes through the bytecode. Each stage has its own gamma powers
    /// and formula for Val(k).

    /// 通过一次并行扫描（Fused Computation）计算所有 5 个 Stage 所需的 Val 多项式。
    ///  该函数遍历静态字节码，利用随机挑战数 (Gammas) 将指令的各种属性（如操作码、寄存器索引、标志位）
    /// 压缩为 5 个多线性多项式。这些多项式将作为 Sumcheck 中的 "Lookup Table" 真值。
    ///
    /// ### 算法原理：融合优化 (Fused Optimization)
    ///
    /// 在标准的 Sumcheck 协议中，通常需要针对每个验证目标（Stage）分别计算其所有的多项式评估值。
    /// 由于 Jolt 包含 5 个不同的验证阶段（从指令解码到寄存器读写），朴素的做法是扫描静态字节码（Bytecode）5 次。
    ///
    /// 此函数利用了多项式计算的独立性，执行一次“融合扫描”：
    /// 1. **并行处理**：使用 `rayon` 将字节码数组切分为多个块（Chunks）并行处理。
    /// 2. **单次遍历**：对每条指令（指令地址 k），只需加载一次，就能同时提取出 5 个 Stage 所需的所有特征（标志位、操作数、寄存器索引等）。
    /// 3. **实时计算**：直接在循环内部计算这一行指令在 5 个不同 RLC（随机线性组合）公式下的贡献值 `v0[k], v1[k], ..., v4[k]`。
    ///
    /// ### 对应关系
    /// - `vals[0]` (Stage 1): 验证指令基础解码 (PC, Imm, Opcode Flags).
    /// - `vals[1]` (Stage 2): 验证控制流逻辑 (Jump, Branch).
    /// - `vals[2]` (Stage 3): 验证操作数选择逻辑 (Input Muxing).
    /// - `vals[3]` (Stage 4): 验证寄存器读取索引 (Register R/W Indices).
    /// - `vals[4]` (Stage 5): 验证寄存器写入索引与查找表选择 (Register Write & Lookup ID).
    #[allow(clippy::too_many_arguments)]
    fn compute_val_polys(
        bytecode: &[Instruction],       // 静态字节码 (ROM)
        eq_r_register_4: &[F],          // Stage 4 的寄存器指纹表 (预计算的 eq(r, index))
        eq_r_register_5: &[F],          // Stage 5 的寄存器指纹表
        stage1_gammas: &[F],            // Stage 1 的随机压缩因子
        stage2_gammas: &[F],            // Stage 2 的随机压缩因子 (控制流)
        stage3_gammas: &[F],            // Stage 3 的随机压缩因子 (操作数选择)
        stage4_gammas: &[F],            // Stage 4 的随机压缩因子 (寄存器读写)
        stage5_gammas: &[F],            // Stage 5 的随机压缩因子 (指令查找表)
    ) -> [MultilinearPolynomial<F>; N_STAGES] {
        let K = bytecode.len();// ROM 大小
        // 1. 预分配内存
        // 预分配内存：创建 5 个长度为 K 的向量，用于存储每个 Stage 的计算结果。
        // unsafe_allocate_zero_vec 避免了初始化开销，假设后续会完全覆盖写入。
        let mut vals: [Vec<F>; N_STAGES] = array::from_fn(|_| unsafe_allocate_zero_vec(K));
        let [v0, v1, v2, v3, v4] = &mut vals;

        // Fused parallel iteration: 并行迭代字节码，一次性计算 5 个值

        // 2. 并行融合迭代 (Fused Parallel Iteration)
        // 使用 Rayon (par_iter) 并行遍历字节码。
        // 同时操作 5 个输出向量 (v0...v4)，一次读取指令，计算所有属性。
        bytecode
            .par_iter()
            // 使用 zip 将 5 个结果向量与指令流对齐，以便在闭包中同时写入
            .zip(v0.par_iter_mut())
            .zip(v1.par_iter_mut())
            .zip(v2.par_iter_mut())
            .zip(v3.par_iter_mut())
            .zip(v4.par_iter_mut())
            .for_each(|(((((instruction, o0), o1), o2), o3), o4)| {
                // 标准化指令并提取各类标志位，避免重复解析
                // 1. 预处理指令与提取特征标志位
                // - normalize(): 对原始指令进行标准化处理（例如统一立即数格式），确保后续数值计算的一致性。
                // - circuit_flags / instr_flags: 一次性提取指令的所有布尔属性（如是否是跳转指令、是否操作内存、操作数类型等）。
                //   这些 Flags 将在下方作为指示函数（Indicator Functions，取值 0 或 1），参与到不同 Stage 的加权求和（RLC）中，
                //   从而避免在后续计算中频繁重复解析指令。
                let instr = instruction.normalize();
                info!("Instruction: {:#?}", instr);
                let circuit_flags = instruction.circuit_flags();
                let instr_flags = instruction.instruction_flags();
                info!("Circuit Flags: {:#?}", circuit_flags);
                info!("Instruction Flags: {:#?}", instr_flags);

                // =========================================================
                // Stage 1: Spartan Outer Sumcheck (基础指令属性)
                // ---------------------------------------------------------
                // 目标：验证 (Address, Immediate, Opcode Flags)
                // 原理：RLC (随机线性组合)。将多个属性压扁成一个数。
                // 公式：Val = Address + γ_1 * Imm + Σ (γ_i * Flag_i)
                // 作用：防止 Prover 篡改指令类型（比如把 ADD 改成 MUL）或立即数。
                // =========================================================
                {
                    // 1. 累加地址 (Address)
                    let mut lc = F::from_u64(instr.address as u64);
                    // 2. 累加立即数 (Immediate)
                    lc += instr.operands.imm.field_mul(stage1_gammas[1]);
                    // 一致性检查：压缩指令不能更新 UnexpandedPC
                    // Sanity check: 确保压缩标志位逻辑正常
                    debug_assert!(
                        !circuit_flags[CircuitFlags::IsCompressed]
                            || !circuit_flags[CircuitFlags::DoNotUpdateUnexpandedPC]
                    );
                    // 遍历并累加所有电路标志位的贡献
                    // 3. 累加电路标志位 (Circuit Flags)
                    // 遍历每一个 Flag，如果该指令激活了这个 Flag (如 IsADD 为true)，则加上对应的 Gamma
                    for (flag, gamma_power) in circuit_flags.iter().zip(stage1_gammas[2..].iter()) {
                        if *flag {
                            lc += *gamma_power;
                        }
                    }
                    *o0 = lc; // 写入 Stage 1 结果，写入 v0 向量
                }
                // =========================================================
                // Stage 2: Product Virtualization (控制流属性)
                // ---------------------------------------------------------
                // 目标：验证跳转 (Jump) 和分支 (Branch) 逻辑
                // 原理：只累加与控制流相关的 Flag。
                // 作用：确保 Prover 正确识别了跳转指令。如果 Prover 把 JUMP 当作 ADD 执行，
                //      这里的真值检查会导致 Sumcheck 失败。
                // Val(k) = jump_flag + γ·branch_flag
                //          + γ²·rd_not_zero + γ³·write_lookup_to_rd
                // 含义：这些参数决定了 PC 跳转行为和是否写寄存器。
                // =========================================================
                {
                    let mut lc = F::zero();
                    if circuit_flags[CircuitFlags::Jump] {
                        lc += stage2_gammas[0];
                    }
                    if instr_flags[InstructionFlags::Branch] {
                        lc += stage2_gammas[1];
                    }
                    if instr_flags[InstructionFlags::IsRdNotZero] {
                        lc += stage2_gammas[2];
                    }
                    if circuit_flags[CircuitFlags::WriteLookupOutputToRD] {
                        lc += stage2_gammas[3];
                    }
                    *o1 = lc; // 写入 Stage 2 结果，写入 v1 向量
                }

                // ===== Stage 3 (Shift sumcheck / Input Muxing) =====
                // 目标：验证指令的“输入数据通路”配置。
                // 即验证：ALU 的左输入和右输入分别取自哪里？是寄存器值、立即数、还是 PC？
                // 同时也验证虚拟指令序列（Virtual Sequence）的元数据。
                //
                // 公式：Val(k) = imm
                //          + γ^1 · PC
                //          + γ^2 · Left_is_RS1 + γ^3 · Left_is_PC
                //          + γ^4 · Right_is_RS2 + γ^5 · Right_is_Imm
                //          + ... (Noop, Virtual, FirstInSeq)
                {
                    // 1. 基础负载：立即数 (Immediate)
                    // 许多指令（如 ADDI, SHL）直接依赖立即数作为输入之一，因此将其作为常数项
                    let mut lc = F::from_i128(instr.operands.imm);

                    // 2. 绑定当前指令的 PC 地址
                    // 乘以 γ^1，防止指令位置被交换
                    lc += stage3_gammas[1].mul_u64(instr.address as u64);

                    // 3. 左操作数来源选择 (Left Mux)
                    // 如果该指令定义左操作数取自 rs1 寄存器 (例如 ADD, SUB)
                    if instr_flags[InstructionFlags::LeftOperandIsRs1Value] {
                        lc += stage3_gammas[2];
                    }
                    // 如果该指令定义左操作数取自 PC (例如 AUIPC, JAL)
                    if instr_flags[InstructionFlags::LeftOperandIsPC] {
                        lc += stage3_gammas[3];
                    }

                    // 4. 右操作数来源选择 (Right Mux)
                    // 如果右操作数取自 rs2 寄存器 (例如 ADD, SLL)
                    if instr_flags[InstructionFlags::RightOperandIsRs2Value] {
                        lc += stage3_gammas[4];
                    }
                    // 如果右操作数是立即数 (例如 ADDI, SLLI)
                    if instr_flags[InstructionFlags::RightOperandIsImm] {
                        lc += stage3_gammas[5];
                    }

                    // 5. 特殊指令属性
                    // IsNoop: 是否是填充指令（不改变状态）
                    if instr_flags[InstructionFlags::IsNoop] {
                        lc += stage3_gammas[6];
                    }
                    // VirtualInstruction: 是否是 Jolt 拆解后的微指令的一部分
                    if circuit_flags[CircuitFlags::VirtualInstruction] {
                        lc += stage3_gammas[7];
                    }
                    // IsFirstInSequence: 是否是微指令序列的第一条（用于重置序列计数器）
                    if circuit_flags[CircuitFlags::IsFirstInSequence] {
                        lc += stage3_gammas[8];
                    }

                    // 将计算出的唯一指纹写入 Stage 3 的结果向量
                    *o2 = lc;
                }

                // =========================================================
                // Stage 4: Register R/W Checking (寄存器索引检查)
                // ---------------------------------------------------------
                // 目标：验证本条指令实际操作的寄存器索引 (rd, rs1, rs2) 是否正确。
                //
                // 问题：直接相加寄存器号 (如 1+2=3) 容易产生碰撞 (0+3=3)。
                // 解决方案：使用 Schwartz-Zippel 引理衍生出的 Fingerprinting 技术。
                //          不直接使用 index，而是使用预先计算好的随机指纹表 `eq_r_register_4`。
                //          该表是基于 Verifier 的随机挑战 r 生成的：Table[i] = Eq(i, r)。
                //
                // 公式：Val(k) = Fingerprint(rd) + γ^1·Fingerprint(rs1) + γ^2·Fingerprint(rs2)
                // =========================================================
                {
                    // 1. 获取目标寄存器 (Dest Register) 的指纹
                    // 查表逻辑：如果指令有写回目标(rd)，则查表 Eq(rd, r)；如果是分支/存储指令无rd，则为0。
                    let rd_eq = instr
                        .operands
                        .rd
                        .map_or(F::zero(), |r| eq_r_register_4[r as usize]);

                    // 2. 获取源寄存器 1 (Source Register 1) 的指纹
                    // 例如 ADD x1, x2, x3 中的 x2。无 rs1 则为 0。
                    let rs1_eq = instr
                        .operands
                        .rs1
                        .map_or(F::zero(), |r| eq_r_register_4[r as usize]);

                    // 3. 获取源寄存器 2 (Source Register 2) 的指纹
                    // 例如 ADD x1, x2, x3 中的 x3。ADDI 指令没有 rs2，则为 0。
                    let rs2_eq = instr
                        .operands
                        .rs2
                        .map_or(F::zero(), |r| eq_r_register_4[r as usize]);

                    // 4. 加权聚合 (Random Linear Combination)
                    // 使用 Stage 4 专属的 Gamma 因子进行压缩，生成唯一的校验值。
                    // o3 是当前指令在 Stage 4 多项式中的系数值。
                    *o3 = rd_eq * stage4_gammas[0]       // weight: 1 (gamma^0)
                        + rs1_eq * stage4_gammas[1]      // weight: gamma^1
                        + rs2_eq * stage4_gammas[2];     // weight: gamma^2
                }
                // =========================================================
                // Stage 5: Instruction Lookups (查找表路由检查)
                // ---------------------------------------------------------
                // 目标：验证该指令应该去查哪个查找表 (Lookup Table)。验证寄存器写回索引和查找表 ID。
                //
                // 背景：在 Jolt 中，大多数运算不是通过电路计算的，而是通过查表 (Lookup) 完成的。
                //      VM 需要知道当前行指令（如 ADD）对应哪个表（如 Add Table）。
                //      同时也需要验证查表结果最终写回到哪个寄存器。
                //
                // 公式：Val(k) = Fingerprint(rd)
                //            + γ^1 · IsNotInterleaved
                //            + γ^(2 + table_id) · 1
                //
                // 含义：确认指令最终写入哪个寄存器，数据切分模式，以及使用了数十种查找表中的哪一种。
                // =========================================================
                {
                    // 1. Rd 寄存器索引检查 (Register Write Fingerprint)
                    // 再次验证 RD 的目的是为了将其与 Stage 4 中的读操作以及 Stage 5 中的查表结果绑定。
                    // 只有当 Prover 在“写回”阶段操作的是同一个指纹对应的寄存器，验证才通过。
                    // 使用 Stage 5 专属的 challenge r 生成的指纹表 eq_r_register_5。
                    let mut lc = instr
                        .operands
                        .rd
                        .map_or(F::zero(), |r| eq_r_register_5[r as usize]);

                    // 2. 验证操作数编码模式 (Interleaved Flag)
                    // Jolt 有两种查表模式：直接查表或交错(Interleaved)查表。
                    // Interleaved 用于处理需要两个操作数按位混合的情况（如 ADD/SUB），
                    // 此 Flag 决定了 Lookup Query 的构建方式。
                    if !circuit_flags.is_interleaved_operands() {
                        lc += stage5_gammas[1]; // 如果是非交错模式，加上权重 γ^1
                    }

                    // 3. 查找表 ID 路由检查 (Lookup Table ID One-Hot)
                    // 验证当前指令关联了哪个具体的 Lookup Table。
                    // 使用 One-Hot 思想：虽然可能有 100 个表，但每条指令只会激活其中 1 个。
                    if let Some(table) = instruction.lookup_table() {
                        // 获取枚举索引，例如 ADD=5, SUB=6, AND=7...
                        let table_index = LookupTables::enum_index(&table);

                        // 加上对应的 Gamma 权重：γ^(2 + table_index)
                        // 这相当于在多项式层面声明：“我选了第 table_index 号桌子”。
                        // 只有正确的 Table Index 对应的项会被加上，其余表的项系数默认为 0。
                        lc += stage5_gammas[2 + table_index];
                    }

                    // 将计算出的唯一指纹写入 Stage 5 的结果向量
                    *o4 = lc;
                }
            });

        //  3. 将计算好的向量封装为 MultilinearPolynomial
        // 将原始数据向量转换为多线性多项式对象返回
        vals.map(MultilinearPolynomial::from)
    }

    /// 计算 Stage 1 (Spartan Outer Sumcheck) 的聚合声明值 (RV Claim)。
    ///
    /// 该函数负责将 Spartan Outer 阶段产生的多个独立多项式评估值（Claims），
    /// 使用随机挑战数 `gamma` 的幂次进行“批处理”压缩，合并为一个单一的标量值。
    ///
    /// 这里的 Claim 代表了 Prover 声称在某些随机点上，指令的各个组成部分（PC、立即数、标志位）的值。
    ///
    /// 数学公式：
    /// Rv_1 = (Claim_{PC} * γ^0) + (Claim_{Imm} * γ^1) + Σ (Claim_{Flag_i} * γ^{2+i})
    fn compute_rv_claim_1(
        opening_accumulator: &dyn OpeningAccumulator<F>, // 存储上一步 Sumcheck 结果的累加器
        gamma_powers: &[F],                              // 随机挑战数 γ 的幂次序列 [1, γ, γ^2, ...]
    ) -> F {
        // 1. 获取 UnexpandedPC (未扩展的程序计数器) 的 Claim
        // 这代表了 SpartanOuter 阶段对指令 PC 值的承诺评估
        let (_, unexpanded_pc_claim) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::UnexpandedPC,
            SumcheckId::SpartanOuter,
        );

        // 2. 获取 Immediate (立即数) 的 Claim
        // 这代表了 SpartanOuter 阶段对指令立即数部分的承诺评估
        let (_, imm_claim) = opening_accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::Imm, SumcheckId::SpartanOuter);

        // 3. 获取所有 Circuit Flags (电路控制标志) 的 Claims
        // 遍历所有定义的电路标志（例如是否挑战 Jump、Branch 等），
        // 提取它们在 SpartanOuter 阶段的评估值。
        let circuit_flag_claims: Vec<F> = CircuitFlags::iter()
            .map(|flag| {
                opening_accumulator
                    .get_virtual_polynomial_opening(
                        VirtualPolynomial::OpFlags(flag),
                        SumcheckId::SpartanOuter,
                    )
                    .1
            })
            .collect();

        // 4. 计算随机线性组合 (RLC - Random Linear Combination)
        // 将所有提取出的 Claims 按照特定的顺序链接起来：
        // [UnexpandedPC, Imm, Flag_0, Flag_1, ..., Flag_N]
        // 然后与对应的 gamma_powers 进行点积运算 (claim * gamma)，最后求和。
        // 这确保了如果 Prover 在任意一个分量上作弊，RLC 的结果都会以极大概率不匹配。
        std::iter::once(unexpanded_pc_claim)
            .chain(std::iter::once(imm_claim))
            .chain(circuit_flag_claims)
            .zip_eq(gamma_powers)
            .map(|(claim, gamma)| claim * gamma)
            .sum()
    }

    /// 计算 Stage 2 (Spartan Product Virtualization) 的聚合声明值 (RV Claim)。
    ///
    /// 此函数负责验证与“指令乘法化虚拟化”相关的逻辑，主要关注控制流（Jump/Branch）
    /// 和寄存器写入条件。
    ///
    /// 它从 Opening Accumulator 中提取 Spartan Product Virtualization 阶段生成的 4 个多项式评估值，
    /// 并使用随机挑战数 `gamma` 进行线性组合（RLC）。
    ///
    /// 聚合公式：
    /// Rv_2 = (Claim_{Jump} * γ^0)
    ///      + (Claim_{Branch} * γ^1)
    ///      + (Claim_{IsRdNotZero} * γ^2)
    ///      + (Claim_{WriteLookupToRD} * γ^3)
    fn compute_rv_claim_2(
        opening_accumulator: &dyn OpeningAccumulator<F>,
        gamma_powers: &[F],
    ) -> F {
        // 1. 获取 Jump 标志的 Claim
        // 验证当前指令是否为无条件跳转指令（JAL/JALR）。
        let (_, jump_claim) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::OpFlags(CircuitFlags::Jump),
            SumcheckId::SpartanProductVirtualization,
        );

        // 2. 获取 Branch 标志的 Claim
        // 验证当前指令是否为条件分支指令（BEQ, BNE, BLT 等）。
        let (_, branch_claim) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::InstructionFlags(InstructionFlags::Branch),
            SumcheckId::SpartanProductVirtualization,
        );

        // 3. 获取 IsRdNotZero 标志的 Claim
        // 验证目标寄存器索引 (rd) 是否非零（RISC-V 中 x0 恒为 0，写入无效）。
        // 这通常用于判断是否需要执行寄存器写入操作。
        let (_, rd_wa_claim) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::InstructionFlags(InstructionFlags::IsRdNotZero),
            SumcheckId::SpartanProductVirtualization,
        );

        // 4. 获取 WriteLookupOutputToRD 标志的 Claim
        // 验证当前指令是否需要将查找表（Lookup Table）的结果写入目标寄存器。
        let (_, write_lookup_output_to_rd_flag_claim) = opening_accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::OpFlags(CircuitFlags::WriteLookupOutputToRD),
                SumcheckId::SpartanProductVirtualization,
            );

        // 5. 计算加权和 (RLC)
        // 将上述 4 个独立的 Claim 使用 Gamma 幂次进行压缩，生成单一的标量值以便于后续验证。
        [
            jump_claim,
            branch_claim,
            rd_wa_claim,
            write_lookup_output_to_rd_flag_claim,
        ]
            .into_iter()
            .zip_eq(gamma_powers)
            .map(|(claim, gamma)| claim * gamma)
            .sum()
    }

    /// 计算 Stage 3 (Instruction Input / Shift) 的聚合声明值 (RV Claim)。
    ///
    /// 此函数负责验证指令操作数来源的逻辑（操作数是来自寄存器、立即数还是 PC），
    /// 以及与指令序列化（虚拟指令拆分）相关的标志。
    ///
    /// 聚合公式：
    /// Rv_3 = (Claim_{Imm} * γ^0) + (Claim_{PC} * γ^1)
    ///      + (Claim_{L_is_rs1} * γ^2) + (Claim_{L_is_pc} * γ^3)
    ///      + (Claim_{R_is_rs2} * γ^4) + (Claim_{R_is_imm} * γ^5)
    ///      + (Claim_{IsNoop} * γ^6) + (Claim_{Virtual} * γ^7) + (Claim_{FirstInSeq} * γ^8)
    fn compute_rv_claim_3(
        opening_accumulator: &dyn OpeningAccumulator<F>,
        gamma_powers: &[F],
    ) -> F {
        // 1. 获取立即数 (Imm) 的 Claim
        // 验证指令中立即数部分的正确性。
        let (_, imm_claim) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::Imm,
            SumcheckId::InstructionInputVirtualization,
        );

        // 2. 获取并校验程序计数器 (PC) 的 Claim
        // 这里有一个一致性检查：PC 值在 "SpartanShift" 阶段和 "InstructionInputVirtualization" 阶段
        // 应该是相同的。这是连接不同子协议的关键约束。
        let (_, spartan_shift_unexpanded_pc_claim) = opening_accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::UnexpandedPC,
                SumcheckId::SpartanShift,
            );
        let (_, instruction_input_unexpanded_pc_claim) = opening_accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::UnexpandedPC,
                SumcheckId::InstructionInputVirtualization,
            );

        // 确保跨子协议的一致性
        assert_eq!(
            spartan_shift_unexpanded_pc_claim,
            instruction_input_unexpanded_pc_claim
        );
        let unexpanded_pc_claim = spartan_shift_unexpanded_pc_claim;

        // 3. 验证左操作数 (Left Operand) 的来源
        // left_is_rs1_claim: 标志位，为 1 表示左操作数来自 rs1 寄存器。
        // left_is_pc_claim: 标志位，为 1 表示左操作数来自 PC（例如 AUIPC 指令）。
        let (_, left_is_rs1_claim) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::InstructionFlags(InstructionFlags::LeftOperandIsRs1Value),
            SumcheckId::InstructionInputVirtualization,
        );
        let (_, left_is_pc_claim) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::InstructionFlags(InstructionFlags::LeftOperandIsPC),
            SumcheckId::InstructionInputVirtualization,
        );

        // 4. 验证右操作数 (Right Operand) 的来源
        // right_is_rs2_claim: 标志位，为 1 表示右操作数来自 rs2 寄存器。
        // right_is_imm_claim: 标志位，为 1 表示右操作数来自立即数。
        let (_, right_is_rs2_claim) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::InstructionFlags(InstructionFlags::RightOperandIsRs2Value),
            SumcheckId::InstructionInputVirtualization,
        );
        let (_, right_is_imm_claim) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::InstructionFlags(InstructionFlags::RightOperandIsImm),
            SumcheckId::InstructionInputVirtualization,
        );

        // 5. 验证指令状态标志
        // is_noop_claim: 标志位，为 1 表示该指令是填充的空操作（padding）。
        // is_virtual_claim: 标志位，为 1 表示该指令是 Jolt 内部的虚拟宏指令。
        // is_first_in_sequence_claim: 标志位，为 1 表示这是多步指令序列的第一步。
        let (_, is_noop_claim) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::InstructionFlags(InstructionFlags::IsNoop),
            SumcheckId::SpartanShift,
        );
        let (_, is_virtual_claim) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::OpFlags(CircuitFlags::VirtualInstruction),
            SumcheckId::SpartanShift,
        );
        let (_, is_first_in_sequence_claim) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::OpFlags(CircuitFlags::IsFirstInSequence),
            SumcheckId::SpartanShift,
        );

        // 6. 计算加权和 (RLC)
        // 将所有相关的标志位和数值 claim 组合成一个标量。
        [
            imm_claim,
            unexpanded_pc_claim,
            left_is_rs1_claim,
            left_is_pc_claim,
            right_is_rs2_claim,
            right_is_imm_claim,
            is_noop_claim,
            is_virtual_claim,
            is_first_in_sequence_claim,
        ]
            .into_iter()
            .zip_eq(gamma_powers)
            .map(|(claim, gamma)| claim * gamma)
            .sum()
    }

    /// 计算 Stage 4 (Registers Read/Write Checking) 的聚合声明值 (RV Claim)。
    ///
    /// 此函数负责验证指令所访问的通用寄存器索引（Register Index）是否正确。
    /// 它从 "寄存器读写检查" 子协议中提取对 Rd (写目标), Rs1 (读源1), Rs2 (读源2) 的评估。
    ///
    /// 聚合公式：
    /// Rv_4 = (Claim_{Rd_Address} * γ^0)
    ///      + (Claim_{Rs1_Address} * γ^1)
    ///      + (Claim_{Rs2_Address} * γ^2)
    fn compute_rv_claim_4(
        opening_accumulator: &dyn OpeningAccumulator<F>,
        gamma_powers: &[F],
    ) -> F {
        // 遍历三个寄存器操作数类型：Rd (Write Address), Rs1 (Read Address 1), Rs2 (Read Address 2)
        std::iter::empty()
            .chain(once(VirtualPolynomial::RdWa))
            .chain(once(VirtualPolynomial::Rs1Ra))
            .chain(once(VirtualPolynomial::Rs2Ra))
            .map(|vp| {
                // 从 OpeningAccumulator 获取对应的评估值。
                // 这些值实际上是 "Flag(指令是否有该操作数) * Index(寄存器号)" 的某种编码形式的承诺。
                opening_accumulator
                    .get_virtual_polynomial_opening(vp, SumcheckId::RegistersReadWriteChecking)
                    .1
            })
            .zip(gamma_powers)
            .map(|(claim, gamma)| claim * gamma)
            .sum::<F>()
    }

    /// 计算 Stage 5 (Registers Val Evaluation & Instruction Lookups) 的聚合声明值 (RV Claim)。
    ///
    /// 此函数负责验证指令执行结果的写入逻辑（写哪个寄存器）以及指令所使用的查找表类型。
    /// 它从两个不同的子协议（RegistersValEvaluation 和 InstructionReadRaf）中提取评估值，
    /// 并将其合并。
    ///
    /// 聚合公式：
    /// Rv_5 = (Claim_{Rd_Address} * γ^0)
    ///      + (Claim_{Raf_Flag} * γ^1)
    ///      + Σ (Claim_{LookupTableFlag_i} * γ^{2+i})
    fn compute_rv_claim_5(
        opening_accumulator: &dyn OpeningAccumulator<F>,
        gamma_powers: &[F],
    ) -> F {
        // 1. 获取目标寄存器写地址 (Rd Write Address) 的 Claim
        // 验证指令打算写入的寄存器索引是否正确。
        // 这里的 SumcheckId 是 RegistersValEvaluation。
        let (_, rd_wa_claim) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RdWa,
            SumcheckId::RegistersValEvaluation,
        );

        // 2. 获取 RAF (Read Access Frequency) 标志的 Claim
        // 根据 compute_val_polys 中的逻辑，这个标志位对应 `!is_interleaved_operands`。
        // 它用于区分普通的指令操作数读取和其他类型的内存访问模式。
        // 这里的 SumcheckId 是 InstructionReadRaf。
        let (_, raf_flag_claim) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::InstructionRafFlag,
            SumcheckId::InstructionReadRaf,
        );

        // 初始化 RLC 和，包含 Rd 和 RAF 标志的贡献
        let mut sum = rd_wa_claim * gamma_powers[0];
        sum += raf_flag_claim * gamma_powers[1];

        // 3. 获取所有查找表标志 (Lookup Table Flags) 的 Claims
        // 遍历所有可能的查找表（例如 AND, ADD, XOR 等），验证当前指令使用了哪一个表。
        // 只有当指令实际执行了某种查找操作时，对应的标志位才为 1。
        for i in 0..LookupTables::<XLEN>::COUNT {
            let (_, claim) = opening_accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::LookupTableFlag(i),
                SumcheckId::InstructionReadRaf,
            );
            // 累加每个表的贡献：Flag_i * γ^{2+i}
            sum += claim * gamma_powers[2 + i];
        }

        sum
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for BytecodeReadRafSumcheckParams<F> {
    fn degree(&self) -> usize {
        self.d + 1
    }

    fn num_rounds(&self) -> usize {
        self.log_K + self.log_T
    }

    fn input_claim(&self, _: &dyn OpeningAccumulator<F>) -> F {
        self.input_claim
    }

    fn normalize_opening_point(
        &self,
        sumcheck_challenges: &[<F as JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        let mut r = sumcheck_challenges.to_vec();
        r[0..self.log_K].reverse();
        r[self.log_K..].reverse();
        OpeningPoint::new(r)
    }
}