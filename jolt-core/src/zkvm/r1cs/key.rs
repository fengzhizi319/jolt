use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use sha3::{Digest, Sha3_256};

use crate::{
    field::JoltField, poly::lagrange_poly::LagrangePolynomial, zkvm::r1cs::inputs::NUM_R1CS_INPUTS,
};

use super::constraints::{
    OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE, R1CS_CONSTRAINTS, R1CS_CONSTRAINTS_FIRST_GROUP,
    R1CS_CONSTRAINTS_SECOND_GROUP,
};
use crate::utils::math::Math;
use crate::zkvm::r1cs::inputs::JoltR1CSInputs;

///UniformSpartanKey 是 Jolt 中用于 Uniform Spartan 证明系统的核心结构体，
/// 充当了证明者（Prover）和验证者（Verifier）共享的“索引”或“键（Key）”。
///它的主要作用是描述和唯一标识整个计算电路的结构（即 VM 执行了多少步，每步有哪些约束），
/// 而不需要显式地存储巨大的约束矩阵。
#[derive(Clone, Copy, CanonicalSerialize, CanonicalDeserialize)]
pub struct UniformSpartanKey<F: JoltField> {
    /// Number of constraints across all steps padded to nearest power of 2
    ///  填充到最近2的幂次的所有步骤的总约束数
    pub num_cons_total: usize,

    /// Number of steps padded to the nearest power of 2
    /// 填充到最近2的幂次的执行步骤数 (Number of steps)
    pub num_steps: usize,

    /// Digest of verifier key
    /// 验证密钥的摘要 (Digest)
    pub(crate) vk_digest: F,
}

impl<F: JoltField> UniformSpartanKey<F> {
    pub fn new(num_steps: usize) -> Self {
        assert!(num_steps.is_power_of_two());
        let rows_per_step_padded = Self::num_rows_per_step().next_power_of_two();
        let total_rows = (num_steps * rows_per_step_padded).next_power_of_two();
        let vk_digest = Self::digest(num_steps);
        Self {
            num_cons_total: total_rows,
            num_steps,
            vk_digest,
        }
    }

    #[inline]
    fn num_vars() -> usize {
        JoltR1CSInputs::num_inputs()
    }

    #[inline]
    fn num_rows_per_step() -> usize {
        R1CS_CONSTRAINTS.len()
    }

    pub fn num_vars_uniform_padded(&self) -> usize {
        Self::num_vars().next_power_of_two()
    }

    /// Number of cycle variables, e.g. number of bits needed to represent all cycles in the trace
    pub fn num_cycle_vars(&self) -> usize {
        self.num_steps.next_power_of_two().log_2()
    }

    /// Number of bits needed for all rows.
    /// With univariate skip, this is the number of cycle variables plus two (one for univariate skip of degree ~13-15, and one for the streaming round)
    pub fn num_rows_bits(&self) -> usize {
        self.num_cycle_vars() + 2
    }

    /// Evaluates `sum_y A(rx_constr, y)*z(y) * sum_y B(rx_constr, y)*z(y)`.
    ///
    /// Note `rx_constr` is the randomness used to bind the rows of `A` and `B`.
    /// `r1cs_input_evals` should ordered as per [`ALL_R1CS_INPUTS`].
    ///
    /// [`ALL_R1CS_INPUTS`]: crate::zkvm::r1cs::inputs::ALL_R1CS_INPUTS
    pub fn evaluate_inner_sum_product_at_point(
        &self,
        rx_constr: &[F::Challenge],
        r1cs_input_evals: [F; NUM_R1CS_INPUTS],
    ) -> F {
        // Row axis: r_constr = [r_stream, r0]; use Lagrange basis for first-round
        // (half the number of R1CS constraints)
        // and linear blend for the two groups using r_stream
        debug_assert!(rx_constr.len() >= 2);
        let r_stream = rx_constr[0];
        let r0 = rx_constr[1];

        // Lagrange weights over the univariate-skip base domain at r0
        let w =
            LagrangePolynomial::<F>::evals::<F::Challenge, OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE>(&r0);

        // Build z(r_cycle) vector with a trailing 1 for the constant column
        let z_const_col = JoltR1CSInputs::num_inputs();
        let mut z = r1cs_input_evals.to_vec();
        z.push(F::one());

        // Group 0 fused Az,Bz via dot product of LC with z(r_cycle)
        let mut az_g0 = F::zero();
        let mut bz_g0 = F::zero();
        for i in 0..R1CS_CONSTRAINTS_FIRST_GROUP.len() {
            let lc_a = &R1CS_CONSTRAINTS_FIRST_GROUP[i].cons.a;
            let lc_b = &R1CS_CONSTRAINTS_FIRST_GROUP[i].cons.b;
            az_g0 += w[i] * lc_a.dot_product::<F>(&z, z_const_col);
            bz_g0 += w[i] * lc_b.dot_product::<F>(&z, z_const_col);
        }

        // Group 1 fused Az,Bz (use same Lagrange weights order as construction)
        let mut az_g1 = F::zero();
        let mut bz_g1 = F::zero();
        let g2_len = core::cmp::min(
            R1CS_CONSTRAINTS_SECOND_GROUP.len(),
            OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE,
        );
        for i in 0..g2_len {
            let lc_a = &R1CS_CONSTRAINTS_SECOND_GROUP[i].cons.a;
            let lc_b = &R1CS_CONSTRAINTS_SECOND_GROUP[i].cons.b;
            az_g1 += w[i] * lc_a.dot_product::<F>(&z, z_const_col);
            bz_g1 += w[i] * lc_b.dot_product::<F>(&z, z_const_col);
        }

        // Bind by r_stream to match the outer streaming combination used for final Az,Bz
        let az_final = az_g0 + r_stream * (az_g1 - az_g0);
        let bz_final = bz_g0 + r_stream * (bz_g1 - bz_g0);

        az_final * bz_final
    }

    /// Returns the digest of the R1CS "shape" derived from compile-time constants
    /// Canonical serialization of constants:
    /// - domain tag
    /// - num_steps (u64 BE)
    /// - num_vars (u32 BE)
    /// - for each row in R1CS_CONSTRAINTS:
    ///   - tag 'A' | row.a terms (sorted by input_index asc) + const term
    ///   - tag 'B' | row.b terms (sorted by input_index asc) + const term
    fn digest(num_steps: usize) -> F {
        let mut bytes: Vec<u8> = Vec::new();
        bytes.extend_from_slice(b"JOLT_R1CS_CONSTRAINTS");
        bytes.extend_from_slice(&num_steps.to_be_bytes());

        let num_vars: u32 = JoltR1CSInputs::num_inputs() as u32;
        bytes.extend_from_slice(&num_vars.to_be_bytes());

        for row_named in R1CS_CONSTRAINTS.iter() {
            let row = &row_named.cons;
            row.a.serialize_canonical(b'A', &mut bytes);
            row.b.serialize_canonical(b'B', &mut bytes);
        }

        let mut hasher = Sha3_256::new();
        hasher.update(&bytes);

        let map_to_field = |digest: &[u8]| -> F {
            let bv = (0..250).map(|i| {
                let (byte_pos, bit_pos) = (i / 8, i % 8);
                let bit = (digest[byte_pos] >> bit_pos) & 1;
                bit == 1
            });

            // turn the bit vector into a scalar
            let mut digest = F::zero();
            let mut coeff = F::one();
            for bit in bv {
                if bit {
                    digest += coeff;
                }
                coeff += coeff;
            }
            digest
        };
        map_to_field(&hasher.finalize())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;

    #[test]
    fn test_digest_consistency() {
        let num_steps = 1024;
        let digest1 = UniformSpartanKey::<Fr>::digest(num_steps);
        let digest2 = UniformSpartanKey::<Fr>::digest(num_steps);
        assert_eq!(digest1, digest2, "Digest should be deterministic");
    }

    #[test]
    fn test_digest_different_steps() {
        let digest1 = UniformSpartanKey::<Fr>::digest(1024);
        let digest2 = UniformSpartanKey::<Fr>::digest(2048);
        assert_ne!(
            digest1, digest2,
            "Digest should differ for different num_steps"
        );
    }

    #[test]
    fn test_key_creation_sets_digest() {
        let num_steps = 1024;
        let key = UniformSpartanKey::<Fr>::new(num_steps);
        let expected_digest = UniformSpartanKey::<Fr>::digest(num_steps);
        assert_eq!(key.vk_digest, expected_digest);
    }
}
