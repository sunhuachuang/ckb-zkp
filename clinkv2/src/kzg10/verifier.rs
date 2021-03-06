use ark_ec::PairingEngine;
use ark_ff::{Field, One, ToBytes, Zero};
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain, Polynomial};
use merlin::Transcript;

use crate::{
    kzg10::{Proof, VerifyAssignment, VerifyKey, KZG10},
    r1cs::{Index, SynthesisError},
    Vec,
};

pub fn verify_proof<E: PairingEngine>(
    circuit: &VerifyAssignment<E>,
    kzg10_vk: &VerifyKey<E>,
    proof: &Proof<E>,
    io: &Vec<Vec<E::Fr>>,
) -> Result<bool, SynthesisError> {
    let mut transcript = Transcript::new(b"CLINKv2");
    let zero = E::Fr::zero();
    let one = E::Fr::one();
    let m_abc = circuit.at.len();
    let m_io = io.len();
    let m_mid = proof.r_mid_comms.len();
    let n = io[0].len();

    let mut r_mid_comms_bytes = vec![];
    proof.r_mid_comms.write(&mut r_mid_comms_bytes)?;
    transcript.append_message(b"witness polynomial commitments", &r_mid_comms_bytes);

    let mut c = [0u8; 31];
    transcript.challenge_bytes(b"batching challenge", &mut c);
    let eta = E::Fr::from_random_bytes(&c).unwrap();

    let mut q_comm_bytes = vec![];
    proof.q_comm.write(&mut q_comm_bytes)?;
    transcript.append_message(b"quotient polynomial commitments", &q_comm_bytes);

    c = [0u8; 31];
    transcript.challenge_bytes(b"random point", &mut c);
    let zeta = E::Fr::from_random_bytes(&c).unwrap();

    let r_mid_q_comms = [&proof.r_mid_comms, &[proof.q_comm][..]].concat();

    assert!(KZG10::<E>::batch_check(
        &kzg10_vk,
        &r_mid_q_comms,
        zeta,
        &proof.r_mid_q_values,
        &proof.r_mid_q_proof,
        proof.opening_challenge
    )?);

    let domain: GeneralEvaluationDomain<E::Fr> =
        EvaluationDomain::<E::Fr>::new(n).ok_or(SynthesisError::PolynomialDegreeTooLarge)?;

    //let domain_size = domain.size();

    let mut r_io_values = vec![];
    let lag_values = domain.evaluate_all_lagrange_coefficients(zeta);
    for j in 0..m_io {
        let mut rj_value = zero;
        for i in 0..io[j].len() {
            rj_value += &(lag_values[i] * &io[j][i]);
        }
        r_io_values.push(rj_value);
    }

    let vanishing_poly = domain.vanishing_polynomial();
    let vanishing_value = vanishing_poly.evaluate(&zeta);

    let mut ab_c = zero;
    let mut eta_i = one;

    for i in 0..m_abc {
        let mut ai = zero;
        for (coeff, index) in (&circuit.at[i]).into_iter() {
            match index {
                Index::Input(j) => ai += &(r_io_values[*j] * coeff),
                Index::Aux(j) => ai += &(proof.r_mid_q_values[*j] * coeff),
            }
        }

        let mut bi = zero;
        for (coeff, index) in (&circuit.bt[i]).into_iter() {
            match index {
                Index::Input(j) => bi += &(r_io_values[*j] * coeff),
                Index::Aux(j) => bi += &(proof.r_mid_q_values[*j] * coeff),
            }
        }

        let mut ci = zero;
        for (coeff, index) in (&circuit.ct[i]).into_iter() {
            match index {
                Index::Input(j) => ci += &(r_io_values[*j] * coeff),
                Index::Aux(j) => ci += &(proof.r_mid_q_values[*j] * coeff),
            }
        }

        ab_c += &(eta_i * &(ai * &bi - &ci));
        eta_i = eta_i * &eta;
    }
    assert_eq!(ab_c, proof.r_mid_q_values[m_mid] * &vanishing_value);

    Ok(true)
}
