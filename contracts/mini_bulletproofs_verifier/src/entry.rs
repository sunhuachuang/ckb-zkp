use alloc::vec::Vec;
use core::result::Result;

use ckb_std::{ckb_constants::Source, high_level::load_cell_data};

use crate::error::Error;

use ckb_zkp::{
    bn_256::{Bn_256 as E, Fr},
    bulletproofs::{verify_proof, Generators, Proof, R1csCircuit},
    math::PrimeField,
    r1cs::{ConstraintSynthesizer, ConstraintSystem, SynthesisError},
};

struct Mini<F: PrimeField> {
    pub x: Option<F>,
    pub y: Option<F>,
    pub z: Option<F>,
    pub num: u32,
}

impl<F: PrimeField> ConstraintSynthesizer<F> for Mini<F> {
    fn generate_constraints<CS: ConstraintSystem<F>>(
        self,
        cs: &mut CS,
    ) -> Result<(), SynthesisError> {
        let var_x = cs.alloc(|| "x", || self.x.ok_or(SynthesisError::AssignmentMissing))?;

        let var_y = cs.alloc(|| "y", || self.y.ok_or(SynthesisError::AssignmentMissing))?;

        let var_z = cs.alloc_input(
            || "z(output)",
            || self.z.ok_or(SynthesisError::AssignmentMissing),
        )?;

        for _ in 0..self.num {
            cs.enforce(
                || "x * (y + 2) = z",
                |lc| lc + var_x,
                |lc| lc + var_y + (F::from(2u32), CS::one()),
                |lc| lc + var_z,
            );
        }

        Ok(())
    }
}

pub fn main() -> Result<(), Error> {
    // load verify key.
    let _vk_data = match load_cell_data(0, Source::Output) {
        Ok(data) => data,
        Err(err) => return Err(err.into()),
    };

    // load proof.
    let proof_data = match load_cell_data(1, Source::Output) {
        Ok(data) => data,
        Err(err) => return Err(err.into()),
    };

    // load public info.
    let public_data = match load_cell_data(2, Source::Output) {
        Ok(data) => data,
        Err(err) => return Err(err.into()),
    };

    let (gens, r1cs, proof): (Generators<E>, R1csCircuit<E>, Proof<E>) =
        postcard::from_bytes(&proof_data).map_err(|_e| Error::Encoding)?;
    let publics: Vec<Fr> = postcard::from_bytes(&public_data).map_err(|_e| Error::Encoding)?;

    // Demo circuit
    let _c = Mini::<Fr> {
        x: None,
        y: None,
        z: None,
        num: 10,
    };

    match verify_proof(&gens, &proof, &r1cs, &publics) {
        Ok(true) => Ok(()),
        _ => Err(Error::Verify),
    }
}
