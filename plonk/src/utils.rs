use ark_ff::{FftField as Field, Zero};
use ark_poly::{
    univariate::DensePolynomial, EvaluationDomain, Evaluations as EvaluationsOnDomain,
    UVPolynomial,
};
use ark_std::{cfg_iter, vec};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::data_structures::LabeledPolynomial;

pub fn scalar_mul<F: Field>(poly: &DensePolynomial<F>, scalar: &F) -> DensePolynomial<F> {
    if poly.is_zero() || scalar.is_zero() {
        return DensePolynomial::zero();
    }
    let coeffs: Vec<_> = cfg_iter!(poly.coeffs)
        .map(|coeff| *scalar * coeff)
        .collect();
    DensePolynomial::from_coefficients_vec(coeffs)
}

pub fn pad_to_size<F: Field>(v: &[F], expected_size: usize) -> Vec<F> {
    let diff = expected_size - v.len();
    let zeros = vec![F::zero(); diff];
    let mut v = v.to_vec();
    v.extend(zeros.iter());

    v
}

pub fn to_labeled<F: Field>(
    label: &str,
    poly: DensePolynomial<F>,
) -> LabeledPolynomial<F> {
    LabeledPolynomial::new(label.to_string(), poly, None, None)
}

pub fn generator<F: Field>(domain: impl EvaluationDomain<F>) -> F {
    domain.element(1)
}

pub fn vanishing_poly<F: Field>(domain: impl EvaluationDomain<F>) -> DensePolynomial<F> {
    let n = domain.size();
    let mut coeffs = vec![F::zero(); n + 1];
    coeffs[0] = F::one();
    coeffs[n] = -F::zero();
    DensePolynomial::from_coefficients_vec(coeffs)
}

pub fn last_lagrange<F: Field>(domain: impl EvaluationDomain<F>) -> DensePolynomial<F> {
    let n = domain.size();
    let mut evals = vec![F::zero(); n];
    evals[n - 1] = F::one();
    EvaluationsOnDomain::from_vec_and_domain(evals, domain).interpolate()
}

pub fn evaluate_last_lagrange<F: Field>(domain: impl EvaluationDomain<F>, zeta: F) -> F {
    let n = domain.size();
    let root = domain.element(n - 1);
    let c = F::from(n as u64).inverse().unwrap() * root;
    match (zeta - root).inverse() {
        None => F::one(),
        Some(denumerator) => c * denumerator * (zeta.pow(&[n as u64]) - F::one()),
    }
}

pub fn first_lagrange<F: Field>(domain: impl EvaluationDomain<F>) -> DensePolynomial<F> {
    let mut l = vec![F::zero(); domain.size()];
    l[0] = F::one();
    EvaluationsOnDomain::from_vec_and_domain(l, domain).interpolate()
}

pub fn evaluate_first_lagrange<F: Field>(domain: impl EvaluationDomain<F>, zeta: F) -> F {
    let n = domain.size();
    let c = F::from(n as u64).inverse().unwrap();

    match (zeta - F::one()).inverse() {
        None => F::one(),
        Some(denumerator) => c * denumerator * (zeta.pow(&[n as u64]) - F::one()),
    }
}

#[cfg(test)]
mod tests {
    use ark_bls12_381::Fr;
    use ark_ff::UniformRand;
    use ark_poly::{EvaluationDomain, GeneralEvaluationDomain, Polynomial};
    use ark_std::test_rng;

    use super::*;
    use crate::composer::Error;

    #[test]
    fn last_lagrange_evaluation() -> Result<(), Error> {
        let rng = &mut test_rng();
        let domain = GeneralEvaluationDomain::<Fr>::new(5)
            .ok_or(Error::PolynomialDegreeTooLarge)?;
        let ln = last_lagrange(domain);

        let root = domain.element(domain.size() - 1);
        assert!(ln.evaluate(&root) == evaluate_last_lagrange(domain, root));
        for _ in 0..10 {
            let zeta = Fr::rand(rng);
            assert!(ln.evaluate(&zeta) == evaluate_last_lagrange(domain, zeta));
        }

        Ok(())
    }

    #[test]
    fn first_lagrange_evaluation() -> Result<(), Error> {
        let rng = &mut test_rng();
        let domain = GeneralEvaluationDomain::<Fr>::new(5)
            .ok_or(Error::PolynomialDegreeTooLarge)?;
        let l1 = first_lagrange(domain);

        let root = domain.element(0);
        assert!(l1.evaluate(&root) == evaluate_first_lagrange(domain, root));

        for _ in 0..10 {
            let zeta = Fr::rand(rng);
            assert!(l1.evaluate(&zeta) == evaluate_first_lagrange(domain, zeta));
        }

        Ok(())
    }
}
