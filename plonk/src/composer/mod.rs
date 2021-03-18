use ark_ff::FftField as Field;
use ark_std::vec::Vec;

use crate::Map;

mod permutation;
use permutation::Permutation;

mod arithmetic;

mod synthesize;
pub use synthesize::{Error, Selectors, Witnesses};

#[derive(Debug, Eq, PartialEq, Clone, Copy, Hash)]
pub struct Variable(usize);

#[derive(Debug)]
pub struct Composer<F: Field> {
    n: usize,

    q_0: Vec<F>,
    q_1: Vec<F>,
    q_2: Vec<F>,
    q_3: Vec<F>,
    q_m: Vec<F>,
    q_c: Vec<F>,

    q_arith: Vec<F>,

    pi: Vec<F>,

    w_0: Vec<Variable>,
    w_1: Vec<Variable>,
    w_2: Vec<Variable>,
    w_3: Vec<Variable>,

    null_var: Variable,
    permutation: Permutation<F>,
    assignment: Map<Variable, F>,
}

impl<F: Field> Composer<F> {
    pub fn new() -> Self {
        let mut cs = Composer {
            n: 0,

            q_0: Vec::new(),
            q_1: Vec::new(),
            q_2: Vec::new(),
            q_3: Vec::new(),
            q_m: Vec::new(),
            q_c: Vec::new(),
            pi: Vec::new(),

            q_arith: Vec::new(),

            w_0: Vec::new(),
            w_1: Vec::new(),
            w_2: Vec::new(),
            w_3: Vec::new(),

            null_var: Variable(0),
            permutation: Permutation::new(),
            assignment: Map::new(),
        };
        cs.null_var = cs.alloc_and_assign(F::zero());

        cs
    }

    pub fn size(&self) -> usize {
        self.n
    }

    pub fn alloc_and_assign(&mut self, value: F) -> Variable {
        let var = self.permutation.alloc();
        self.assignment.insert(var, value);

        var
    }
}

#[cfg(test)]
mod test {
    use ark_bls12_381::Fr;
    use ark_ff::{One, Zero};

    use super::*;
    use crate::utils::pad_to_size;

    #[test]
    fn preprocess() {
        let ks = [
            Fr::one(),
            Fr::from(7_u64),
            Fr::from(13_u64),
            Fr::from(17_u64),
        ];
        let mut cs = Composer::new();
        let one = Fr::one();
        let two = one + one;
        let three = two + one;
        let four = two + two;
        let var_one = cs.alloc_and_assign(one);
        let var_two = cs.alloc_and_assign(two);
        let var_three = cs.alloc_and_assign(three);
        let var_four = cs.alloc_and_assign(four);
        cs.create_add_gate(
            (var_three, one),
            (var_one, one),
            var_four,
            None,
            Fr::zero(),
            Fr::zero(),
        );
        cs.create_add_gate(
            (var_two, one),
            (var_two, one),
            var_four,
            None,
            Fr::zero(),
            Fr::zero(),
        );
        cs.create_mul_gate(
            var_one,
            var_two,
            var_two,
            None,
            Fr::one(),
            Fr::zero(),
            Fr::zero(),
        );
        let s = cs.process(&ks).unwrap();
        let pi = pad_to_size(cs.public_inputs(), s.size());

        let witnesses = cs.synthesize().unwrap();
        let Witnesses { w_0, w_1, w_2, w_3 } = witnesses;
        assert_eq!(w_0.len(), s.q_0.len());
        (0..s.size()).into_iter().for_each(|i| {
            assert_eq!(
                Fr::zero(),
                w_0[i] * s.q_0[i]
                    + w_1[i] * s.q_1[i]
                    + w_2[i] * s.q_2[i]
                    + w_3[i] * s.q_3[i]
                    + w_1[i] * w_2[i] * s.q_m[i]
                    + s.q_c[i]
                    + pi[i]
            )
        });
    }
}
