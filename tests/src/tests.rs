use super::*;
use ckb_testtool::{builtin::ALWAYS_SUCCESS, context::Context};
use ckb_tool::ckb_types::{
    bytes::Bytes,
    core::{TransactionBuilder, TransactionView},
    packed::*,
    prelude::*,
};
use std::fs::File;
use std::io::Read;
use std::time::Instant;

const BULLETPROOFS_CONTRACT_NAME: &str = "bulletproofs-verifier";
const GROTH16_CONTRACT_NAME: &str = "groth16-verifier";
const MAX_CYCLES: u64 = 1_000_000_000_000;

// Relative path starts from capsuled-contracts/tests.
const VK_DIR: &str = "../cli/trusted_setup";
const PROOF_DIR: &str = "../cli/proofs_files";

// Names of vk files and proof files.
const VK_BN_256: &str = "mini-groth16-bn_256.vk";
const PROOF_BN_256: &str = "mini.groth16-bn_256.proof";
const VK_BLS12_381: &str = "mini-groth16-bls12_381.vk";
const PROOF_BLS12_381: &str = "mini.groth16-bls12_381.proof";

const BULLETPROOFS_BN_256: &str = "mini.bulletproofs-bn_256.proof";
//const BULLETPROOFS_BN_256: &str = "mimc.bulletproofs-bn_256.proof"; // test for benchmark

#[test]
fn test_groth16_proof_bn_256() {
    proving_test(VK_BN_256, PROOF_BN_256, GROTH16_CONTRACT_NAME);
}

#[test]
fn test_proof_bls12_381() {
    proving_test(VK_BLS12_381, PROOF_BLS12_381, GROTH16_CONTRACT_NAME);
}

#[test]
fn test_bulletproofs_bn_256() {
    proving_test(VK_BN_256, BULLETPROOFS_BN_256, BULLETPROOFS_CONTRACT_NAME);
}

#[test]
fn test_no_proof() {
    let (mut context, tx) = build_test_context(Bytes::new(), Bytes::new(), GROTH16_CONTRACT_NAME);

    let tx = context.complete_tx(tx);
    context
        .verify_tx(&tx, MAX_CYCLES)
        .expect_err("should not pass verification");
}

fn build_test_context(proof_file: Bytes, vk: Bytes, contract: &str) -> (Context, TransactionView) {
    // deploy contract.
    let mut context = Context::default();
    let contract_bin: Bytes = Loader::default().load_binary(contract);
    let contract_out_point = context.deploy_contract(contract_bin);
    // Deploy always_success script as lock script.
    let always_success_out_point = context.deploy_contract(ALWAYS_SUCCESS.clone());

    // Build LOCK script using always_success script.
    let lock_script = context
        .build_script(&always_success_out_point, Default::default())
        .expect("build lock script");
    let lock_script_dep = CellDep::new_builder()
        .out_point(always_success_out_point)
        .build();

    // Build TYPE script using the ckb-zkp contract
    let type_script = context
        .build_script(&contract_out_point, vk)
        .expect("build type script");
    let type_script_dep = CellDep::new_builder().out_point(contract_out_point).build();

    // prepare cells
    let input_out_point = context.create_cell(
        CellOutput::new_builder()
            .capacity(1000u64.pack())
            .lock(lock_script.clone())
            .build(),
        Bytes::new(),
    );
    let input = CellInput::new_builder()
        .previous_output(input_out_point)
        .build();
    let outputs = vec![
        CellOutput::new_builder()
            .capacity(500u64.pack())
            .lock(lock_script.clone())
            .type_(Some(type_script).pack())
            .build(),
        CellOutput::new_builder()
            .capacity(500u64.pack())
            .lock(lock_script)
            .build(),
    ];

    let outputs_data = vec![proof_file, Bytes::new()];

    // build transaction
    let tx = TransactionBuilder::default()
        .input(input)
        .outputs(outputs)
        .outputs_data(outputs_data.pack())
        .cell_dep(lock_script_dep)
        .cell_dep(type_script_dep)
        .build();
    (context, tx)
}

fn proving_test(vk_file: &str, proof_file_s: &str, contract: &str) {
    let mut proof_file =
        File::open(format!("{}/{}", PROOF_DIR, proof_file_s)).expect("proof file not exists");
    let mut proof_bin = Vec::new();
    // read the whole file
    proof_file
        .read_to_end(&mut proof_bin)
        .expect("Failed to read proof file");
    println!("proof file size: {}", proof_bin.len());

    let mut vk_file = File::open(format!("{}/{}", VK_DIR, vk_file)).expect("VK file not exists");
    let mut vk_bin = Vec::new();
    vk_file
        .read_to_end(&mut vk_bin)
        .expect("Failed to read VK file");

    let (mut context, tx) = build_test_context(proof_bin.into(), vk_bin.into(), contract);

    let tx = context.complete_tx(tx);

    let start = Instant::now();
    match context.verify_tx(&tx, MAX_CYCLES) {
        Ok(cycles) => {
            println!("cycles: {}", cycles);
        }
        Err(err) => panic!("Failed to pass test: {}", err),
    }
    println!("Verify {} Time: {:?}", proof_file_s, start.elapsed());
}
