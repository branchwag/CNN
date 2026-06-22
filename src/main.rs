mod data;
mod model;
mod predict;
mod training;

use burn::backend::{
    ndarray::{NdArray, NdArrayDevice},
    wgpu::{Wgpu, WgpuDevice},
    Autodiff,
};
use std::io::Write;

const ARTIFACT_DIR: &str = "artifacts";

fn main() {
    // Training uses the Wgpu backend, which auto-selects the best available GPU
    // (discrete > integrated > CPU software renderer). Inference stays on NdArray
    // since prediction is fast enough on CPU and avoids a GPU round-trip.
    type TrainBackend = Autodiff<Wgpu>;
    type InferBackend = NdArray<f32>;

    loop {
        println!("FashionMNIST CNN");
        println!("  1. Train a new model");
        println!("  2. Predict on a test image");

        match prompt("Choose an option [1/2]: ").trim() {
            "1" => {
                let device = WgpuDevice::DefaultDevice;
                println!("Training device: {device:?}");
                if let Err(e) = training::run::<TrainBackend>(ARTIFACT_DIR, device) {
                    eprintln!("{e}");
                    println!();
                    continue;
                }
            }
            "2" => {
                if !std::path::Path::new(&format!("{ARTIFACT_DIR}/config.json")).exists() {
                    eprintln!("No trained model found in '{ARTIFACT_DIR}/'. Run option 1 to train first.");
                    println!();
                    continue;
                }
                // Inference doesn't need autodiff, so use the plain backend.
                let index = prompt("Test image index (0-9999): ")
                    .trim()
                    .parse()
                    .unwrap_or(0);
                if let Err(e) = predict::run::<InferBackend>(ARTIFACT_DIR, NdArrayDevice::Cpu, index) {
                    eprintln!("{e}");
                    println!();
                    continue;
                }
            }
            other => {
                eprintln!("'{other}' is not a valid option.");
                println!();
                continue;
            }
        }
        break;
    }
}

/// Prints a prompt (without a trailing newline) and reads one line from stdin.
fn prompt(message: &str) -> String {
    print!("{message}");
    std::io::stdout().flush().expect("flush stdout");

    let mut input = String::new();
    std::io::stdin()
        .read_line(&mut input)
        .expect("read from stdin");
    input
}
