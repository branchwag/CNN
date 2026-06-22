mod data;
mod model;
mod predict;
mod training;

use burn::backend::{ndarray::NdArrayDevice, Autodiff, NdArray};
use std::io::Write;

const ARTIFACT_DIR: &str = "artifacts";

fn main() {
    type MyBackend = NdArray<f32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = NdArrayDevice::Cpu;

    loop {
        println!("FashionMNIST CNN");
        println!("  1. Train a new model");
        println!("  2. Predict on a test image");

        match prompt("Choose an option [1/2]: ").trim() {
            "1" => {
                training::run::<MyAutodiffBackend>(ARTIFACT_DIR, device.clone());
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
                if let Err(e) = predict::run::<MyBackend>(ARTIFACT_DIR, device.clone(), index) {
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
