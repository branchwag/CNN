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

    println!("FashionMNIST CNN");
    println!("  1. Train a new model");
    println!("  2. Predict on a test image");

    match prompt("Choose an option [1/2]: ").trim() {
        "1" => {
            training::run::<MyAutodiffBackend>(ARTIFACT_DIR, device);
        }
        "2" => {
            // Inference doesn't need autodiff, so use the plain backend.
            let index = prompt("Test image index (0-9999): ")
                .trim()
                .parse()
                .unwrap_or(0);
            predict::run::<MyBackend>(ARTIFACT_DIR, device, index);
        }
        other => {
            eprintln!("'{other}' is not a valid option. Run again and pick 1 or 2.");
            std::process::exit(1);
        }
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
