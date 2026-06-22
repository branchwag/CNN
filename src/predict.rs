//! Inference mode: load the trained model from `artifacts/` and classify a
//! single test image, rendering it as ASCII art so you can eyeball the picture
//! and the prediction together — no retraining required.

use crate::{
    data::{FashionMnistBatcher, FashionMnistDataset, CLASS_NAMES, HEIGHT, WIDTH},
    model::Model,
    training::TrainingConfig,
};
use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    prelude::*,
    record::CompactRecorder,
};

pub fn run<B: Backend>(artifact_dir: &str, device: B::Device, index: usize) -> Result<(), String> {
    // Rebuild the architecture from the saved config, then load the weights.
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json")).map_err(|_| {
        format!("No trained model found in '{artifact_dir}/'. Run option 1 to train first.")
    })?;

    let model: Model<B> = config
        .model
        .init::<B>(&device)
        .load_file(format!("{artifact_dir}/model"), &CompactRecorder::new(), &device)
        .map_err(|e| format!("Failed to load model weights: {e}"))?;

    let dataset = FashionMnistDataset::test()?;
    let item = dataset.get(index).ok_or_else(|| {
        format!("Index {index} is out of range (test set has {} images).", dataset.len())
    })?;

    print_image(&item.image);

    let batch = FashionMnistBatcher.batch(vec![item.clone()], &device);
    let predicted = model
        .forward(batch.images)
        .argmax(1)
        .into_data()
        .iter::<i64>()
        .next()
        .expect("model produced no output") as usize;

    let truth = item.label as usize;
    println!("Prediction: {}", CLASS_NAMES[predicted]);
    println!("Truth:      {}", CLASS_NAMES[truth]);
    println!(
        "Result:     {}",
        if predicted == truth { "correct" } else { "wrong" }
    );
    Ok(())
}

/// Renders a 28x28 grayscale image as ASCII art. Each pixel becomes two
/// characters wide so the picture isn't squished by the terminal's cell ratio.
fn print_image(image: &[[f32; WIDTH]; HEIGHT]) {
    // Darkest -> lightest. Pixels are stored 0 (black background) to 255 (white).
    const RAMP: &[u8] = b" .:-=+*#%@";

    println!();
    for row in image {
        let mut line = String::with_capacity(WIDTH * 2);
        for &pixel in row {
            let level = (pixel / 255.0 * (RAMP.len() - 1) as f32).round() as usize;
            let ch = RAMP[level] as char;
            line.push(ch);
            line.push(ch);
        }
        println!("{line}");
    }
    println!();
}
