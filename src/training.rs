//! Training loop using burn's high-level `Learner` (metrics, checkpointing, and
//! the live dashboard come for free). Equivalent to the notebook's manual
//! `train_step` / `test_step` over 3 epochs with SGD (lr=0.1).

use crate::{
    data::{FashionMnistBatcher, FashionMnistDataset, CLASS_NAMES},
    model::{Model, ModelConfig},
};
use burn::{
    data::{
        dataloader::{batcher::Batcher, DataLoaderBuilder},
        dataset::Dataset,
    },
    optim::SgdConfig,
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{
        metric::{AccuracyMetric, LossMetric},
        Learner, SupervisedTraining,
    },
};

#[derive(Config, Debug)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: SgdConfig,
    #[config(default = 3)]
    pub num_epochs: usize,
    #[config(default = 32)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 0.1)]
    pub learning_rate: f64,
}

pub fn run<B: AutodiffBackend>(artifact_dir: &str, device: B::Device) {
    let config = TrainingConfig::new(ModelConfig::new(), SgdConfig::new());

    std::fs::create_dir_all(artifact_dir).ok();
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("save training config");

    B::seed(&device, config.seed);

    // The training split runs on the autodiff backend `B`; the validation split
    // runs on its inner (non-autodiff) backend, so gradients aren't tracked.
    let dataloader_train = DataLoaderBuilder::<B, _, _>::new(FashionMnistBatcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(FashionMnistDataset::train());

    let dataloader_test = DataLoaderBuilder::<B::InnerBackend, _, _>::new(FashionMnistBatcher)
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .build(FashionMnistDataset::test());

    let learner = Learner::new(
        config.model.init::<B>(&device),
        config.optimizer.init(),
        config.learning_rate,
    );

    let result = SupervisedTraining::new(artifact_dir, dataloader_train, dataloader_test)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(config.num_epochs)
        .summary()
        .launch(learner);

    print_sample_predictions(&result.model);

    result
        .model
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("save trained model");
}

/// Mirrors the notebook's final cells: run the trained model over a handful of
/// test images and print predicted vs. true class for each.
fn print_sample_predictions<B: Backend>(model: &Model<B>) {
    let device = model
        .devices()
        .into_iter()
        .next()
        .expect("model should live on at least one device");

    let dataset = FashionMnistDataset::test();
    let items: Vec<_> = (0..9).filter_map(|i| dataset.get(i)).collect();
    let truths: Vec<u8> = items.iter().map(|item| item.label).collect();

    let batch = FashionMnistBatcher.batch(items, &device);
    let preds = model
        .forward(batch.images)
        .argmax(1)
        .into_data()
        .iter::<i64>()
        .collect::<Vec<_>>();

    println!("\nSample predictions:");
    for (pred, truth) in preds.iter().zip(truths) {
        let pred_name = CLASS_NAMES[*pred as usize];
        let truth_name = CLASS_NAMES[truth as usize];
        let mark = if *pred as u8 == truth { "OK " } else { "XX " };
        println!("  {mark} Pred: {pred_name:<12} | Truth: {truth_name}");
    }
}
