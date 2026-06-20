//! TinyVGG-style CNN, a direct port of `FashionMNISTModelV2` from the notebook.
//!
//! Two conv blocks (each: conv -> ReLU -> conv -> ReLU -> max-pool) followed by
//! a linear classifier. With 28x28 input, each block halves the spatial dims, so
//! we reach 7x7 feature maps before flattening.

use crate::data::{FashionMnistBatch, NUM_CLASSES};
use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig},
        loss::CrossEntropyLossConfig,
        pool::{MaxPool2d, MaxPool2dConfig},
        Linear, LinearConfig, PaddingConfig2d, Relu,
    },
    prelude::*,
    tensor::backend::AutodiffBackend,
    train::{ClassificationOutput, InferenceStep, TrainOutput, TrainStep},
};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    conv3: Conv2d<B>,
    conv4: Conv2d<B>,
    pool: MaxPool2d,
    linear: Linear<B>,
    activation: Relu,
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    #[config(default = 10)]
    hidden_units: usize,
    #[config(default = "NUM_CLASSES")]
    num_classes: usize,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        let h = self.hidden_units;

        // All convs use a 3x3 kernel with stride 1 and padding 1, so they
        // preserve spatial size (kernel_size=3, stride=1, padding=1 in PyTorch).
        let conv = |in_ch: usize, out_ch: usize| {
            Conv2dConfig::new([in_ch, out_ch], [3, 3])
                .with_stride([1, 1])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device)
        };

        Model {
            conv1: conv(1, h),
            conv2: conv(h, h),
            conv3: conv(h, h),
            conv4: conv(h, h),
            pool: MaxPool2dConfig::new([2, 2]).init(),
            linear: LinearConfig::new(h * 7 * 7, self.num_classes).init(device),
            activation: Relu::new(),
        }
    }
}

impl<B: Backend> Model<B> {
    /// Forward pass: [batch, 1, 28, 28] -> [batch, num_classes] logits.
    pub fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 2> {
        // conv_block_1
        let x = self.activation.forward(self.conv1.forward(images));
        let x = self.activation.forward(self.conv2.forward(x));
        let x = self.pool.forward(x);

        // conv_block_2
        let x = self.activation.forward(self.conv3.forward(x));
        let x = self.activation.forward(self.conv4.forward(x));
        let x = self.pool.forward(x);

        // classifier
        let x = x.flatten(1, 3);
        self.linear.forward(x)
    }

    pub fn forward_classification(
        &self,
        images: Tensor<B, 4>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(images);
        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        ClassificationOutput::new(loss, output, targets)
    }
}

impl<B: AutodiffBackend> TrainStep for Model<B> {
    type Input = FashionMnistBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: FashionMnistBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.images, batch.targets);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> InferenceStep for Model<B> {
    type Input = FashionMnistBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: FashionMnistBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.images, batch.targets)
    }
}
