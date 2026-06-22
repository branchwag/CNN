//! FashionMNIST dataset: download, parse, and batch.
//!
//! Mirrors what `torchvision.datasets.FashionMNIST(..., transform=ToTensor())`
//! does in the notebook: fetch the raw IDX files, decode them, and scale pixel
//! values into the [0, 1] range.

use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    prelude::*,
};
use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
};

/// Zalando's official FashionMNIST mirror (same IDX format as MNIST).
const BASE_URL: &str = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion";
const DATA_DIR: &str = "data/FashionMNIST/raw";

pub const WIDTH: usize = 28;
pub const HEIGHT: usize = 28;
pub const NUM_CLASSES: usize = 10;

/// Human-readable class names, matching `train_data.classes` in the notebook.
pub const CLASS_NAMES: [&str; NUM_CLASSES] = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
];

#[derive(Clone, Debug)]
pub struct FashionMnistItem {
    /// Raw pixel values in the [0, 255] range; normalized to [0, 1] when batched.
    pub image: [[f32; WIDTH]; HEIGHT],
    pub label: u8,
}

pub struct FashionMnistDataset {
    items: Vec<FashionMnistItem>,
}

impl FashionMnistDataset {
    pub fn train() -> Result<Self, String> {
        Self::new("train-images-idx3-ubyte", "train-labels-idx1-ubyte")
    }

    pub fn test() -> Result<Self, String> {
        Self::new("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")
    }

    fn new(images_file: &str, labels_file: &str) -> Result<Self, String> {
        let images_path = ensure_file(images_file)?;
        let labels_path = ensure_file(labels_file)?;
        let images = read_images(&images_path);
        let labels = read_labels(&labels_path);

        if images.len() != labels.len() {
            return Err(format!("image/label count mismatch for {images_file}"));
        }

        let items = images
            .into_iter()
            .zip(labels)
            .map(|(image, label)| FashionMnistItem { image, label })
            .collect();

        Ok(Self { items })
    }
}

impl Dataset<FashionMnistItem> for FashionMnistDataset {
    fn get(&self, index: usize) -> Option<FashionMnistItem> {
        self.items.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}

/// Returns the path to a decompressed IDX file, downloading + gunzipping it on
/// first use (the `.gz` lives on the mirror; we cache the plain file locally).
fn ensure_file(name: &str) -> Result<PathBuf, String> {
    let dir = Path::new(DATA_DIR);
    fs::create_dir_all(dir).map_err(|e| format!("Failed to create data dir: {e}"))?;

    let path = dir.join(name);
    if path.exists() {
        return Ok(path);
    }

    let url = format!("{BASE_URL}/{name}.gz");
    println!("Downloading {url}");
    let gz = download(&url).map_err(|e| format!("Could not download {name}: {e}"))?;

    let mut decoder = flate2::read::GzDecoder::new(&gz[..]);
    let mut bytes = Vec::new();
    decoder
        .read_to_end(&mut bytes)
        .map_err(|e| format!("Failed to decompress {name}: {e}"))?;
    fs::write(&path, &bytes).map_err(|e| format!("Failed to cache {name}: {e}"))?;

    Ok(path)
}

fn download(url: &str) -> Result<Vec<u8>, String> {
    let response = ureq::get(url)
        .call()
        .map_err(|e| format!("{e}"))?;
    let mut buf = Vec::new();
    response
        .into_body()
        .into_reader()
        .read_to_end(&mut buf)
        .map_err(|e| format!("{e}"))?;
    Ok(buf)
}

/// Parses an IDX3 image file: 16-byte header (magic, count, rows, cols) then
/// `count * rows * cols` bytes of pixels.
fn read_images(path: &Path) -> Vec<[[f32; WIDTH]; HEIGHT]> {
    let data = fs::read(path).expect("read image file");
    let count = u32::from_be_bytes(data[4..8].try_into().unwrap()) as usize;

    let mut images = Vec::with_capacity(count);
    let mut offset = 16;
    for _ in 0..count {
        let mut image = [[0.0f32; WIDTH]; HEIGHT];
        for row in image.iter_mut() {
            for pixel in row.iter_mut() {
                *pixel = data[offset] as f32;
                offset += 1;
            }
        }
        images.push(image);
    }
    images
}

/// Parses an IDX1 label file: 8-byte header (magic, count) then one byte each.
fn read_labels(path: &Path) -> Vec<u8> {
    let data = fs::read(path).expect("read label file");
    data[8..].to_vec()
}

#[derive(Clone, Default)]
pub struct FashionMnistBatcher;

#[derive(Clone, Debug)]
pub struct FashionMnistBatch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> Batcher<B, FashionMnistItem, FashionMnistBatch<B>> for FashionMnistBatcher {
    fn batch(&self, items: Vec<FashionMnistItem>, device: &B::Device) -> FashionMnistBatch<B> {
        let images = items
            .iter()
            .map(|item| TensorData::from(item.image))
            .map(|data| Tensor::<B, 2>::from_data(data, device))
            .map(|tensor| tensor.reshape([1, 1, HEIGHT, WIDTH]))
            // ToTensor: scale pixels from [0, 255] to [0, 1].
            .map(|tensor| tensor / 255.0)
            .collect();

        let targets = items
            .iter()
            .map(|item| Tensor::<B, 1, Int>::from_data([item.label as i64], device))
            .collect();

        FashionMnistBatch {
            images: Tensor::cat(images, 0),
            targets: Tensor::cat(targets, 0),
        }
    }
}
