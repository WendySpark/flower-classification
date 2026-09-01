# Flower Classification (Oxford 102 Flower Dataset)

Image classification project fine-tuning MobileNetV3, ResNet50, and VGG16
(transfer learning, ImageNet-pretrained) on the Oxford 102 category flower
dataset.

## Results (validation set)

From an earlier run (2 epochs/model); re-run `model-training-and-evaluation.ipynb`
to regenerate after the fixes below.

| Model      | Accuracy | Precision | Recall | F1-score |
|------------|----------|-----------|--------|----------|
| MobileNetV3| 0.9254   | 0.9374    | 0.9254 | 0.9254   |
| ResNet50   | 0.9205   | 0.9400    | 0.9205 | 0.9197   |
| VGG16      | 0.8081   | 0.8487    | 0.8081 | 0.8022   |

These are **validation-set** metrics — the Kaggle `test` split has no
ground-truth labels, so there's no held-out labeled data locally for an
unbiased estimate. True test accuracy is only visible after submitting
`submission.csv` (see Pipeline below) to Kaggle. MobileNetV3 is the best of
the three tested here, but VGG16 is a weak, dated baseline — a modern
architecture such as EfficientNetV2-S or ConvNeXt-Tiny would be a fairer
and likely stronger comparison point if extending this project.

## Project Structure

```
├── notebooks/
│   ├── preprocessing.ipynb                # Loads raw images, applies transforms,
│   │                                       # caches train/valid/test as .pt tensors
│   └── model-training-and-evaluation.ipynb# Trains MobileNetV3/ResNet50/VGG16 on the
│                                           # preprocessed tensors and reports metrics
├── dataset/
│   ├── README.md              # Dataset description and source (Oxford VGG + Kaggle)
│   ├── cat_to_name.json       # Category index -> flower name mapping
│   └── sample_submission.csv  # Kaggle submission format
└── README.md
```

## Dataset

The raw image data (~350MB, 102 classes) is **not** included in this
repository. Get it from one of:

- Original source: [Oxford 102 Category Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
- Kaggle (train/valid split + test set used by the notebooks): [Oxford 102 Flower Pytorch](https://www.kaggle.com/c/oxford-102-flower-pytorch/)
- Preprocessed `.pt` tensors: run `preprocessing.ipynb` yourself to generate
  `train.pt`/`valid.pt`/`test.pt` + `idx_to_class.json` from the raw dataset above.

Expected local layout for `preprocessing.ipynb`:

```
dataset/
├── cat_to_name.json
└── dataset/
    ├── train/<class_id>/*.jpg
    ├── valid/<class_id>/*.jpg
    └── test/*.jpg
```

Each notebook's second cell sets its Kaggle-default paths from environment
variables, so you can point them at a local copy without editing the
notebook: set `FLOWER_DATA_ROOT` (default `/kaggle/input/dataset`) to the
folder above, and for the training notebook, `FLOWER_PROCESSED_ROOT`
(default `/kaggle/input/training-dataset`) to wherever the `.pt` files from
`preprocessing.ipynb` live, and `FLOWER_CHECKPOINT_ROOT` (default
`/kaggle/working/checkpoints`) for where model checkpoints get written.

## Pipeline

1. **`preprocessing.ipynb`** — resizes/augments images (224x224, ImageNet
   normalization), builds `ImageFolder` datasets for train/valid and a custom
   `Dataset` for the unlabeled test set, then caches each split to a `.pt`
   file (plus `idx_to_class.json`, mapping label indices back to the real
   category ids used by `cat_to_name.json`) so training doesn't need to
   re-decode images every run.
2. **`model-training-and-evaluation.ipynb`** — loads the cached `.pt` tensors,
   fine-tunes MobileNetV3-Large, ResNet50, and VGG16 (final classifier layer
   replaced for 102 classes), plots loss/accuracy curves, reports
   accuracy/precision/recall/F1 on the validation set, then runs the
   best-performing model on the test set and writes `submission.csv` in the
   `sample_submission.csv` format.

## Notes

- Both notebooks seed `random`/`numpy`/`torch` (`SEED = 42`) for reproducible
  runs.
- Model checkpoints (`checkpoints/<model>/best_model.pth`), training-curve
  plots (`*_training_curves.png`), and `submission.csv` are written by the
  training notebook at runtime and are not tracked in this repo.
- `train_model()`'s learning-rate scheduler now actually steps each epoch,
  and early stopping restores the best checkpoint (by validation loss)
  before returning, rather than leaving whatever state training happened to
  stop on.
