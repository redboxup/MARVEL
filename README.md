# MARVEL: Margin Adjusted Robust von Mises-Fischer Expert Learning for Long-Tailed Out-of-Distribution Detection

## Overview

MARVEL is a novel approach for out-of-distribution (OOD) detection in medical imaging that addresses critical gaps in existing methods. Designed with clinical deployment in mind, MARVEL combines a nonlinear von Mises-Fisher classifier with a multi-expert framework to reliably identify unfamiliar cases while handling imbalanced medical datasets. By explicitly training an outlier expert to distinguish inlier from outlier data, MARVEL achieves state-of-the-art OOD detection performance across multiple clinical benchmarks (RFMiD, ISIC2019, NCTCRC), enabling safer AI-assisted diagnosis through confident case deferral to clinicians.

This repository contains the official implementation of MARVEL. It is designed to facilitate reproducibility of our results and enable adoption in medical imaging workflows.

## Installation

### Setting Up the Environment

**Prerequisites:** Python 3.8+ and pip

1. Clone the repository:
```bash
git clone <repository-url>
cd MARVEL
```

2. Create a virtual environment (recommended):
```bash
python3 -m venv marvel_env
source marvel_env/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

Your environment is now ready to use MARVEL!

### Dataset Preprocessing

Before training, you need to preprocess your medical imaging datasets. MARVEL supports multiple datasets (RFMiD, ISIC2019, NearOOD) with automatic handling of dataset-specific requirements.

```bash
python preprocess.py \
  --dataset <dataset_name> \
  --raw-data-dir <path_to_raw_dataset> \
  --out-data-dir <path_to_output> \
  --resize 224 \
  --num-workers 8
```

**Arguments:**
- `--dataset`: Dataset to preprocess. Options: `rfmid`, `isic2019`, `nnctcrc`
- `--raw-data-dir`: Path to the downloaded and unzipped raw dataset
- `--out-data-dir`: Path where preprocessed dataset will be saved
- `--resize`: Target image resolution (default: 224)
- `--num-workers`: Number of parallel workers for preprocessing (default: 8)

Once preprocessing is complete, your datasets are ready for training!

## Usage

### Training

After preprocessing your datasets, you can train MARVEL using the configuration-based training pipeline.

**Basic training command:**

```bash
python run.py
```

This will:
1. Load the default configuration from `expconf.yaml`
2. Create experiment directories for logs, results, and model weights
3. Train the model on your dataset
4. Perform two-stage training (initial training + stage two refinement)
5. Evaluate on the test split
6. Generate OOD detection metrics
7. Create t-SNE visualizations

**Output files:**

The training pipeline generates the following outputs in the experiment directory:
- `logs/`: Training logs and debugging information
- `weights/`: Model checkpoints saved at each epoch
- `results/all_results.csv`: Comprehensive evaluation metrics and OOD detection results
- `results/`: t-SNE visualizations and additional analysis

**Configuration:**

Training behavior is controlled via `expconf.yaml`. Key parameters include:
- Dataset name and paths
- Model architecture and hyperparameters
- Optimizer and scheduler settings
- Training epochs and batch sizes
- OOD evaluation datasets

Modify `expconf.yaml` to customize training for your specific use case.

**Example - Training on ISIC 2019:**

Update `expconf.yaml` with:
```yaml
dataset:
  name: isic2019
  data_dir: /path/to/processed/isic2019
trainer:
  epochs: 100
  batch_size: 32
```

Then run:
```bash
python run.py
```

Once training completes, results and model weights will be saved to the experiment directory for evaluation and deployment.