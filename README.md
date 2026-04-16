# MARVEL: Margin Adjusted Robust von Mises-Fischer Expert Learning for Long-Tailed Out-of-Distribution Detection

MARVEL is a novel approach for out-of-distribution (OOD) detection in medical imaging that addresses critical gaps in existing methods. Designed with clinical deployment in mind, MARVEL combines a nonlinear von Mises-Fisher classifier with a multi-expert framework to reliably identify unfamiliar cases while handling imbalanced medical datasets. By explicitly training an outlier expert to distinguish inlier from outlier data, MARVEL achieves state-of-the-art OOD detection performance across multiple clinical benchmarks (RFMiD, ISIC2019, NCTCRC), enabling safer AI-assisted diagnosis through confident case deferral to clinicians.

This repository contains the official implementation of MARVEL. It is designed to facilitate reproducibility of our results and enable adoption in medical imaging workflows.

## Installation

### Setting Up the Environment

**Prerequisites:** Python 3.8+ and pip

1. Clone the repository:
```bash
git clone https://github.com/redboxup/MARVEL
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

### Creating Corruptions

Corrupted (near-OOD) datasets used in the paper can be generated using `corr.py`.

```bash
python corr.py +dataset=<name>
```
**Available options:**
* `dataset:` isic2019, rfmid, nctcrc
This script applies a fixed set of corruptions (noise, blur, compression, etc.) to the test split using pre-generated augmentation replays, ensuring consistency across runs.

Generated images are saved to:
```bash
data/preprocessed/nearood/<dataset>/corr/
```

## Usage

### Training

After preprocessing your datasets, you can train MARVEL using the configuration-based training pipeline.
All arguments are passed as configuration overrides.

**Basic command:**

```bash
python run.py <configs>
```

Select predefined configurations:
```bash
python run.py +dataset=<name> +trainer=<name> +scheduler=<name> +optimizer=<name>
```
**Available options:**
* `dataset`: isic2019, rfmid, nctcrc
* `trainer`: crossentropy, ceoutlierexposure, marvel
* `scheduler`: cosine, linear, none
* `optimizer`: adam, sgd

To reproduce the results reported in the paper, run the following commands.

#### ISIC2019
```bash
python run.py \
  +dataset=isic2019 \
  +trainer=marvel \
  +optimizer=adam \
  +scheduler=cosine \
  trainer.epochs=75 \
  trainer.batch_size=64 \
  trainer.learning_rate=1e-4 \
```
#### RFMiD
```bash
python run.py \
  +dataset=rfmid \
  +trainer=marvel \
  +optimizer=adam \
  +scheduler=cosine \
  trainer.epochs=75 \
  trainer.batch_size=64 \
  trainer.learning_rate=1e-4 \
```
#### NCTCRC
```bash
python run.py \
  +dataset=nctcrc \
  +trainer=marvel \
  +optimizer=adam \
  +scheduler=cosine \
  trainer.epochs=75 \
  trainer.batch_size=64 \
  trainer.learning_rate=1e-4 \
```