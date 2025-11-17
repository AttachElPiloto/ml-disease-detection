# ML Disease Detection

A machine learning pipeline for disease classification using gene expression samples. This project implements multiple classification models (Decision Tree, XGBoost) and also an autoencoder-based dimensionality reduction to detect diseases such as Rheumatoid Arthritis (RA) and Systemic Lupus Erythematosus (SLE).

## Project Overview

This project analyzes gene expression data to classify patients into disease categories:
- **Healthy**
- **RA** (Rheumatoid Arthritis)
- **SLE** (Systemic Lupus Erythematosus)

## Project Structure

```
.
├── main.py                    # Main orchestration script
├── models.py                  # Classification models (Decision Tree, XGBoost)
├── autoencoder.py             # Autoencoder for dimensionality reduction
├── loading_data.py            # Data loading utilities
├── preprocessing.py           # Data preprocessing and feature engineering
├── visualize_data.py          # Data visualization functions
├── analysis.py                # Model analysis and interpretation
├── reporting.py               # Results reporting and formatting
├── training_data/             # Training datasets (TSV files)
│   ├── healthy_train_data.tsv
│   ├── ra_train_data.tsv
│   └── sle_train_data.tsv
├── test_data/                 # Test datasets (TSV files)
│   ├── healthy_test_data.tsv
│   ├── ra_test_data.tsv
│   └── sle_test_data.tsv
├── saved_pipeline/            # Pre-trained models and pipelines
│   ├── ae.pt                  # Autoencoder weights (PyTorch)
│   └── xgb.json               # XGBoost model
└── notebook/                  # Jupyter notebook
    └── notebook.ipynb
```

## Data

The project uses gene expression data in TSV format with the following structure:
- **Input**: Gene expression (rows = genes, columns = samples)
- **Labels**: Disease classification (Healthy, RA, SLE)

### Training Data
- `healthy_train_data.tsv` - Healthy samples
- `ra_train_data.tsv` - Rheumatoid Arthritis patient samples
- `sle_train_data.tsv` - Systemic Lupus Erythematosus patient samples

### Test Data
- `healthy_test_data.tsv` - Healthy test samples
- `ra_test_data.tsv` - RA test samples
- `sle_test_data.tsv` - SLE test samples

## Models

### 1. Decision Tree Baseline
- Decision tree with ID3
- Serves as an interpretable baseline model

### 2. Decision Tree Regularized
- Depth-limited decision tree for better generalization
- Helps prevent overfitting

### 3. XGBoost
- Gradient boosting classifier
- Standard implementation without class weighting

### 4. XGBoost with Class Weighting
- XGBoost with adjusted class weights
- Addresses class imbalances and difficulty to detect SLE samples correctly.

### 5. Autoencoder + XGBoost
- Autoencoder for dimensionality reduction and faster XGBoost fit
- XGBoost trained on latent representations
- Latent features compress gene expression into abstract representations that are able to reveal disease-specific patterns and insights

## Key Features

- **Preprocessing**: Log transformation, feature variance filtering, normalization
- **Feature Analysis**: Gene variance analysis, importance ranking 
- **Model Evaluation**: F-beta scores, confusion matrices
- **Visualization**: Most important genes analysis, thresholds definition, and model performance comparisons
- **Pipeline Saving**: Save/load trained models and preprocessing pipelines

## Usage

### Run the Complete Pipeline
You need to uncomment #analysis() and run 
```python
python main.py
```

This will:
1. Load training and test data
2. Preprocess gene expression data
3. Train multiple classification models
4. Evaluate models on test set
5. Generate analysis plots and reports

### Evaluate a New Patient

```python
from main import evaluate_model
import pandas as pd

# Load patient gene expression data
patient_data = pd.Series({...})  # gene_name -> expression_value

# Get prediction
pred_label, proba, classes = evaluate_model(patient_data)
```
## Requirements

- Python 3.8+
- NumPy
- Pandas
- Scikit-learn
- XGBoost
- PyTorch
- Matplotlib
- Seaborn

## Installation

```bash
pip install numpy pandas scikit-learn xgboost torch matplotlib seaborn
```