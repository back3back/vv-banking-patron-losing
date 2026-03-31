# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Start

```bash
# Install dependencies
uv sync

# Run the main demo
uv run python -m bank_patron_losing

# Run preprocessing only
uv run src/bank_patron_losing/preprocess.py

# Run training only
uv run src/bank_patron_losing/train.py

# Run analysis only
uv run src/bank_patron_losing/analysis.py
```

## Project Structure

```
bank-patron-losing/
├── src/bank_patron_losing/
│   ├── __init__.py       # Module exports, main() demo function
│   ├── preprocess.py     # Data preprocessing (StandardScaler/Quantile discretization)
│   ├── split.py          # Dataset splitting with stratified sampling
│   ├── train.py          # Model training (DecisionTree/SVM/MLP)
│   └── analysis.py       # ROC, confusion matrix, K-fold CV
├── dataset/
│   ├── Churn-Modelling-0-original.csv  # Raw data
│   ├── scaled/                         # Standardized features (SVM/NN)
│   └── tree/                           # Discretized features (DecisionTree)
├── model/                              # Saved models (.pkl)
├── main.ipynb                          # Jupyter notebook workflow
└── pyproject.toml                      # uv project config
```

## Architecture

**Data Flow**: Raw CSV → Preprocessor (encode → balance → transform) → Train/Test Split → Model Training → Analysis

**Three Model Pipelines**:

| Model                | Preprocessing                               | Key Settings                                         |
| -------------------- | ------------------------------------------- | ---------------------------------------------------- |
| DecisionTree         | Discretization (33%/66% quantiles → 0/1/2) | `criterion='gini'`, `max_depth=6`                |
| SVM                  | Z-score StandardScaler                      | `class_weight='balanced'`                          |
| MLP (Neural Network) | Z-score StandardScaler                      | `early_stopping=True`, `validation_fraction=0.1` |

**Key Classes/Functions**:

- `BankDataPreprocessor` (preprocess.py): Reusable preprocessor with `fit()`/`transform()`/`save()`/`load()` pattern
- `create_train_test_data()`: One-stop function for load → preprocess → split → save
- `train_*()` functions (train.py): Each model has its own training function with appropriate preprocessing
- `draw_roc_curve()`, `draw_confusion_matrix()`, `fold_cross_validation()` (analysis.py): Evaluation utilities

## Usage Patterns

**Programmatic API**:

```python
from bank_patron_losing import create_train_test_data, train_svm, BankDataPreprocessor

# Option 1: One-stop data preparation
preprocessor, X_train, X_test, y_train, y_test = create_train_test_data(
    "./dataset/Churn-Modelling-0-original.csv",
    output_dir="./dataset/scaled",
    discretize=False,  # False=StandardScaler for SVM/NN, True=quantile bins for DT
    random_state=10
)

# Option 2: Manual preprocessing with reusable preprocessor
df = pd.read_csv("./dataset/Churn-Modelling-0-original.csv")
preprocessor = BankDataPreprocessor(random_state=10)
X, y = preprocessor.fit_transform(df, balance=True, discretize=False)
preprocessor.save("./dataset/preprocessor.pkl")  # Save for inference
```

**Jupyter Workflow**: Execute `main.ipynb` cells in order for complete analysis pipeline.

## Important Notes

- **Preprocessing is model-specific**: DecisionTree works with raw/discretized features; SVM/NN require standardized features
- **Class imbalance handling**: Training data is balanced via random undersampling to ~400 samples per class
- **Hardcoded paths**: Model/data paths are hardcoded in functions (e.g., `./model/dt_model.pkl`, `./dataset/`), not configurable via parameters
- **Target column**: `Exited` (1=churned, 0=retained)
- **Dropped columns**: `RowNumber`, `CustomerId`, `Surname`, `EB` (not used in features)
