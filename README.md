# Cyberattack Detection Using Machine Learning

A machine learning system for detecting malicious network traffic using the UNSW-NB15 dataset. The system classifies network connections as **NORMAL** or **ATTACK** using XGBoost.

## Problem Statement

Network intrusion detection is critical for cybersecurity. This project builds a binary classifier that analyzes network traffic features to identify potential cyberattacks in real-time. Missing an actual attack (false negative) is far more dangerous than a false alarm, so the system prioritizes high recall.

## Dataset

**UNSW-NB15** — Created by the Australian Centre for Cyber Security (ACCS).

| Set | Samples | Normal | Attack |
|-----|---------|--------|--------|
| Training | 175,341 | 56,000 (31.9%) | 119,341 (68.1%) |
| Testing | 82,332 | 37,000 (44.9%) | 45,332 (55.1%) |

Attack categories include: Generic, Exploits, Fuzzers, DoS, Reconnaissance, Analysis, Backdoor, Shellcode, and Worms.

## Features

The dataset contains **42 network traffic features** (after removing ID and target columns):

- **Flow features**: duration, protocol, service, state
- **Packet features**: source/destination packets, bytes, TTL
- **Content features**: HTTP transaction depth, response body length
- **Time features**: jitter, inter-packet arrival time
- **Connection features**: TCP window, sequence numbers, RTT
- **General purpose features**: connection counts, FTP/HTTP indicators

### Categorical Features (3)
- `proto` — Network protocol (133 unique values)
- `service` — Network service (13 unique values)  
- `state` — Connection state (9 unique values)

## Preprocessing

1. Drop `id` column (row identifier, no predictive value)
2. Separate `attack_cat` (multiclass target) from features
3. Ordinal encode categorical features (`proto`, `service`, `state`)
4. Standard scale numerical features
5. scikit-learn `ColumnTransformer` ensures reproducibility and prevents data leakage

## Models Used

### Baseline: Random Forest
- 100 trees, max depth 20
- Used for quick performance reference

### Main Model: XGBoost Classifier
- 300 estimators, max depth 8, learning rate 0.1
- Histogram-based tree method for speed
- Regularization (L1 + L2) to prevent overfitting

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Accuracy | Overall correctness |
| Precision | Of predicted attacks, how many are real |
| Recall | Of real attacks, how many are caught |
| F1 Score | Harmonic mean of precision and recall |
| ROC-AUC | Area under the ROC curve |
| FPR | False Positive Rate |
| FNR | False Negative Rate (critical for cybersecurity) |

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python train_model.py
```

This will:
- Load and preprocess the UNSW-NB15 dataset
- Train Random Forest baseline and XGBoost main model
- Evaluate on the official test set
- Generate visualization plots in `plots/`
- Save the model to `cyberattack_model.pkl`
- Run prediction demos

### 3. Make Predictions
```bash
python predict.py
```

### 4. Use in Your Code
```python
from predict import load_model, predict

pipeline = load_model()

# Predict on a sample (dict or DataFrame)
result = predict(sample_data, pipeline)
print(result['prediction'])   # 'NORMAL' or 'ATTACK'
print(result['confidence'])   # e.g., 98.5
```

## Example Prediction

```
Sample #0:
  Prediction: NORMAL (confidence: 99.2%)
  Actual:     NORMAL
  P(Normal):  0.9920
  P(Attack):  0.0080
```

## Project Structure

```
Cyberattack_Detection_ML/
├── UNSW_NB15_training-set.csv    # Training data (175,341 samples)
├── UNSW_NB15_testing-set.csv     # Testing data (82,332 samples)
├── train_model.py                # Main training pipeline
├── predict.py                    # Prediction module
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── cyberattack_model.pkl         # Saved model + preprocessor
├── preprocessing.pkl             # Saved preprocessor (standalone)
└── plots/
    ├── class_distribution.png    # Normal vs Attack distribution
    ├── confusion_matrix.png      # Confusion matrix heatmap
    ├── roc_curve.png             # ROC curve
    └── feature_importance.png    # Top 20 feature importances
```

## Future Improvements

1. **Multiclass classification** — Classify specific attack types (Generic, Exploits, DoS, etc.)
2. **Hyperparameter optimization** — Bayesian optimization with Optuna
3. **Feature engineering** — Interaction features, polynomial features
4. **Ensemble methods** — Stacking XGBoost with LightGBM and CatBoost
5. **Real-time detection** — Deploy as a streaming pipeline
6. **Threshold tuning** — Optimize decision threshold for recall vs precision trade-off
7. **Deep learning** — LSTM/Transformer for sequential network traffic patterns
8. **Explainability** — SHAP values for individual prediction explanations
