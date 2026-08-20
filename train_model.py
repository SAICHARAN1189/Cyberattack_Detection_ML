"""
Cyberattack Detection Using Machine Learning
=============================================
Binary classification of network traffic as NORMAL or ATTACK
using the UNSW-NB15 dataset with XGBoost.

Author: ML Workshop Project
Dataset: UNSW-NB15
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, 'UNSW_NB15_training-set.csv')
TEST_PATH = os.path.join(BASE_DIR, 'UNSW_NB15_testing-set.csv')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
MODEL_PATH = os.path.join(BASE_DIR, 'cyberattack_model.pkl')
PIPELINE_PATH = os.path.join(BASE_DIR, 'preprocessing.pkl')
RANDOM_STATE = 42

os.makedirs(PLOTS_DIR, exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("=" * 60)
print("CYBERATTACK DETECTION - TRAINING PIPELINE")
print("=" * 60)

print("\n[1/8] Loading data...")
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
print(f"  Training set: {train_df.shape}")
print(f"  Testing set:  {test_df.shape}")

# ============================================================
# 2. PREPROCESSING
# ============================================================
print("\n[2/8] Preprocessing...")

# --- Drop identifier column ---
DROP_COLS = ['id']
# attack_cat is the multiclass target; drop for binary classification
MULTICLASS_COL = 'attack_cat'
TARGET_COL = 'label'

# Save attack_cat for optional multiclass later
train_attack_cat = train_df[MULTICLASS_COL].copy()
test_attack_cat = test_df[MULTICLASS_COL].copy()

# Separate features and target
X_train = train_df.drop(columns=DROP_COLS + [TARGET_COL, MULTICLASS_COL])
y_train = train_df[TARGET_COL].values
X_test = test_df.drop(columns=DROP_COLS + [TARGET_COL, MULTICLASS_COL])
y_test = test_df[TARGET_COL].values

print(f"  Features: {X_train.shape[1]}")
print(f"  Target distribution (train): Normal={sum(y_train==0):,} | Attack={sum(y_train==1):,}")
print(f"  Target distribution (test):  Normal={sum(y_test==0):,} | Attack={sum(y_test==1):,}")

# --- Identify column types ---
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

print(f"  Categorical columns ({len(categorical_cols)}): {categorical_cols}")
print(f"  Numerical columns:  {len(numerical_cols)}")

# --- Build preprocessing pipeline ---
# OrdinalEncoder handles unseen categories with 'use_encoded_value'
cat_transformer = OrdinalEncoder(
    handle_unknown='use_encoded_value',
    unknown_value=-1
)

num_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, numerical_cols),
        ('cat', cat_transformer, categorical_cols)
    ],
    remainder='drop'
)

# Fit on training data only (no data leakage)
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(f"  Processed feature shape: {X_train_processed.shape}")

# Get feature names for later
num_feature_names = numerical_cols
cat_feature_names = categorical_cols
all_feature_names = num_feature_names + cat_feature_names

# ============================================================
# 3. CLASS DISTRIBUTION PLOT
# ============================================================
print("\n[3/8] Creating class distribution plot...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Training distribution
train_counts = pd.Series(y_train).value_counts().sort_index()
colors_dist = ['#2ecc71', '#e74c3c']
labels_dist = ['Normal (0)', 'Attack (1)']

axes[0].bar(labels_dist, train_counts.values, color=colors_dist, edgecolor='white', linewidth=1.5)
axes[0].set_title('Training Set Distribution', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Count', fontsize=12)
for i, v in enumerate(train_counts.values):
    axes[0].text(i, v + 1000, f'{v:,}\n({v/len(y_train)*100:.1f}%)',
                 ha='center', fontsize=11, fontweight='bold')

# Testing distribution
test_counts = pd.Series(y_test).value_counts().sort_index()
axes[1].bar(labels_dist, test_counts.values, color=colors_dist, edgecolor='white', linewidth=1.5)
axes[1].set_title('Testing Set Distribution', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Count', fontsize=12)
for i, v in enumerate(test_counts.values):
    axes[1].text(i, v + 500, f'{v:,}\n({v/len(y_test)*100:.1f}%)',
                 ha='center', fontsize=11, fontweight='bold')

plt.suptitle('Normal vs Attack Class Distribution', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'class_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: plots/class_distribution.png")

# ============================================================
# 4. BASELINE MODEL (Random Forest)
# ============================================================
print("\n[4/8] Training baseline model (Random Forest)...")
rf_start = time.time()

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf_model.fit(X_train_processed, y_train)
rf_time = time.time() - rf_start

rf_pred = rf_model.predict(X_test_processed)
rf_acc = accuracy_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)
print(f"  Random Forest: Accuracy={rf_acc:.4f}, F1={rf_f1:.4f} (trained in {rf_time:.1f}s)")

# ============================================================
# 5. MAIN MODEL (XGBoost)
# ============================================================
print("\n[5/8] Training main model (XGBoost)...")
xgb_start = time.time()

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=RANDOM_STATE,
    eval_metric='logloss',
    use_label_encoder=False,
    n_jobs=-1,
    tree_method='hist'  # Fast histogram-based method (works on CPU and GPU)
)

xgb_model.fit(
    X_train_processed, y_train,
    eval_set=[(X_test_processed, y_test)],
    verbose=False
)
xgb_time = time.time() - xgb_start

print(f"  XGBoost trained in {xgb_time:.1f}s")

# ============================================================
# 6. EVALUATION
# ============================================================
print("\n[6/8] Evaluating on official test set...")

y_pred = xgb_model.predict(X_test_processed)
y_pred_proba = xgb_model.predict_proba(X_test_processed)[:, 1]

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
cm = confusion_matrix(y_test, y_pred)

# False Positive Rate and False Negative Rate
TN, FP, FN, TP = cm.ravel()
fpr_val = FP / (FP + TN)
fnr_val = FN / (FN + TP)

print(f"\n  Accuracy:    {accuracy:.4f}")
print(f"  Precision:   {precision:.4f}")
print(f"  Recall:      {recall:.4f}")
print(f"  F1 Score:    {f1:.4f}")
print(f"  ROC-AUC:     {roc_auc:.4f}")
print(f"  FPR:         {fpr_val:.4f}")
print(f"  FNR:         {fnr_val:.4f}")

print(f"\n  Confusion Matrix:")
print(f"    TN={TN:,}  FP={FP:,}")
print(f"    FN={FN:,}  TP={TP:,}")

print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))

# Model comparison
print("  MODEL COMPARISON:")
print(f"  {'Model':<20} {'Accuracy':<12} {'F1':<12} {'Time (s)':<10}")
print(f"  {'-'*54}")
print(f"  {'Random Forest':<20} {rf_acc:<12.4f} {rf_f1:<12.4f} {rf_time:<10.1f}")
print(f"  {'XGBoost':<20} {accuracy:<12.4f} {f1:<12.4f} {xgb_time:<10.1f}")

# ============================================================
# 7. VISUALIZATIONS
# ============================================================
print("\n[7/8] Creating visualizations...")

# --- 7a. Confusion Matrix ---
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt=',d', cmap='Blues',
            xticklabels=['Normal', 'Attack'],
            yticklabels=['Normal', 'Attack'],
            annot_kws={'size': 16}, ax=ax)
ax.set_xlabel('Predicted Label', fontsize=13)
ax.set_ylabel('True Label', fontsize=13)
ax.set_title('Confusion Matrix - XGBoost\nCyberattack Detection', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: plots/confusion_matrix.png")

# --- 7b. ROC Curve ---
fpr_curve, tpr_curve, _ = roc_curve(y_test, y_pred_proba)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr_curve, tpr_curve, color='#3498db', linewidth=2.5,
        label=f'XGBoost (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], color='gray', linewidth=1, linestyle='--', label='Random Baseline')
ax.fill_between(fpr_curve, tpr_curve, alpha=0.15, color='#3498db')
ax.set_xlabel('False Positive Rate', fontsize=13)
ax.set_ylabel('True Positive Rate', fontsize=13)
ax.set_title('ROC Curve - Cyberattack Detection', fontsize=15, fontweight='bold')
ax.legend(fontsize=12, loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'roc_curve.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: plots/roc_curve.png")

# --- 7c. Feature Importance ---
feature_importance = xgb_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': feature_importance
}).sort_values('Importance', ascending=True)

# Top 20 features
top_n = 20
top_features = importance_df.tail(top_n)

fig, ax = plt.subplots(figsize=(10, 8))
bars = ax.barh(top_features['Feature'], top_features['Importance'],
               color=plt.cm.viridis(np.linspace(0.3, 0.9, top_n)), edgecolor='white')
ax.set_xlabel('Feature Importance (Gain)', fontsize=13)
ax.set_title(f'Top {top_n} Most Important Features - XGBoost', fontsize=15, fontweight='bold')
ax.grid(True, axis='x', alpha=0.3)

# Add value labels
for bar, val in zip(bars, top_features['Importance']):
    ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: plots/feature_importance.png")

# Print top 10 feature importances
print("\n  TOP 10 MOST IMPORTANT FEATURES:")
top10 = importance_df.tail(10).iloc[::-1]
for i, (_, row) in enumerate(top10.iterrows(), 1):
    print(f"    {i:2d}. {row['Feature']:<25s} {row['Importance']:.4f}")

# ============================================================
# 8. SAVE MODEL
# ============================================================
print("\n[8/8] Saving model and preprocessing pipeline...")

# Save the full pipeline (preprocessor + model)
full_pipeline = {
    'preprocessor': preprocessor,
    'model': xgb_model,
    'feature_columns': list(X_train.columns),
    'categorical_cols': categorical_cols,
    'numerical_cols': numerical_cols,
    'drop_cols': DROP_COLS,
    'target_col': TARGET_COL,
    'multiclass_col': MULTICLASS_COL,
}

joblib.dump(full_pipeline, MODEL_PATH)
joblib.dump(preprocessor, PIPELINE_PATH)
print(f"  Saved: {MODEL_PATH}")
print(f"  Saved: {PIPELINE_PATH}")

# ============================================================
# PREDICTION DEMO
# ============================================================
print("\n" + "=" * 60)
print("PREDICTION DEMO")
print("=" * 60)

# Take a sample from the test set
sample_indices = [0, 1, 100, 500, 1000]
for idx in sample_indices:
    sample = X_test.iloc[[idx]]
    sample_processed = preprocessor.transform(sample)
    pred = xgb_model.predict(sample_processed)[0]
    proba = xgb_model.predict_proba(sample_processed)[0]
    label = "ATTACK" if pred == 1 else "NORMAL"
    actual = "ATTACK" if y_test[idx] == 1 else "NORMAL"
    confidence = max(proba) * 100
    print(f"\n  Sample #{idx}:")
    print(f"    Prediction:  {label} (confidence: {confidence:.1f}%)")
    print(f"    Actual:      {actual}")
    print(f"    Probabilities: Normal={proba[0]:.4f} | Attack={proba[1]:.4f}")

# ============================================================
# CYBERSECURITY NOTE
# ============================================================
print("\n" + "=" * 60)
print("CYBERSECURITY NOTE")
print("=" * 60)
print("""
  Why Recall and False Negative Rate matter in cybersecurity:

  - A FALSE NEGATIVE means the system classified an ATTACK as NORMAL.
  - This means the attack goes UNDETECTED and can cause:
    * Data breaches and data exfiltration
    * Ransomware deployment
    * Network compromise and lateral movement
    * Financial loss and regulatory penalties
  
  - In cybersecurity, MISSING AN ATTACK is far more dangerous than
    a false alarm (false positive).
  - Therefore, we prioritize HIGH RECALL (catching as many attacks
    as possible) even at the cost of some false positives.
  
  Our model's False Negative Rate: {:.4f} ({:.2f}%)
  This means {:.2f}% of actual attacks would go undetected.
""".format(fnr_val, fnr_val*100, fnr_val*100))

# ============================================================
# FINAL REPORT
# ============================================================
print("=" * 60)
print("CYBERATTACK DETECTION RESULTS")
print("-" * 60)
print(f"Dataset:            UNSW-NB15")
print(f"Training Samples:   {len(y_train):,}")
print(f"Testing Samples:    {len(y_test):,}")
print(f"Number of Features: {X_train_processed.shape[1]}")
print()
print(f"Model:              XGBoost Classifier")
print(f"Training Time:      {xgb_time:.1f}s")
print()
print(f"Accuracy:           {accuracy:.4f}")
print(f"Precision:          {precision:.4f}")
print(f"Recall:             {recall:.4f}")
print(f"F1 Score:           {f1:.4f}")
print(f"ROC-AUC:            {roc_auc:.4f}")
print()
print(f"False Positive Rate: {fpr_val:.4f}")
print(f"False Negative Rate: {fnr_val:.4f}")
print()
print(f"Confusion Matrix:")
print(f"                 Predicted Normal  Predicted Attack")
print(f"  Actual Normal      {TN:>10,}      {FP:>10,}")
print(f"  Actual Attack      {FN:>10,}      {TP:>10,}")
print("=" * 60)
print("Pipeline complete! Model saved to cyberattack_model.pkl")
print("=" * 60)
