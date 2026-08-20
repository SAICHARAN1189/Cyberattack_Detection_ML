"""
Cyberattack Detection - Multiclass Classification (OPTIONAL EXTENSION)
======================================================================
Classifies network traffic into specific attack categories:
Normal, Generic, Exploits, Fuzzers, DoS, Reconnaissance,
Analysis, Backdoor, Shellcode, Worms
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report
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
MODEL_PATH = os.path.join(BASE_DIR, 'multiclass_model.pkl')
RANDOM_STATE = 42

os.makedirs(PLOTS_DIR, exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("=" * 60)
print("MULTICLASS CYBERATTACK CLASSIFICATION")
print("=" * 60)

print("\n[1/5] Loading data...")
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# ============================================================
# 2. PREPROCESSING
# ============================================================
print("\n[2/5] Preprocessing...")

DROP_COLS = ['id']
TARGET_COL = 'attack_cat'
BINARY_COL = 'label'

# Clean attack_cat - strip whitespace
train_df[TARGET_COL] = train_df[TARGET_COL].str.strip()
test_df[TARGET_COL] = test_df[TARGET_COL].str.strip()

# Encode target labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_df[TARGET_COL])
y_test = label_encoder.transform(test_df[TARGET_COL])
class_names = label_encoder.classes_

print(f"  Classes ({len(class_names)}): {list(class_names)}")

# Separate features
X_train = train_df.drop(columns=DROP_COLS + [TARGET_COL, BINARY_COL])
X_test = test_df.drop(columns=DROP_COLS + [TARGET_COL, BINARY_COL])

# Identify column types
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

# Build preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_cols)
    ],
    remainder='drop'
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(f"  Features: {X_train_processed.shape[1]}")
print(f"  Training samples: {len(y_train):,}")
print(f"  Testing samples:  {len(y_test):,}")

# ============================================================
# 3. TRAIN MULTICLASS XGBOOST
# ============================================================
print("\n[3/5] Training multiclass XGBoost...")
start_time = time.time()

xgb_multi = XGBClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    random_state=RANDOM_STATE,
    eval_metric='mlogloss',
    use_label_encoder=False,
    n_jobs=-1,
    tree_method='hist',
    objective='multi:softprob',
    num_class=len(class_names)
)

xgb_multi.fit(
    X_train_processed, y_train,
    eval_set=[(X_test_processed, y_test)],
    verbose=False
)
train_time = time.time() - start_time
print(f"  Trained in {train_time:.1f}s")

# ============================================================
# 4. EVALUATE
# ============================================================
print("\n[4/5] Evaluating...")

y_pred = xgb_multi.predict(X_test_processed)
accuracy = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average='macro')
f1_weighted = f1_score(y_test, y_pred, average='weighted')

print(f"\n  Accuracy:          {accuracy:.4f}")
print(f"  F1 (macro):        {f1_macro:.4f}")
print(f"  F1 (weighted):     {f1_weighted:.4f}")

print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(14, 11))
sns.heatmap(cm, annot=True, fmt=',d', cmap='YlOrRd',
            xticklabels=class_names, yticklabels=class_names,
            annot_kws={'size': 10}, ax=ax)
ax.set_xlabel('Predicted Label', fontsize=13)
ax.set_ylabel('True Label', fontsize=13)
ax.set_title('Multiclass Confusion Matrix\nUNSW-NB15 Attack Categories', fontsize=15, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'multiclass_confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: plots/multiclass_confusion_matrix.png")

# Per-class accuracy
print("\n  PER-CLASS ACCURACY:")
for i, cls in enumerate(class_names):
    mask = y_test == i
    if mask.sum() > 0:
        cls_acc = accuracy_score(y_test[mask], y_pred[mask])
        print(f"    {cls:<20s} {cls_acc:.4f} ({mask.sum():,} samples)")

# ============================================================
# 5. SAVE MODEL
# ============================================================
print("\n[5/5] Saving multiclass model...")

multiclass_pipeline = {
    'preprocessor': preprocessor,
    'model': xgb_multi,
    'label_encoder': label_encoder,
    'class_names': list(class_names),
    'feature_columns': list(X_train.columns),
    'categorical_cols': categorical_cols,
    'numerical_cols': numerical_cols,
}

joblib.dump(multiclass_pipeline, MODEL_PATH)
print(f"  Saved: {MODEL_PATH}")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("MULTICLASS RESULTS SUMMARY")
print("-" * 60)
print(f"Classes:        {len(class_names)}")
print(f"Accuracy:       {accuracy:.4f}")
print(f"F1 (macro):     {f1_macro:.4f}")
print(f"F1 (weighted):  {f1_weighted:.4f}")
print(f"Training Time:  {train_time:.1f}s")
print("=" * 60)
