"""
Cyberattack Detection - Prediction Module
==========================================
Load the trained model and make predictions on new network traffic samples.
"""

import os
import numpy as np
import pandas as pd
import joblib


def load_model(model_path=None):
    """Load the trained cyberattack detection pipeline."""
    if model_path is None:
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cyberattack_model.pkl')
    
    pipeline = joblib.load(model_path)
    return pipeline


def predict(sample_data, pipeline=None):
    """
    Predict whether network traffic is NORMAL or ATTACK.
    
    Parameters
    ----------
    sample_data : dict or pd.DataFrame
        Network traffic features. If dict, will be converted to DataFrame.
    pipeline : dict, optional
        Loaded pipeline. If None, will load from default path.
    
    Returns
    -------
    dict
        Prediction result with label, confidence, and probabilities.
    """
    if pipeline is None:
        pipeline = load_model()
    
    preprocessor = pipeline['preprocessor']
    model = pipeline['model']
    feature_columns = pipeline['feature_columns']
    
    # Convert dict to DataFrame if necessary
    if isinstance(sample_data, dict):
        sample_data = pd.DataFrame([sample_data])
    
    # Ensure only expected feature columns are used
    # Fill missing columns with 0 (handles partial input)
    for col in feature_columns:
        if col not in sample_data.columns:
            sample_data[col] = 0
    
    sample_data = sample_data[feature_columns]
    
    # Preprocess
    sample_processed = preprocessor.transform(sample_data)
    
    # Predict
    prediction = model.predict(sample_processed)[0]
    probabilities = model.predict_proba(sample_processed)[0]
    
    label = "ATTACK" if prediction == 1 else "NORMAL"
    confidence = max(probabilities) * 100
    
    result = {
        'prediction': label,
        'prediction_code': int(prediction),
        'confidence': round(confidence, 2),
        'probability_normal': round(float(probabilities[0]), 4),
        'probability_attack': round(float(probabilities[1]), 4),
    }
    
    return result


def predict_batch(data, pipeline=None):
    """
    Predict on multiple network traffic samples.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with network traffic features.
    pipeline : dict, optional
        Loaded pipeline.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with predictions and probabilities.
    """
    if pipeline is None:
        pipeline = load_model()
    
    preprocessor = pipeline['preprocessor']
    model = pipeline['model']
    feature_columns = pipeline['feature_columns']
    
    for col in feature_columns:
        if col not in data.columns:
            data[col] = 0
    
    data_filtered = data[feature_columns]
    data_processed = preprocessor.transform(data_filtered)
    
    predictions = model.predict(data_processed)
    probabilities = model.predict_proba(data_processed)
    
    results = pd.DataFrame({
        'prediction': ['ATTACK' if p == 1 else 'NORMAL' for p in predictions],
        'prediction_code': predictions,
        'confidence': [round(max(prob) * 100, 2) for prob in probabilities],
        'probability_normal': probabilities[:, 0],
        'probability_attack': probabilities[:, 1],
    })
    
    return results


# ============================================================
# DEMO
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("CYBERATTACK DETECTION - PREDICTION DEMO")
    print("=" * 60)
    
    # Load model
    print("\nLoading model...")
    pipeline = load_model()
    print("Model loaded successfully!")
    
    # Load test data for demo
    test_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'UNSW_NB15_testing-set.csv')
    test_df = pd.read_csv(test_path)
    
    drop_cols = pipeline['drop_cols']
    target_col = pipeline['target_col']
    multiclass_col = pipeline['multiclass_col']
    
    X_test = test_df.drop(columns=drop_cols + [target_col, multiclass_col])
    y_test = test_df[target_col].values
    
    # Single prediction demo
    print("\n--- Single Prediction Demo ---")
    for i in [0, 50, 200, 1000, 5000]:
        sample = X_test.iloc[[i]]
        result = predict(sample, pipeline)
        actual = "ATTACK" if y_test[i] == 1 else "NORMAL"
        
        print(f"\nSample #{i}:")
        print(f"  Prediction: {result['prediction']} (confidence: {result['confidence']}%)")
        print(f"  Actual:     {actual}")
        print(f"  P(Normal):  {result['probability_normal']}")
        print(f"  P(Attack):  {result['probability_attack']}")
    
    # Batch prediction demo
    print("\n\n--- Batch Prediction Demo (first 10 samples) ---")
    batch_results = predict_batch(X_test.head(10), pipeline)
    batch_results['actual'] = ['ATTACK' if y_test[i] == 1 else 'NORMAL' for i in range(10)]
    print(batch_results.to_string(index=False))
    
    print("\nPrediction demo complete!")
