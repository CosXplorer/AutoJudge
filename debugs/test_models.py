import pickle
import numpy as np

# Load all models
with open(r'D:\AutoJudge_v2\models\classification_model.pkl', 'rb') as f:
    clf = pickle.load(f)

with open(r'D:\AutoJudge_v2\models\label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

print("="*60)
print("MODEL DIAGNOSTICS")
print("="*60)

print("\n1. Label Encoder Classes:")
print(f"   Classes: {le.classes_}")

print("\n2. Classification Model Info:")
print(f"   Type: {type(clf).__name__}")
if hasattr(clf, 'n_features_in_'):
    print(f"   Expected features: {clf.n_features_in_}")

print("\n3. Model Parameters:")
if hasattr(clf, 'get_params'):
    params = clf.get_params()
    for key, value in list(params.items())[:5]:
        print(f"   {key}: {value}")

print("\n4. Can predict probabilities:", hasattr(clf, 'predict_proba'))

print("\n" + "="*60)