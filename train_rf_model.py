"""
Train and Save Random Forest Model for Resume Scoring
This creates a trained model that the app will use for predictions
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import pickle

print("=" * 80)
print("TRAINING RANDOM FOREST MODEL FOR RESUME SCORING")
print("=" * 80)

# Generate training data (1000 samples)
np.random.seed(42)
n_samples = 1000

# Features: technical_skills, experience, education, leadership, achievements, cultural_fit
X = np.random.rand(n_samples, 6)

# Generate realistic labels based on weighted sum
weights = np.array([0.35, 0.25, 0.15, 0.10, 0.10, 0.05])
scores = X @ weights

# Create labels: 0=Rejected, 1=Shortlisted, 2=Selected
y = np.zeros(n_samples, dtype=int)
y[scores >= 0.8] = 2  # Selected
y[(scores >= 0.65) & (scores < 0.8)] = 1  # Shortlisted
y[scores < 0.65] = 0  # Rejected

print(f"\nDataset: {n_samples} samples")
print(f"Class distribution:")
print(f"  Rejected (0): {np.sum(y == 0)}")
print(f"  Shortlisted (1): {np.sum(y == 1)}")
print(f"  Selected (2): {np.sum(y == 2)}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Random Forest
print("\nTraining Random Forest (100 trees, max_depth=10)...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# Evaluate
train_score = rf_model.score(X_train, y_train)
test_score = rf_model.score(X_test, y_test)

print(f"\nTraining Accuracy: {train_score:.4f}")
print(f"Test Accuracy: {test_score:.4f}")

# Cross-validation
cv_scores = cross_val_score(rf_model, X, y, cv=5)
print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Feature importance
feature_names = ['technical_skills', 'experience', 'education', 'leadership', 'achievements', 'cultural_fit']
importances = rf_model.feature_importances_

print("\nFeature Importances:")
for name, importance in zip(feature_names, importances):
    print(f"  {name}: {importance:.4f}")

# Save model
model_data = {
    'model': rf_model,
    'feature_names': feature_names,
    'train_accuracy': train_score,
    'test_accuracy': test_score,
    'cv_mean': cv_scores.mean(),
    'cv_std': cv_scores.std()
}

with open('resume_scoring_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("\n" + "=" * 80)
print("✅ MODEL SAVED: resume_scoring_model.pkl")
print("=" * 80)
print("\nThis model will be used by futuristic_app.py for accurate predictions!")
print("Run the app now: streamlit run futuristic_app.py")
