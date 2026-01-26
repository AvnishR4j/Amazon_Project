"""
Amazon Supply Chain Intelligence
Fast Model Training (No Hyperparameter Tuning)
==============================================

Quick training for demonstration purposes.

Author: ARUN AK
Course: UCS321 - AI for Engineers
"""

import pandas as pd
import numpy as np
import warnings
import os
import joblib
from datetime import datetime

# Scikit-learn imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

# Configuration
PROCESSED_DATA_PATH = "data/processed/"
MODEL_SAVE_PATH = "models/"
REPORTS_PATH = "reports/"
RANDOM_STATE = 42
TARGET_NAMES = ['On-Time', 'At-Risk', 'Delayed']

def main():
    print("\n" + "=" * 60)
    print("AMAZON SUPPLY CHAIN INTELLIGENCE")
    print("Fast Model Training Pipeline")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    print("\n▶ Loading data...")
    X_train = pd.read_csv(f"{PROCESSED_DATA_PATH}X_train.csv")
    X_test = pd.read_csv(f"{PROCESSED_DATA_PATH}X_test.csv")
    y_train = pd.read_csv(f"{PROCESSED_DATA_PATH}y_train.csv").values.ravel()
    y_test = pd.read_csv(f"{PROCESSED_DATA_PATH}y_test.csv").values.ravel()
    print(f"  ✓ Training: {X_train.shape}, Test: {X_test.shape}")

    # Handle class imbalance
    print("\n▶ Applying SMOTE...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"  ✓ Resampled: {X_resampled.shape}")

    # Define models with good default parameters
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced', C=1
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=150, max_depth=15, random_state=RANDOM_STATE,
            class_weight='balanced', n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=150, max_depth=6, learning_rate=0.1,
            random_state=RANDOM_STATE, use_label_encoder=False,
            eval_metric='mlogloss', n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            random_state=RANDOM_STATE
        ),
        'SVM': SVC(
            C=10, kernel='rbf', gamma='scale', probability=True,
            random_state=RANDOM_STATE, class_weight='balanced'
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=7, weights='distance', n_jobs=-1
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=15, random_state=RANDOM_STATE, class_weight='balanced'
        ),
        'Neural Network': MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=300,
            random_state=RANDOM_STATE, early_stopping=True
        )
    }

    # Train and evaluate
    print("\n" + "=" * 60)
    print("MODEL TRAINING & EVALUATION")
    print("=" * 60)

    results = []
    best_score = 0
    best_model = None
    best_model_name = ""

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for name, model in models.items():
        print(f"\n▶ Training {name}...", end=" ")

        try:
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_resampled, y_resampled,
                cv=cv, scoring='f1_weighted', n_jobs=-1
            )
            cv_mean = cv_scores.mean()

            # Train on full data
            model.fit(X_resampled, y_resampled)

            # Evaluate on test set
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted')
            rec = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            results.append({
                'Model': name,
                'CV_Score': cv_mean,
                'Accuracy': acc,
                'Precision': prec,
                'Recall': rec,
                'F1-Score': f1
            })

            print(f"✓ F1={f1:.4f}")

            # Track best model
            if f1 > best_score:
                best_score = f1
                best_model = model
                best_model_name = name

            # Save model
            os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
            model_filename = name.lower().replace(' ', '_') + '.joblib'
            joblib.dump(model, os.path.join(MODEL_SAVE_PATH, model_filename))

        except Exception as e:
            print(f"✗ Error: {str(e)[:50]}")

    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('F1-Score', ascending=False)

    print("\n" + "=" * 60)
    print("MODEL RANKING")
    print("=" * 60)
    print(results_df.to_string(index=False))

    # Save best model
    print(f"\n▶ Saving best model: {best_model_name}")
    joblib.dump(best_model, os.path.join(MODEL_SAVE_PATH, 'best_model.joblib'))
    joblib.dump({
        'model_name': best_model_name,
        'cv_score': best_score,
        'trained_at': datetime.now().isoformat()
    }, os.path.join(MODEL_SAVE_PATH, 'model_info.joblib'))

    # Save results
    os.makedirs(REPORTS_PATH, exist_ok=True)
    results_df.to_csv(os.path.join(REPORTS_PATH, 'model_comparison.csv'), index=False)

    # Print classification report for best model
    print("\n" + "=" * 60)
    print(f"BEST MODEL: {best_model_name}")
    print("=" * 60)
    y_pred_best = best_model.predict(X_test)
    print(classification_report(y_test, y_pred_best, target_names=TARGET_NAMES))

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nModels saved to: {MODEL_SAVE_PATH}")
    print(f"Results saved to: {REPORTS_PATH}model_comparison.csv")

if __name__ == "__main__":
    main()
