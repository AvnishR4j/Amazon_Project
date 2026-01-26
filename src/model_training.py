"""
Amazon Supply Chain Intelligence
Model Training Module
======================

This module handles:
- Training multiple classification models
- Hyperparameter tuning
- Cross-validation
- Model comparison
- Best model selection
- Model persistence

Author: ARUN AK
Course: UCS321 - AI for Engineers
"""

import pandas as pd
import numpy as np
import warnings
import os
import joblib
from datetime import datetime
from tqdm import tqdm

# Scikit-learn imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    AdaBoostClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# XGBoost
from xgboost import XGBClassifier

# Model selection & metrics
from sklearn.model_selection import (
    cross_val_score,
    StratifiedKFold,
    RandomizedSearchCV
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# Imbalanced learning
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class ModelConfig:
    """Configuration for model training."""

    # Data paths
    PROCESSED_DATA_PATH = "data/processed/"
    MODEL_SAVE_PATH = "models/"
    REPORTS_PATH = "reports/"

    # Cross-validation
    CV_FOLDS = 5
    RANDOM_STATE = 42

    # Hyperparameter tuning
    N_ITER_SEARCH = 10  # Reduced for faster training
    SCORING = 'f1_weighted'

    # Target classes
    TARGET_NAMES = ['On-Time', 'At-Risk', 'Delayed']


# ============================================================================
# DATA LOADER
# ============================================================================

class ModelDataLoader:
    """Loads preprocessed data for model training."""

    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()

    def load(self) -> tuple:
        """Load train and test data."""
        print("Loading preprocessed data...")

        X_train = pd.read_csv(f"{self.config.PROCESSED_DATA_PATH}X_train.csv")
        X_test = pd.read_csv(f"{self.config.PROCESSED_DATA_PATH}X_test.csv")
        y_train = pd.read_csv(f"{self.config.PROCESSED_DATA_PATH}y_train.csv").values.ravel()
        y_test = pd.read_csv(f"{self.config.PROCESSED_DATA_PATH}y_test.csv").values.ravel()

        print(f"✓ Training data: {X_train.shape}")
        print(f"✓ Test data: {X_test.shape}")

        return X_train, X_test, y_train, y_test


# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

class ModelFactory:
    """Factory class for creating ML models with their hyperparameter spaces."""

    @staticmethod
    def get_models() -> dict:
        """Return dictionary of models with their configs."""
        models = {
            'Logistic Regression': {
                'model': LogisticRegression(
                    max_iter=1000,
                    random_state=ModelConfig.RANDOM_STATE,
                    class_weight='balanced'
                ),
                'params': {
                    'C': [0.1, 1, 10],
                    'solver': ['lbfgs']
                }
            },

            'Random Forest': {
                'model': RandomForestClassifier(
                    random_state=ModelConfig.RANDOM_STATE,
                    n_jobs=-1,
                    class_weight='balanced'
                ),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20],
                    'min_samples_split': [2, 5]
                }
            },

            'XGBoost': {
                'model': XGBClassifier(
                    random_state=ModelConfig.RANDOM_STATE,
                    use_label_encoder=False,
                    eval_metric='mlogloss',
                    n_jobs=-1
                ),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [5, 7],
                    'learning_rate': [0.05, 0.1]
                }
            },

            'Gradient Boosting': {
                'model': GradientBoostingClassifier(
                    random_state=ModelConfig.RANDOM_STATE
                ),
                'params': {
                    'n_estimators': [100, 150],
                    'max_depth': [3, 5],
                    'learning_rate': [0.05, 0.1]
                }
            },

            'Support Vector Machine': {
                'model': SVC(
                    random_state=ModelConfig.RANDOM_STATE,
                    probability=True,
                    class_weight='balanced'
                ),
                'params': {
                    'C': [1, 10],
                    'kernel': ['rbf'],
                    'gamma': ['scale']
                }
            },

            'K-Nearest Neighbors': {
                'model': KNeighborsClassifier(
                    n_jobs=-1
                ),
                'params': {
                    'n_neighbors': [5, 7, 9],
                    'weights': ['distance']
                }
            },

            'Decision Tree': {
                'model': DecisionTreeClassifier(
                    random_state=ModelConfig.RANDOM_STATE,
                    class_weight='balanced'
                ),
                'params': {
                    'max_depth': [10, 15, 20],
                    'min_samples_split': [2, 5]
                }
            },

            'Neural Network': {
                'model': MLPClassifier(
                    random_state=ModelConfig.RANDOM_STATE,
                    max_iter=300,
                    early_stopping=True
                ),
                'params': {
                    'hidden_layer_sizes': [(64, 32), (128, 64)],
                    'activation': ['relu'],
                    'alpha': [0.001]
                }
            }
        }

        return models


# ============================================================================
# MODEL TRAINER
# ============================================================================

class ModelTrainer:
    """Handles model training with cross-validation and hyperparameter tuning."""

    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.models = ModelFactory.get_models()
        self.results = {}
        self.best_models = {}
        self.cv = StratifiedKFold(
            n_splits=self.config.CV_FOLDS,
            shuffle=True,
            random_state=self.config.RANDOM_STATE
        )

    def handle_imbalance(self, X_train: pd.DataFrame, y_train: np.ndarray) -> tuple:
        """Apply SMOTE to handle class imbalance."""
        print("\n" + "=" * 60)
        print("HANDLING CLASS IMBALANCE WITH SMOTE")
        print("=" * 60)

        print("Class distribution before SMOTE:")
        unique, counts = np.unique(y_train, return_counts=True)
        for cls, count in zip(unique, counts):
            print(f"  Class {cls}: {count} ({count/len(y_train)*100:.1f}%)")

        smote = SMOTE(random_state=self.config.RANDOM_STATE)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

        print("\nClass distribution after SMOTE:")
        unique, counts = np.unique(y_resampled, return_counts=True)
        for cls, count in zip(unique, counts):
            print(f"  Class {cls}: {count} ({count/len(y_resampled)*100:.1f}%)")

        return X_resampled, y_resampled

    def train_model(self, name: str, model, params: dict,
                    X_train: np.ndarray, y_train: np.ndarray,
                    tune_hyperparams: bool = True) -> dict:
        """Train a single model with optional hyperparameter tuning."""

        result = {
            'name': name,
            'cv_scores': [],
            'mean_cv_score': 0,
            'std_cv_score': 0,
            'best_params': {},
            'best_model': None
        }

        if tune_hyperparams and params:
            # Hyperparameter tuning with RandomizedSearchCV
            search = RandomizedSearchCV(
                model,
                params,
                n_iter=min(self.config.N_ITER_SEARCH, self._count_combinations(params)),
                cv=self.cv,
                scoring=self.config.SCORING,
                n_jobs=-1,
                random_state=self.config.RANDOM_STATE,
                verbose=0
            )
            search.fit(X_train, y_train)

            result['best_params'] = search.best_params_
            result['mean_cv_score'] = search.best_score_
            result['best_model'] = search.best_estimator_

        else:
            # Train without hyperparameter tuning
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=self.cv,
                scoring=self.config.SCORING,
                n_jobs=-1
            )
            result['cv_scores'] = cv_scores
            result['mean_cv_score'] = cv_scores.mean()
            result['std_cv_score'] = cv_scores.std()

            model.fit(X_train, y_train)
            result['best_model'] = model

        return result

    def _count_combinations(self, params: dict) -> int:
        """Count total number of hyperparameter combinations."""
        count = 1
        for values in params.values():
            count *= len(values)
        return count

    def train_all_models(self, X_train: pd.DataFrame, y_train: np.ndarray,
                         tune_hyperparams: bool = True) -> dict:
        """Train all models and return results."""

        print("\n" + "=" * 60)
        print("MODEL TRAINING")
        print("=" * 60)
        print(f"Training {len(self.models)} models with {self.config.CV_FOLDS}-fold CV")
        print(f"Hyperparameter tuning: {'Enabled' if tune_hyperparams else 'Disabled'}")
        print("=" * 60)

        # Handle class imbalance
        X_resampled, y_resampled = self.handle_imbalance(X_train, y_train)

        for name, config in self.models.items():
            print(f"\n▶ Training {name}...", end=" ")

            try:
                result = self.train_model(
                    name,
                    config['model'],
                    config['params'],
                    X_resampled,
                    y_resampled,
                    tune_hyperparams
                )
                self.results[name] = result
                self.best_models[name] = result['best_model']
                print(f"✓ CV Score: {result['mean_cv_score']:.4f}")

            except Exception as e:
                print(f"✗ Error: {str(e)[:50]}")
                self.results[name] = {'error': str(e)}

        return self.results

    def evaluate_models(self, X_test: pd.DataFrame, y_test: np.ndarray) -> pd.DataFrame:
        """Evaluate all trained models on test set."""

        print("\n" + "=" * 60)
        print("MODEL EVALUATION ON TEST SET")
        print("=" * 60)

        evaluation_results = []

        for name, model in self.best_models.items():
            if model is None:
                continue

            try:
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

                metrics = {
                    'Model': name,
                    'Accuracy': accuracy_score(y_test, y_pred),
                    'Precision': precision_score(y_test, y_pred, average='weighted'),
                    'Recall': recall_score(y_test, y_pred, average='weighted'),
                    'F1-Score': f1_score(y_test, y_pred, average='weighted'),
                }

                # ROC-AUC for multi-class
                if y_proba is not None:
                    try:
                        metrics['ROC-AUC'] = roc_auc_score(
                            y_test, y_proba, multi_class='ovr', average='weighted'
                        )
                    except:
                        metrics['ROC-AUC'] = 0

                evaluation_results.append(metrics)

            except Exception as e:
                print(f"  Error evaluating {name}: {str(e)[:50]}")

        # Create DataFrame and sort by F1-Score
        df_results = pd.DataFrame(evaluation_results)
        df_results = df_results.sort_values('F1-Score', ascending=False)

        print("\n" + df_results.to_string(index=False))

        return df_results

    def get_best_model(self) -> tuple:
        """Return the best performing model."""
        if not self.results:
            return None, None

        best_name = max(
            self.results,
            key=lambda x: self.results[x].get('mean_cv_score', 0)
        )

        return best_name, self.best_models.get(best_name)

    def save_models(self, save_all: bool = False):
        """Save trained models to disk."""
        print("\n" + "=" * 60)
        print("SAVING MODELS")
        print("=" * 60)

        os.makedirs(self.config.MODEL_SAVE_PATH, exist_ok=True)

        if save_all:
            for name, model in self.best_models.items():
                if model is not None:
                    filename = name.lower().replace(' ', '_') + '.joblib'
                    filepath = os.path.join(self.config.MODEL_SAVE_PATH, filename)
                    joblib.dump(model, filepath)
                    print(f"  ✓ Saved {name} to {filepath}")
        else:
            # Save only the best model
            best_name, best_model = self.get_best_model()
            if best_model is not None:
                filepath = os.path.join(self.config.MODEL_SAVE_PATH, 'best_model.joblib')
                joblib.dump(best_model, filepath)
                print(f"  ✓ Saved {best_name} as best_model.joblib")

                # Save model info
                info = {
                    'model_name': best_name,
                    'cv_score': self.results[best_name]['mean_cv_score'],
                    'best_params': self.results[best_name].get('best_params', {}),
                    'trained_at': datetime.now().isoformat()
                }
                joblib.dump(info, os.path.join(self.config.MODEL_SAVE_PATH, 'model_info.joblib'))


# ============================================================================
# ENSEMBLE MODEL
# ============================================================================

class EnsembleBuilder:
    """Builds ensemble models from trained base models."""

    def __init__(self, models: dict):
        self.models = models

    def build_voting_classifier(self, X_train, y_train, voting='soft') -> VotingClassifier:
        """Build a voting classifier from top models."""
        print("\n" + "=" * 60)
        print("BUILDING ENSEMBLE MODEL")
        print("=" * 60)

        # Select top 3 models
        estimators = [
            ('rf', self.models.get('Random Forest')),
            ('xgb', self.models.get('XGBoost')),
            ('gb', self.models.get('Gradient Boosting'))
        ]

        # Filter out None models
        estimators = [(name, model) for name, model in estimators if model is not None]

        if len(estimators) < 2:
            print("  ✗ Not enough models for ensemble")
            return None

        ensemble = VotingClassifier(estimators=estimators, voting=voting, n_jobs=-1)
        ensemble.fit(X_train, y_train)

        print(f"  ✓ Built Voting Classifier with {len(estimators)} models")
        return ensemble


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class ModelPipeline:
    """Main pipeline for model training and evaluation."""

    def __init__(self):
        self.config = ModelConfig()
        self.trainer = ModelTrainer(self.config)
        self.results_df = None
        self.best_model = None
        self.best_model_name = None

    def run(self, tune_hyperparams: bool = True) -> pd.DataFrame:
        """Execute the full model training pipeline."""
        print("\n" + "=" * 60)
        print("AMAZON SUPPLY CHAIN INTELLIGENCE")
        print("Model Training Pipeline")
        print("=" * 60)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Load data
        loader = ModelDataLoader(self.config)
        X_train, X_test, y_train, y_test = loader.load()

        # Train models
        self.trainer.train_all_models(X_train, y_train, tune_hyperparams)

        # Evaluate models
        self.results_df = self.trainer.evaluate_models(X_test, y_test)

        # Get best model
        self.best_model_name, self.best_model = self.trainer.get_best_model()

        print("\n" + "=" * 60)
        print(f"BEST MODEL: {self.best_model_name}")
        print(f"CV Score: {self.trainer.results[self.best_model_name]['mean_cv_score']:.4f}")
        print("=" * 60)

        # Print best model's classification report
        y_pred = self.best_model.predict(X_test)
        print("\nClassification Report (Best Model):")
        print(classification_report(y_test, y_pred, target_names=self.config.TARGET_NAMES))

        # Build ensemble
        smote = SMOTE(random_state=self.config.RANDOM_STATE)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

        ensemble_builder = EnsembleBuilder(self.trainer.best_models)
        ensemble = ensemble_builder.build_voting_classifier(X_resampled, y_resampled)

        if ensemble:
            y_pred_ens = ensemble.predict(X_test)
            ens_f1 = f1_score(y_test, y_pred_ens, average='weighted')
            print(f"\nEnsemble F1-Score: {ens_f1:.4f}")

            # If ensemble is better, use it
            if ens_f1 > self.results_df['F1-Score'].max():
                self.best_model = ensemble
                self.best_model_name = "Voting Ensemble"
                print("✓ Ensemble is the best model!")

        # Save models
        self.trainer.save_models(save_all=True)

        # Save results
        os.makedirs(self.config.REPORTS_PATH, exist_ok=True)
        self.results_df.to_csv(
            os.path.join(self.config.REPORTS_PATH, 'model_comparison.csv'),
            index=False
        )

        print("\n" + "=" * 60)
        print("MODEL TRAINING COMPLETE!")
        print("=" * 60)
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return self.results_df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    pipeline = ModelPipeline()
    results = pipeline.run(tune_hyperparams=True)

    print("\n" + "=" * 60)
    print("FINAL MODEL RANKING")
    print("=" * 60)
    print(results.to_string(index=False))
