"""
Amazon Supply Chain Intelligence
Model Evaluation Module
========================

This module handles comprehensive model evaluation including:
- Performance metrics calculation
- Visualization generation
- Feature importance analysis
- Model comparison charts
- Learning curves

Author: ARUN AK
Course: UCS321 - AI for Engineers
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from datetime import datetime

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import learning_curve

import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


# ============================================================================
# CONFIGURATION
# ============================================================================

class EvalConfig:
    """Configuration for evaluation."""

    PROCESSED_DATA_PATH = "data/processed/"
    MODEL_PATH = "models/"
    FIGURES_PATH = "reports/figures/"
    REPORTS_PATH = "reports/"

    TARGET_NAMES = ['On-Time', 'At-Risk', 'Delayed']
    TARGET_COLORS = ['#2ecc71', '#f39c12', '#e74c3c']  # Green, Orange, Red

    FIGURE_DPI = 150
    FIGURE_SIZE = (10, 8)


# ============================================================================
# VISUALIZATION GENERATOR
# ============================================================================

class VisualizationGenerator:
    """Generates all evaluation visualizations."""

    def __init__(self, config: EvalConfig = None):
        self.config = config or EvalConfig()
        os.makedirs(self.config.FIGURES_PATH, exist_ok=True)

    def plot_confusion_matrix(self, y_true, y_pred, model_name: str = "Best Model"):
        """Generate confusion matrix heatmap."""
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Raw counts
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.config.TARGET_NAMES,
            yticklabels=self.config.TARGET_NAMES,
            ax=axes[0]
        )
        axes[0].set_title(f'Confusion Matrix - {model_name}\n(Raw Counts)', fontsize=12)
        axes[0].set_ylabel('Actual', fontsize=10)
        axes[0].set_xlabel('Predicted', fontsize=10)

        # Normalized
        sns.heatmap(
            cm_normalized, annot=True, fmt='.2%', cmap='Blues',
            xticklabels=self.config.TARGET_NAMES,
            yticklabels=self.config.TARGET_NAMES,
            ax=axes[1]
        )
        axes[1].set_title(f'Confusion Matrix - {model_name}\n(Normalized)', fontsize=12)
        axes[1].set_ylabel('Actual', fontsize=10)
        axes[1].set_xlabel('Predicted', fontsize=10)

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.config.FIGURES_PATH, 'confusion_matrix.png'),
            dpi=self.config.FIGURE_DPI, bbox_inches='tight'
        )
        plt.close()
        print("  ✓ Saved confusion_matrix.png")

    def plot_roc_curves(self, y_true, y_proba, model_name: str = "Best Model"):
        """Generate ROC curves for multi-class classification."""
        n_classes = len(self.config.TARGET_NAMES)
        y_true_bin = label_binarize(y_true, classes=[0, 1, 2])

        fig, ax = plt.subplots(figsize=self.config.FIGURE_SIZE)

        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)

            ax.plot(
                fpr, tpr,
                color=self.config.TARGET_COLORS[i],
                lw=2,
                label=f'{self.config.TARGET_NAMES[i]} (AUC = {roc_auc:.3f})'
            )

        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'ROC Curves - {model_name}', fontsize=14)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.config.FIGURES_PATH, 'roc_curves.png'),
            dpi=self.config.FIGURE_DPI, bbox_inches='tight'
        )
        plt.close()
        print("  ✓ Saved roc_curves.png")

    def plot_precision_recall_curves(self, y_true, y_proba, model_name: str = "Best Model"):
        """Generate Precision-Recall curves."""
        n_classes = len(self.config.TARGET_NAMES)
        y_true_bin = label_binarize(y_true, classes=[0, 1, 2])

        fig, ax = plt.subplots(figsize=self.config.FIGURE_SIZE)

        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_proba[:, i])
            ap = average_precision_score(y_true_bin[:, i], y_proba[:, i])

            ax.plot(
                recall, precision,
                color=self.config.TARGET_COLORS[i],
                lw=2,
                label=f'{self.config.TARGET_NAMES[i]} (AP = {ap:.3f})'
            )

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(f'Precision-Recall Curves - {model_name}', fontsize=14)
        ax.legend(loc='lower left', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.config.FIGURES_PATH, 'precision_recall_curves.png'),
            dpi=self.config.FIGURE_DPI, bbox_inches='tight'
        )
        plt.close()
        print("  ✓ Saved precision_recall_curves.png")

    def plot_feature_importance(self, model, feature_names: list, top_n: int = 15):
        """Generate feature importance plot."""
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_).mean(axis=0)
        else:
            print("  ✗ Model doesn't support feature importance")
            return

        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=True).tail(top_n)

        fig, ax = plt.subplots(figsize=(10, 8))

        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(importance_df)))

        ax.barh(
            importance_df['Feature'],
            importance_df['Importance'],
            color=colors
        )

        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.set_title(f'Top {top_n} Feature Importance', fontsize=14)

        # Add value labels
        for i, (idx, row) in enumerate(importance_df.iterrows()):
            ax.text(
                row['Importance'] + 0.001,
                i,
                f'{row["Importance"]:.4f}',
                va='center',
                fontsize=9
            )

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.config.FIGURES_PATH, 'feature_importance.png'),
            dpi=self.config.FIGURE_DPI, bbox_inches='tight'
        )
        plt.close()
        print("  ✓ Saved feature_importance.png")

        return importance_df

    def plot_model_comparison(self, results_df: pd.DataFrame):
        """Generate model comparison bar chart."""
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        colors = sns.color_palette("husl", len(results_df))

        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            bars = ax.bar(
                range(len(results_df)),
                results_df[metric],
                color=colors
            )

            ax.set_xticks(range(len(results_df)))
            ax.set_xticklabels(results_df['Model'], rotation=45, ha='right', fontsize=9)
            ax.set_ylabel(metric, fontsize=11)
            ax.set_title(f'{metric} Comparison', fontsize=12)
            ax.set_ylim([0, 1.1])

            # Add value labels
            for bar, val in zip(bars, results_df[metric]):
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.02,
                    f'{val:.3f}',
                    ha='center',
                    va='bottom',
                    fontsize=8
                )

        plt.suptitle('Model Performance Comparison', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.config.FIGURES_PATH, 'model_comparison.png'),
            dpi=self.config.FIGURE_DPI, bbox_inches='tight'
        )
        plt.close()
        print("  ✓ Saved model_comparison.png")

    def plot_class_distribution(self, y_train, y_test):
        """Plot class distribution in train and test sets."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for idx, (data, title) in enumerate([(y_train, 'Training Set'), (y_test, 'Test Set')]):
            unique, counts = np.unique(data, return_counts=True)
            labels = [self.config.TARGET_NAMES[i] for i in unique]

            wedges, texts, autotexts = axes[idx].pie(
                counts,
                labels=labels,
                autopct='%1.1f%%',
                colors=self.config.TARGET_COLORS,
                explode=[0.02] * len(unique),
                shadow=True
            )

            axes[idx].set_title(f'Class Distribution - {title}', fontsize=12)

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.config.FIGURES_PATH, 'class_distribution.png'),
            dpi=self.config.FIGURE_DPI, bbox_inches='tight'
        )
        plt.close()
        print("  ✓ Saved class_distribution.png")

    def plot_learning_curves(self, model, X, y, model_name: str = "Best Model"):
        """Generate learning curves to analyze bias-variance tradeoff."""
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y,
            train_sizes=np.linspace(0.1, 1.0, 10),
            cv=5,
            scoring='f1_weighted',
            n_jobs=-1
        )

        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)

        fig, ax = plt.subplots(figsize=self.config.FIGURE_SIZE)

        ax.fill_between(
            train_sizes, train_mean - train_std, train_mean + train_std,
            alpha=0.2, color='#2ecc71'
        )
        ax.fill_between(
            train_sizes, val_mean - val_std, val_mean + val_std,
            alpha=0.2, color='#e74c3c'
        )

        ax.plot(train_sizes, train_mean, 'o-', color='#2ecc71', label='Training Score')
        ax.plot(train_sizes, val_mean, 'o-', color='#e74c3c', label='Validation Score')

        ax.set_xlabel('Training Set Size', fontsize=12)
        ax.set_ylabel('F1 Score (Weighted)', fontsize=12)
        ax.set_title(f'Learning Curves - {model_name}', fontsize=14)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.config.FIGURES_PATH, 'learning_curves.png'),
            dpi=self.config.FIGURE_DPI, bbox_inches='tight'
        )
        plt.close()
        print("  ✓ Saved learning_curves.png")

    def plot_radar_chart(self, results_df: pd.DataFrame, top_n: int = 5):
        """Generate radar chart for model comparison."""
        # Select top N models
        df = results_df.head(top_n).copy()
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

        # Number of metrics
        num_vars = len(metrics)

        # Compute angle for each metric
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        colors = plt.cm.Set2(np.linspace(0, 1, len(df)))

        for idx, (_, row) in enumerate(df.iterrows()):
            values = row[metrics].values.tolist()
            values += values[:1]  # Complete the circle

            ax.plot(angles, values, 'o-', linewidth=2, label=row['Model'], color=colors[idx])
            ax.fill(angles, values, alpha=0.1, color=colors[idx])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=11)
        ax.set_ylim([0, 1])

        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)
        plt.title('Model Performance Radar Chart', fontsize=14, y=1.1)

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.config.FIGURES_PATH, 'radar_chart.png'),
            dpi=self.config.FIGURE_DPI, bbox_inches='tight'
        )
        plt.close()
        print("  ✓ Saved radar_chart.png")


# ============================================================================
# METRICS CALCULATOR
# ============================================================================

class MetricsCalculator:
    """Calculates and reports all evaluation metrics."""

    def __init__(self, config: EvalConfig = None):
        self.config = config or EvalConfig()

    def calculate_all_metrics(self, y_true, y_pred, y_proba=None) -> dict:
        """Calculate comprehensive metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
        }

        # Per-class metrics
        for i, name in enumerate(self.config.TARGET_NAMES):
            y_true_binary = (y_true == i).astype(int)
            y_pred_binary = (y_pred == i).astype(int)

            metrics[f'precision_{name.lower().replace("-", "_")}'] = precision_score(
                y_true_binary, y_pred_binary, zero_division=0
            )
            metrics[f'recall_{name.lower().replace("-", "_")}'] = recall_score(
                y_true_binary, y_pred_binary, zero_division=0
            )
            metrics[f'f1_{name.lower().replace("-", "_")}'] = f1_score(
                y_true_binary, y_pred_binary, zero_division=0
            )

        # ROC-AUC if probabilities available
        if y_proba is not None:
            try:
                y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
                for i, name in enumerate(self.config.TARGET_NAMES):
                    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
                    metrics[f'auc_{name.lower().replace("-", "_")}'] = auc(fpr, tpr)
            except:
                pass

        return metrics

    def print_classification_report(self, y_true, y_pred):
        """Print formatted classification report."""
        print("\n" + "=" * 60)
        print("CLASSIFICATION REPORT")
        print("=" * 60)
        print(classification_report(
            y_true, y_pred,
            target_names=self.config.TARGET_NAMES,
            digits=4
        ))


# ============================================================================
# MAIN EVALUATION PIPELINE
# ============================================================================

class EvaluationPipeline:
    """Main pipeline for model evaluation."""

    def __init__(self):
        self.config = EvalConfig()
        self.viz = VisualizationGenerator(self.config)
        self.metrics_calc = MetricsCalculator(self.config)

    def run(self):
        """Execute the full evaluation pipeline."""
        print("\n" + "=" * 60)
        print("AMAZON SUPPLY CHAIN INTELLIGENCE")
        print("Model Evaluation Pipeline")
        print("=" * 60)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Load data
        print("\n▶ Loading data...")
        X_train = pd.read_csv(f"{self.config.PROCESSED_DATA_PATH}X_train.csv")
        X_test = pd.read_csv(f"{self.config.PROCESSED_DATA_PATH}X_test.csv")
        y_train = pd.read_csv(f"{self.config.PROCESSED_DATA_PATH}y_train.csv").values.ravel()
        y_test = pd.read_csv(f"{self.config.PROCESSED_DATA_PATH}y_test.csv").values.ravel()

        # Load best model
        print("▶ Loading best model...")
        model = joblib.load(f"{self.config.MODEL_PATH}best_model.joblib")
        model_info = joblib.load(f"{self.config.MODEL_PATH}model_info.joblib")
        model_name = model_info.get('model_name', 'Best Model')
        print(f"  ✓ Loaded: {model_name}")

        # Load model comparison results
        results_df = pd.read_csv(f"{self.config.REPORTS_PATH}model_comparison.csv")

        # Make predictions
        print("\n▶ Making predictions...")
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

        # Calculate metrics
        print("\n▶ Calculating metrics...")
        metrics = self.metrics_calc.calculate_all_metrics(y_test, y_pred, y_proba)
        self.metrics_calc.print_classification_report(y_test, y_pred)

        # Generate visualizations
        print("\n▶ Generating visualizations...")
        print("=" * 60)

        # 1. Confusion Matrix
        self.viz.plot_confusion_matrix(y_test, y_pred, model_name)

        # 2. ROC Curves
        if y_proba is not None:
            self.viz.plot_roc_curves(y_test, y_proba, model_name)
            self.viz.plot_precision_recall_curves(y_test, y_proba, model_name)

        # 3. Feature Importance
        feature_names = X_test.columns.tolist()
        importance_df = self.viz.plot_feature_importance(model, feature_names)

        # 4. Model Comparison
        self.viz.plot_model_comparison(results_df)

        # 5. Class Distribution
        self.viz.plot_class_distribution(y_train, y_test)

        # 6. Radar Chart
        self.viz.plot_radar_chart(results_df)

        # 7. Learning Curves (on sample for speed)
        print("  → Generating learning curves (this may take a moment)...")
        sample_size = min(5000, len(X_train))
        X_sample = X_train.sample(sample_size, random_state=42)
        y_sample = y_train[X_sample.index]
        self.viz.plot_learning_curves(model, X_sample, y_sample, model_name)

        # Save metrics
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(
            os.path.join(self.config.REPORTS_PATH, 'evaluation_metrics.csv'),
            index=False
        )
        print(f"\n✓ Saved metrics to {self.config.REPORTS_PATH}evaluation_metrics.csv")

        # Print summary
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Best Model: {model_name}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1-Score (Weighted): {metrics['f1_weighted']:.4f}")
        print(f"Precision (Weighted): {metrics['precision_weighted']:.4f}")
        print(f"Recall (Weighted): {metrics['recall_weighted']:.4f}")

        if importance_df is not None:
            print(f"\nTop 3 Features:")
            for _, row in importance_df.tail(3).iloc[::-1].iterrows():
                print(f"  • {row['Feature']}: {row['Importance']:.4f}")

        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE!")
        print("=" * 60)
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nAll visualizations saved to: {self.config.FIGURES_PATH}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    pipeline = EvaluationPipeline()
    pipeline.run()
