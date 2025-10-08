import time


import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
import numpy as np
from sklearn.preprocessing import label_binarize

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    average_precision_score,
)
import matplotlib.pyplot as plt
import seaborn as sns


from src.data.data_process import run_optimized_pipeline


import warnings

warnings.filterwarnings("ignore")


# New ML Benchmarking System
class IDSModelBenchmark:
    """
    Comprehensive ML model benchmarking system for IDS in IIoT environments
    """

    def __init__(self, data_dict):
        """Initialize with preprocessed data"""
        self.data = data_dict
        self.models = {}
        self.results = {"binary": {}, "multi": {}}
        self.training_times = {"binary": {}, "multi": {}}

        # Initialize models with IDS-optimized hyperparameters from literature
        self._initialize_models()

    def _initialize_models(self):
        """Initialize ML models with hyperparameters optimized for IDS"""
        self.model_configs = {
            "Naive Bayes": {"binary": GaussianNB(), "multi": GaussianNB()},
            "Random Forest": {
                # Based on IDS literature: 100-200 estimators, max_depth to prevent overfitting
                "binary": RandomForestClassifier(
                    n_estimators=150,
                    max_depth=20,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
                "multi": RandomForestClassifier(
                    n_estimators=150,
                    max_depth=20,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            },
            "Decision Tree": {
                # Pruned trees to prevent overfitting in IDS
                "binary": DecisionTreeClassifier(
                    max_depth=15,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=42,
                ),
                "multi": DecisionTreeClassifier(
                    max_depth=15,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=42,
                ),
            },
            "SVM": {
                # enable probabilities for PR/AUPRC
                "binary": SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    probability=True,
                    random_state=42,
                ),
                "multi": SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    probability=True,
                    random_state=42,
                ),
            },
        }

    def _predict_scores(self, model, X, y_true, task_type):
        """
        Returns score arrays suitable for PR/AUPRC:
          - binary: shape (n_samples,), positive-class probabilities
          - multi : shape (n_samples, n_classes), per-class probabilities
        """
        # Prefer predict_proba if available
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            if task_type == "binary":
                # take probability of positive class (assumes classes are [0,1] or [neg,pos])
                if proba.shape[1] == 2:
                    return proba[:, 1]
                # rare edge case: single column
                return proba.ravel()
            else:
                return proba

        # Fallback to decision_function -> map to pseudo-probabilities
        if hasattr(model, "decision_function"):
            df = model.decision_function(X)
            if task_type == "binary":
                # sigmoid to map scores to (0,1)
                return 1 / (1 + np.exp(-df))
            else:
                # softmax across classes
                # df shape (n_samples, n_classes)
                e = np.exp(df - np.max(df, axis=1, keepdims=True))
                return e / np.sum(e, axis=1, keepdims=True)

        # If neither, fallback to one-hot of predictions (not ideal for PR but safe)
        y_pred = model.predict(X)
        if task_type == "binary":
            return (y_pred == 1).astype(float)
        else:
            classes = np.unique(y_true)
            y_bin = label_binarize(y_pred, classes=classes)
            return y_bin

    def calculate_metrics(self, y_true, y_pred, y_score, task_type):
        """
        Calculate evaluation metrics.
        - y_score: binary -> (n,), multi -> (n, n_classes) probabilities/scores
        """
        avg_type = "binary" if task_type == "binary" else "weighted"

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(
                y_true, y_pred, average=avg_type, zero_division=0
            ),
            "recall": recall_score(y_true, y_pred, average=avg_type, zero_division=0),
            "f1": f1_score(y_true, y_pred, average=avg_type, zero_division=0),
            # new macro variants
            "macro_recall": recall_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        }

        # AUPRC
        if task_type == "binary":
            # y_score shape (n,)
            try:
                metrics["auprc"] = average_precision_score(y_true, y_score)
            except Exception:
                metrics["auprc"] = np.nan
        else:
            # multi-class macro AUPRC (one-vs-rest)
            classes = np.unique(y_true)
            y_true_bin = label_binarize(y_true, classes=classes)
            try:
                metrics["auprc"] = average_precision_score(
                    y_true_bin, y_score, average="macro"
                )
            except Exception:
                metrics["auprc"] = np.nan

        return metrics

    def train_and_evaluate_model(self, model_name, task_type):
        """Train and evaluate a single model"""
        print(f"Training {model_name} for {task_type} classification...")

        # Get data
        data = self.data[task_type]
        X_train, X_val, X_test = data["X_train"], data["X_val"], data["X_test"]
        y_train, y_val, y_test = data["y_train"], data["y_val"], data["y_test"]

        # Get model
        model = self.model_configs[model_name][task_type]

        # Train model and measure time
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time

        # Store training time
        self.training_times[task_type][model_name] = training_time

        # Predictions (labels)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)

        # Scores (probabilities for PR/AUPRC)
        y_val_score = self._predict_scores(model, X_val, y_val, task_type)
        y_test_score = self._predict_scores(model, X_test, y_test, task_type)

        # Calculate metrics
        val_metrics = self.calculate_metrics(y_val, y_val_pred, y_val_score, task_type)
        test_metrics = self.calculate_metrics(
            y_test, y_test_pred, y_test_score, task_type
        )

        # Store results
        self.results[task_type][model_name] = {
            "validation": val_metrics,
            "test": test_metrics,
            "val_predictions": y_val_pred,
            "test_predictions": y_test_pred,
            "val_scores": y_val_score,
            "test_scores": y_test_score,
            "val_confusion_matrix": confusion_matrix(y_val, y_val_pred),
            "test_confusion_matrix": confusion_matrix(y_test, y_test_pred),
            "model": model,
            "training_time": training_time,
        }

        print(f"  Training time: {training_time:.2f} seconds")
        print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  Test F1-Score: {test_metrics['f1']:.4f}")
        print(f"  Test Macro-F1: {test_metrics['macro_f1']:.4f}")
        print(f"  Test AUPRC: {test_metrics['auprc']:.4f}")

    def run_benchmark(self):
        """Run complete benchmark for all models"""
        print("=" * 60)
        print("STARTING IDS MODEL BENCHMARKING")
        print("=" * 60)

        for task_type in ["multi"]:
            print(f"\n{'-'*40}")
            print(f"TASK: {task_type.upper()} CLASSIFICATION")
            print(f"{'-'*40}")

            for model_name in self.model_configs.keys():
                try:
                    self.train_and_evaluate_model(model_name, task_type)
                except Exception as e:
                    print(f"Error training {model_name}: {str(e)}")

        print("\n" + "=" * 60)
        print("BENCHMARKING COMPLETED")
        print("=" * 60)

    def plot_confusion_matrices(
        self, save_figures=False, output_dir="confusion_matrices"
    ):
        """Plot confusion matrices with different layouts for binary and multi-class"""
        # Class labels
        binary_labels = ["Attack", "Normal"]
        multi_labels = [
            "BruteForce",
            "C&C",
            "Dictionary",
            "Discovering_resources",
            "Exfiltration",
            "Fake_notification",
            "False_data_injection",
            "Generic_scanning",
            "MQTT_cloud_broker_subscription",
            "MitM",
            "Modbus_register_reading",
            "Normal",
            "RDOS",
            "Reverse_shell",
            "Scanning_vulnerability",
            "TCP Relay",
            "crypto-ransomware",
            "fuzzing",
            "insider_malcious",
        ]

        if save_figures:
            import os

            os.makedirs(output_dir, exist_ok=True)

        tasks = ["binary", "multi"]

        for task in tasks:
            class_labels = binary_labels if task == "binary" else multi_labels
            model_names = list(self.results[task].keys())

            if task == "binary":
                # Binary classification - 4 models per figure
                batch_size = 4
                for batch_start in range(0, len(model_names), batch_size):
                    batch_end = min(batch_start + batch_size, len(model_names))
                    batch_models = model_names[batch_start:batch_end]

                    fig, axes = plt.subplots(
                        2, len(batch_models), figsize=(4 * len(batch_models), 10)
                    )
                    if len(batch_models) == 1:
                        axes = axes.reshape(2, 1)

                    fig.suptitle(
                        f"Binary Classification - Models {batch_start+1}-{batch_end}",
                        fontsize=16,
                        y=1.02,
                    )

                    for idx, model_name in enumerate(batch_models):
                        # Validation matrix
                        self._plot_single_confusion_matrix(
                            self.results[task][model_name]["val_confusion_matrix"],
                            axes[0, idx],
                            model_name,
                            "Validation",
                            class_labels,
                        )
                        # Test matrix
                        self._plot_single_confusion_matrix(
                            self.results[task][model_name]["test_confusion_matrix"],
                            axes[1, idx],
                            model_name,
                            "Test",
                            class_labels,
                        )

                    plt.tight_layout(pad=3.0)
                    if save_figures:
                        plt.savefig(
                            f"{output_dir}/binary_confusion_{batch_start//batch_size+1}.png",
                            dpi=300,
                            bbox_inches="tight",
                        )
                        plt.close()
                    else:
                        plt.show()

            else:
                # Multi-class - individual figures per model
                for model_name in model_names:
                    fig, axes = plt.subplots(1, 2, figsize=(24, 12))

                    # Validation matrix
                    self._plot_single_confusion_matrix(
                        self.results[task][model_name]["val_confusion_matrix"],
                        axes[0],
                        model_name,
                        "Validation",
                        class_labels,
                        annot_size=8,
                    )
                    # Test matrix
                    self._plot_single_confusion_matrix(
                        self.results[task][model_name]["test_confusion_matrix"],
                        axes[1],
                        model_name,
                        "Test",
                        class_labels,
                        annot_size=8,
                    )

                    plt.suptitle(
                        f"Multi-class Classification - {model_name}",
                        fontsize=16,
                        y=1.02,
                    )
                    plt.tight_layout(pad=4.0)
                    if save_figures:
                        plt.savefig(
                            f"{output_dir}/multi_confusion_{model_name}.png",
                            dpi=300,
                            bbox_inches="tight",
                        )
                        plt.close()
                    else:
                        plt.show()

    def _plot_single_confusion_matrix(
        self, cm, ax, model_name, matrix_type, class_labels, annot_size=12
    ):
        """Helper function to plot a single confusion matrix"""
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            annot_kws={"size": annot_size},
            xticklabels=class_labels,
            yticklabels=class_labels,
        )
        ax.set_title(f"{model_name} - {matrix_type}", pad=20)
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_xlabel("Predicted Label", fontsize=12)

        if len(class_labels) > 2:  # Multi-class
            plt.setp(
                ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
            )
            plt.setp(
                ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor"
            )

    def plot_performance_comparison(self):
        """
        Bar-chart comparison for ALL available metrics in self.results
        across Validation and Test, for both binary and multi tasks.
        Dynamically adapts to whatever metrics you computed (e.g., accuracy,
        precision, recall, f1, macro_recall, macro_f1, auprc).
        """
        import numpy as np
        import matplotlib.pyplot as plt

        tasks = ["binary", "multi"]

        # Discover the union of metric keys present in results
        def discover_metrics(task):
            if not self.results[task]:
                return []
            # take metrics from the first model found
            first_model = next(iter(self.results[task].values()))
            return list(first_model["validation"].keys())

        # union across tasks (preserves order from 'binary' when possible)
        metric_keys = []
        for t in tasks:
            for m in discover_metrics(t):
                if m not in metric_keys:
                    metric_keys.append(m)

        if not metric_keys:
            print("No metrics found to plot.")
            return

        # layout
        n_cols = len(metric_keys)
        n_rows = len(tasks)
        # scale figure width with number of metrics and models
        n_models_max = max(len(self.results[t]) for t in tasks) or 1
        fig_w = max(10, 2 + 1.2 * n_cols + 0.5 * n_models_max)
        fig_h = max(6, 4.5 * n_rows)

        fig, axs = plt.subplots(
            n_rows, n_cols, figsize=(fig_w, fig_h), constrained_layout=True
        )
        if n_rows == 1 and n_cols == 1:
            axs = np.array([[axs]])
        elif n_rows == 1:
            axs = axs[np.newaxis, :]
        elif n_cols == 1:
            axs = axs[:, np.newaxis]

        for r, task in enumerate(tasks):
            models = list(self.results[task].keys())
            if not models:
                # no models for this task—render empty panels
                for c, metric in enumerate(metric_keys):
                    axs[r, c].set_title(f"{task.title()} - {metric.title()}")
                    axs[r, c].text(0.5, 0.5, "No results", ha="center", va="center")
                    axs[r, c].axis("off")
                continue

            x = np.arange(len(models))
            width = 0.38

            for c, metric in enumerate(metric_keys):
                ax = axs[r, c]

                # Pull values; if a metric is missing, use NaN so it shows as empty
                val_scores = []
                test_scores = []
                for m in models:
                    v = self.results[task][m]["validation"].get(metric, np.nan)
                    t_ = self.results[task][m]["test"].get(metric, np.nan)
                    val_scores.append(v)
                    test_scores.append(t_)

                bars1 = ax.bar(
                    x - width / 2, val_scores, width, label="Validation", alpha=0.85
                )
                bars2 = ax.bar(
                    x + width / 2, test_scores, width, label="Test", alpha=0.85
                )

                ax.set_title(f"{task.title()} - {metric.replace('_',' ').title()}")
                ax.set_ylabel(metric.replace("_", " ").title())
                ax.set_xlabel("Models")
                ax.set_xticks(x)
                ax.set_xticklabels(models, rotation=55, ha="right")
                ax.grid(True, axis="y", alpha=0.25)

                # add value labels (skip NaNs)
                for bar in list(bars1) + list(bars2):
                    h = bar.get_height()
                    if not (isinstance(h, float) and np.isnan(h)):
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            h + 0.01,
                            f"{h:.3f}",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                        )

                if r == 0 and c == 0:
                    ax.legend()

        plt.show()

    def plot_training_times(self):
        """Plot training time comparison"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        tasks = ["binary", "multi"]

        for idx, task in enumerate(tasks):
            ax = ax1 if idx == 0 else ax2

            models = list(self.training_times[task].keys())
            times = list(self.training_times[task].values())

            bars = ax.bar(
                models,
                times,
                alpha=0.7,
                color=["skyblue", "lightgreen", "lightcoral", "gold"],
            )
            ax.set_title(f"{task.title()} Classification - Training Times")
            ax.set_ylabel("Training Time (seconds)")
            ax.set_xlabel("Models")
            ax.tick_params(axis="x", rotation=45)

            # Add value labels on bars
            for bar, time_val in zip(bars, times):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{time_val:.2f}s",
                    ha="center",
                    va="bottom",
                )

        plt.tight_layout()
        plt.show()

    def plot_per_class_confusion_matrix(
        self, y_true, y_pred, model_variant, is_binary=True
    ):
        """Plots normalized per-class confusion matrix with class names."""
        model_type = "Binary" if is_binary else "Multi-class"

        if is_binary:
            class_labels = ["Attack", "Normal"]
        else:
            class_labels = [
                "BruteForce",
                "C&C",
                "Dictionary",
                "Discovering_resources",
                "Exfiltration",
                "Fake_notification",
                "False_data_injection",
                "Generic_scanning",
                "MQTT_cloud_broker_subscription",
                "MitM",
                "Modbus_register_reading",
                "Normal",
                "RDOS",
                "Reverse_shell",
                "Scanning_vulnerability",
                "TCP Relay",
                "crypto-ransomware",
                "fuzzing",
                "insider_malcious",
            ]

        cm = confusion_matrix(y_true, y_pred, normalize="true")

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=class_labels,
            yticklabels=class_labels,
        )
        plt.title(
            f"{model_type} Classification - Per-Class Normalized Confusion Matrix"
        )
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        output_dir = "./out/figs/confusion_matrices_per_class"
        os.makedirs(output_dir, exist_ok=True)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                output_dir,
                f"{model_variant}_{model_type.lower()}_per_class_conf_matrix.png",
            ),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def plot_pr_curve(self, y_true, y_score, model_variant, is_binary=True):
        """Plots Precision-Recall curve and AUPRC value."""
        model_type = "Binary" if is_binary else "Multi-class"

        if is_binary:
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            auprc = average_precision_score(y_true, y_score)
        else:
            # Micro-average PR using flattened one-vs-rest labels and scores
            classes = np.unique(y_true)
            y_true_bin = label_binarize(y_true, classes=classes)
            precision, recall, _ = precision_recall_curve(
                y_true_bin.ravel(), y_score.ravel()
            )
            auprc = average_precision_score(y_true_bin, y_score, average="macro")

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f"AUPRC = {auprc:.4f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"{model_type} Classification - Precision-Recall Curve")
        plt.legend()

        output_dir = "./out/figs/pr_curves"
        os.makedirs(output_dir, exist_ok=True)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                output_dir, f"{model_variant}-{model_type.lower()}_pr_curve.png"
            ),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def plot_all_per_class_confusions(self):
        """
        For each task/model, save the per-class normalized confusion matrix on the TEST split.
        """
        for task in ["binary", "multi"]:
            for model_name, res in self.results[task].items():
                y_true = self.data[task]["y_test"]
                y_pred = res["test_predictions"]
                self.plot_per_class_confusion_matrix(
                    y_true,
                    y_pred,
                    model_variant=model_name,
                    is_binary=(task == "binary"),
                )

    def plot_all_pr_curves(self):
        """
        For each task/model, save PR curve on the TEST split.
        """
        for task in ["binary", "multi"]:
            for model_name, res in self.results[task].items():
                y_true = self.data[task]["y_test"]
                y_score = res["test_scores"]
                self.plot_pr_curve(
                    y_true,
                    y_score,
                    model_variant=model_name,
                    is_binary=(task == "binary"),
                )


# Main execution function
def run_ids_benchmark(filepath):
    """
    Main function to run the complete IDS benchmarking pipeline

    Args:
        filepath (str): Path to the CSV dataset file
    """
    try:
        # Step 1: Run data preprocessing pipeline
        print("Step 1: Data Preprocessing")
        data_dict = run_optimized_pipeline(filepath)

        # Step 2: Initialize and run ML benchmark
        print("\nStep 2: Machine Learning Benchmarking")
        benchmark = IDSModelBenchmark(data_dict)
        benchmark.run_benchmark()

        # Step 3: Generate visualizations
        print("\nStep 3: Generating Visualizations")
        benchmark.plot_performance_comparison()
        benchmark.plot_confusion_matrices()
        benchmark.plot_training_times()
        benchmark.plot_all_per_class_confusions()
        benchmark.plot_all_pr_curves()

        # Step 4: Generate summary report
        print("\nStep 4: Generating Summary Report")
        benchmark.generate_summary_report()

        return benchmark

    except Exception as e:
        print(f"Error in benchmarking pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    filepath = "../data/X-IIoTID dataset.csv"

    print("Step 1: Data Preprocessing")
    data_dict = run_optimized_pipeline(filepath)

    print("\nStep 2: Machine Learning Benchmarking")
    benchmark = IDSModelBenchmark(data_dict)
    benchmark.run_benchmark()

    print("\nStep 3: Generating Visualizations")
    benchmark.plot_performance_comparison()
    benchmark.plot_confusion_matrices()
    benchmark.plot_training_times()
