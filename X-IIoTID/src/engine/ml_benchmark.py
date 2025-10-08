import time


import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
import numpy as np
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
                # RBF kernel commonly used in IDS, C tuned for balance
                "binary": SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42),
                "multi": SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42),
            },
        }

    def calculate_metrics(self, y_true, y_pred, task_type):
        """Calculate all evaluation metrics"""
        avg_type = "binary" if task_type == "binary" else "weighted"

        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(
                y_true, y_pred, average=avg_type, zero_division=0
            ),
            "recall": recall_score(y_true, y_pred, average=avg_type, zero_division=0),
            "f1": f1_score(y_true, y_pred, average=avg_type, zero_division=0),
        }

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

        # Predictions
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)

        # Calculate metrics
        val_metrics = self.calculate_metrics(y_val, y_val_pred, task_type)
        test_metrics = self.calculate_metrics(y_test, y_test_pred, task_type)

        # Store results
        self.results[task_type][model_name] = {
            "validation": val_metrics,
            "test": test_metrics,
            "val_predictions": y_val_pred,
            "test_predictions": y_test_pred,
            "val_confusion_matrix": confusion_matrix(y_val, y_val_pred),
            "test_confusion_matrix": confusion_matrix(y_test, y_test_pred),
            "model": model,
            "training_time": training_time,
        }

        print(f"  Training time: {training_time:.2f} seconds")
        print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  Test F1-Score: {test_metrics['f1']:.4f}")

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
        """Create comprehensive performance comparison plots"""
        n_models = max(len(self.results[task]) for task in self.results)
        fig, axs = plt.subplots(
            2, 4, figsize=(4 * n_models, 10), constrained_layout=True
        )

        metrics = ["accuracy", "precision", "recall", "f1"]
        tasks = ["binary", "multi"]

        for task_idx, task in enumerate(tasks):
            for metric_idx, metric in enumerate(metrics):
                ax = axs[task_idx, metric_idx]

                models = list(self.results[task].keys())
                val_scores = [
                    self.results[task][model]["validation"][metric] for model in models
                ]
                test_scores = [
                    self.results[task][model]["test"][metric] for model in models
                ]

                x = np.arange(len(models))
                width = 0.35

                bars1 = ax.bar(
                    x - width / 2, val_scores, width, label="Validation", alpha=0.8
                )
                bars2 = ax.bar(
                    x + width / 2, test_scores, width, label="Test", alpha=0.8
                )

                ax.set_title(f"{task.title()} - {metric.title()}")
                ax.set_ylabel(metric.title())
                ax.set_xlabel("Models")
                ax.set_xticks(x)
                ax.set_xticklabels(models, rotation=60, ha="right")
                ax.legend()
                ax.grid(True, alpha=0.3)

                for bar in bars1 + bars2:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + 0.01,
                        f"{height:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                    )

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
