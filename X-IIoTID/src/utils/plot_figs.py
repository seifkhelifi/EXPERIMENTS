import os

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_training_history(
    history,
    model_variant,
    is_binary=True,
):
    model_type = "Binary" if is_binary else "Multi-class"
    epochs = list(range(1, len(history["train_loss"]) + 1))

    # Create a 3x2 subplot grid for all metrics
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    fig.suptitle(f"{model_type} Classification Model - Training History", fontsize=16)

    # Plot Loss
    axes[0, 0].plot(epochs, history["train_loss"], "b-", label="Training Loss")
    axes[0, 0].plot(epochs, history["val_loss"], "r-", label="Validation Loss")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epochs")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Plot Accuracy
    axes[0, 1].plot(epochs, history["train_accuracy"], "b-", label="Training Accuracy")
    axes[0, 1].plot(epochs, history["val_accuracy"], "r-", label="Validation Accuracy")
    axes[0, 1].set_title("Accuracy")
    axes[0, 1].set_xlabel("Epochs")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Plot Precision
    axes[1, 0].plot(
        epochs, history["train_precision"], "b-", label="Training Precision"
    )
    axes[1, 0].plot(
        epochs, history["val_precision"], "r-", label="Validation Precision"
    )
    axes[1, 0].set_title("Precision")
    axes[1, 0].set_xlabel("Epochs")
    axes[1, 0].set_ylabel("Precision")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Plot Recall
    axes[1, 1].plot(epochs, history["train_recall"], "b-", label="Training Recall")
    axes[1, 1].plot(epochs, history["val_recall"], "r-", label="Validation Recall")
    axes[1, 1].set_title("Recall")
    axes[1, 1].set_xlabel("Epochs")
    axes[1, 1].set_ylabel("Recall")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    # Plot F1 Score
    axes[2, 0].plot(epochs, history["train_f1_score"], "b-", label="Training F1")
    axes[2, 0].plot(epochs, history["val_f1_score"], "r-", label="Validation F1")
    axes[2, 0].set_title("F1 Score")
    axes[2, 0].set_xlabel("Epochs")
    axes[2, 0].set_ylabel("F1 Score")
    axes[2, 0].legend()
    axes[2, 0].grid(True)

    # Make the last subplot invisible
    axes[2, 1].axis("off")

    output_dir = "figs/history"
    os.makedirs(output_dir, exist_ok=True)  # create folder if not exists

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(
        os.path.join(
            output_dir, f"{model_variant}-{model_type.lower()}_training_history.png"
        )
    )
    plt.show()

    # Function to plot confusion matrix with hardcoded labels


def plot_confusion_matrix(y_true, y_pred, model_variant, is_binary=True):
    model_type = "Binary" if is_binary else "Multi-class"

    # Hardcoded class names
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

    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=class_labels,
        yticklabels=class_labels,
    )

    plt.title(f"{model_type} Classification - Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    output_dir = "figs/confusion_matrices"
    os.makedirs(output_dir, exist_ok=True)  # create folder if it doesn't exist

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            output_dir, f"{model_variant}-{model_type.lower()}_confusion_matrix.png"
        )
    )
    plt.show()
