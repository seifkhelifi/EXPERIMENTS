import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def _to_numpy_arrays(y_true, y_pred, is_binary):
    """Convert tensors to numpy arrays based on problem type."""
    if is_binary:
        y_pred_class = (y_pred > 0.5).float()
        y_true_np = y_true.cpu().detach().numpy().flatten()
        y_pred_np = y_pred_class.cpu().detach().numpy().flatten()
        y_score_np = y_pred.cpu().detach().numpy().flatten()
    else:
        # For multi-class, get predicted class indices
        y_pred_softmax = torch.softmax(y_pred, dim=1)  # Convert logits to probabilities
        _, y_pred_class = torch.max(y_pred, dim=1)
        y_true_np = y_true.cpu().detach().numpy()
        y_pred_np = y_pred_class.cpu().detach().numpy()
        y_score_np = (
            y_pred_softmax.cpu().detach().numpy()
        )  # Use probabilities for AUPRC

    return y_true_np, y_pred_np, y_score_np


def _calculate_base_metrics(y_true_np, y_pred_np, is_binary):
    """Calculate base classification metrics."""
    avg_method = "binary" if is_binary else "macro"

    accuracy = accuracy_score(y_true_np, y_pred_np)
    precision = precision_score(
        y_true_np, y_pred_np, average=avg_method, zero_division=0
    )
    recall = recall_score(y_true_np, y_pred_np, average=avg_method, zero_division=0)
    f1 = f1_score(y_true_np, y_pred_np, average=avg_method, zero_division=0)

    return accuracy, precision, recall, f1


def _calculate_auprc(y_true_np, y_score_np, is_binary):
    """Calculate AUPRC with appropriate averaging."""
    try:
        if is_binary:
            auprc = average_precision_score(y_true_np, y_score_np)
        else:
            # For multi-class, we need to binarize the labels and use One-vs-Rest
            n_classes = y_score_np.shape[1]

            # Binarize labels for multi-class AUPRC
            y_true_bin = label_binarize(y_true_np, classes=np.arange(n_classes))

            # Calculate AUPRC for each class then average
            auprc = average_precision_score(y_true_bin, y_score_np, average="macro")

    except (ValueError, IndexError) as e:
        print(f"Warning: Could not calculate AUPRC: {e}")
        auprc = np.nan

    return auprc


def calculate_metrics(y_true, y_pred, is_binary=True):
    """Calculate comprehensive classification metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Model predictions (logits for binary, raw scores for multi-class)
        is_binary: Whether this is a binary classification problem

    Returns:
        Dictionary containing various classification metrics
    """
    # Convert to numpy arrays
    y_true_np, y_pred_np, y_score_np = _to_numpy_arrays(y_true, y_pred, is_binary)

    # Calculate base metrics
    accuracy, precision, recall, f1 = _calculate_base_metrics(
        y_true_np, y_pred_np, is_binary
    )

    # Calculate AUPRC
    auprc = _calculate_auprc(y_true_np, y_score_np, is_binary)

    # For binary case, calculate macro metrics separately
    if is_binary:
        macro_recall = recall_score(
            y_true_np, y_pred_np, average="macro", zero_division=0
        )
        macro_f1 = f1_score(y_true_np, y_pred_np, average="macro", zero_division=0)
    else:
        macro_recall = recall
        macro_f1 = f1

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "auprc": auprc,
    }
