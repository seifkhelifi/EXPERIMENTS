import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def calculate_metrics(y_true, y_pred, is_binary=True):
    if is_binary:
        # Convert logits to binary predictions
        y_pred_class = (y_pred > 0.5).float()
        # Convert tensors to numpy for sklearn metrics
        y_true_np = y_true.cpu().numpy().flatten()
        y_pred_np = y_pred_class.cpu().numpy().flatten()

        # Calculate metrics
        accuracy = accuracy_score(y_true_np, y_pred_np)
        precision = precision_score(y_true_np, y_pred_np, zero_division=0)
        recall = recall_score(y_true_np, y_pred_np, zero_division=0)
        f1 = f1_score(y_true_np, y_pred_np, zero_division=0)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }
    else:
        # For multi-class, y_pred is logits with shape (batch_size, num_classes)
        _, y_pred_class = torch.max(y_pred, dim=1)
        # Convert tensors to numpy for sklearn metrics
        y_true_np = y_true.cpu().numpy()
        y_pred_np = y_pred_class.cpu().numpy()

        # Calculate metrics (macro averaging)
        accuracy = accuracy_score(y_true_np, y_pred_np)
        precision = precision_score(
            y_true_np, y_pred_np, average="macro", zero_division=0
        )
        recall = recall_score(y_true_np, y_pred_np, average="macro", zero_division=0)
        f1 = f1_score(y_true_np, y_pred_np, average="macro", zero_division=0)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }
