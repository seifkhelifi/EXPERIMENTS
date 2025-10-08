import json
import os
import torch
import numpy as np


def save_metrics_to_json(metrics_dict, model_variant, train_type, phase="train"):
    """Save metrics (training or evaluation) to a JSON file under ./out/logs."""
    os.makedirs("out/logs", exist_ok=True)
    file_path = os.path.join(
        "out/logs", f"{model_variant}_{phase}_{train_type}_log.json"
    )

    # Convert all tensors to floats if needed
    def to_serializable(obj):
        if isinstance(obj, (torch.Tensor, np.ndarray)):
            return obj.tolist()
        elif isinstance(obj, (float, int, str, dict, list)):
            return obj
        else:
            return str(obj)

    with open(file_path, "w") as f:
        json.dump(metrics_dict, f, indent=4, default=to_serializable)

    print(f"📁 Saved {phase} metrics to {file_path}")


def update_running_metrics(running_metrics, batch_metrics, batch_size):
    for k in running_metrics.keys():
        running_metrics[k] += batch_metrics[k] * batch_size


def average_metrics(running_metrics, dataset_size):
    return {k: v / dataset_size for k, v in running_metrics.items()}


def init_metrics_dict():
    return {
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1_score": 0.0,
        "macro_recall": 0.0,
        "macro_f1": 0.0,
        "auprc": 0.0,
    }
