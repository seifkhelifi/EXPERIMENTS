import sys, os
import argparse
import json
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.models.factory import ModelFactory
from src.config.archi_config import create_config

from src.utils.metrics import calculate_metrics
from src.utils.seed import set_seed
from src.utils.plot_figs import (
    plot_training_history,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_per_class_confusion_matrix,
)
from src.utils.train_with_loss_lanscape import plot_loss_landscape_2d
from src.utils.helpers import (
    save_metrics_to_json,
    update_running_metrics,
    average_metrics,
    init_metrics_dict,
)

from src.data.data_process import run_optimized_pipeline
from src.data.data_process_smote import run_optimized_pipeline_with_smote
from src.data.construct_graph_x_iiot_d import example_integration_with_pipeline


# --- training ---
def train_model(
    model,
    model_variant,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    num_epochs=2,
    is_binary=True,
    plot_every=1,
):
    model.to(device)
    best_val_f1 = 0.0

    # Initialize history dictionary automatically from init_metrics_dict
    history = {"train_loss": [], "val_loss": []}
    for k in init_metrics_dict().keys():
        history[f"train_{k}"] = []
        history[f"val_{k}"] = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_metrics = init_metrics_dict()

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            if is_binary:
                targets = targets.view(-1, 1).float()

            if hasattr(model, "edge_index") and model.edge_index is None:
                model.edge_index = edge_index.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track loss and metrics
            train_loss += loss.item() * inputs.size(0)
            batch_metrics = calculate_metrics(targets, outputs, is_binary)
            update_running_metrics(train_metrics, batch_metrics, inputs.size(0))

        # Average over dataset
        train_loss /= len(train_loader.dataset)
        train_metrics = average_metrics(train_metrics, len(train_loader.dataset))

        # Validation
        model.eval()
        val_loss = 0.0
        val_metrics = init_metrics_dict()
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                if is_binary:
                    targets = targets.view(-1, 1).float()

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

                batch_metrics = calculate_metrics(targets, outputs, is_binary)
                update_running_metrics(val_metrics, batch_metrics, inputs.size(0))

        val_loss /= len(val_loader.dataset)
        val_metrics = average_metrics(val_metrics, len(val_loader.dataset))

        # Store in history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        for k in train_metrics.keys():
            history[f"train_{k}"].append(train_metrics[k])
            history[f"val_{k}"].append(val_metrics[k])

        # Logging
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(
            f"Train Loss: {train_loss:.4f} | "
            + ", ".join([f"{k}: {v:.4f}" for k, v in train_metrics.items()])
        )
        print(
            f"Val Loss: {val_loss:.4f} | "
            + ", ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()])
        )

        # Save best model (based on macro-F1 or normal F1 — you choose)
        if val_metrics["f1_score"] > best_val_f1:
            best_val_f1 = val_metrics["f1_score"]
            os.makedirs("out/checkpoints", exist_ok=True)
            torch.save(
                model.state_dict(),
                os.path.join(
                    "out/checkpoints",
                    f"{model_variant}_best_model_{'binary' if is_binary else 'multi'}.pt",
                ),
            )
            print("✅ Saved best model!")

    # Plot training history (extended with new metrics)
    plot_training_history(history, model_variant, is_binary)

    if (epoch + 1) % plot_every == 0:
        print(f"Plotting smooth loss landscape for epoch {epoch+1}...")
        plot_loss_landscape_2d(
            model,
            criterion,
            train_loader,
            device,
            epoch + 1,
            param_range=0.3,
            resolution=20,
            is_binary=is_binary,
            smoothing_sigma=1.0,
        )

    train_type = "binary" if is_binary else "multi"
    save_metrics_to_json(train_metrics, model_variant, train_type, phase="train")
    return model, history


# --- evaluation ---
def evaluate_model(
    model, model_variant, test_loader, criterion, device, is_binary=True
):
    model.eval()
    test_loss = 0.0
    all_targets, all_predictions, all_raw_predictions = [], [], []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            if is_binary:
                targets = targets.view(-1, 1).float()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)

            if is_binary:
                predictions = (outputs > 0.5).float()
                all_targets.extend(targets.cpu().numpy().flatten())
                all_predictions.extend(predictions.cpu().numpy().flatten())
                all_raw_predictions.extend(outputs.cpu().numpy().flatten())
            else:
                _, predictions = torch.max(outputs, dim=1)
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
                all_raw_predictions.extend(outputs.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    all_targets, all_predictions, all_raw_predictions = (
        np.array(all_targets),
        np.array(all_predictions),
        np.array(all_raw_predictions),
    )

    print("\n=== Test Results ===")
    print(f"Loss: {test_loss:.4f}")
    print(classification_report(all_targets, all_predictions))

    # Calculate metrics
    metrics = calculate_metrics(
        torch.tensor(all_targets),
        torch.tensor(all_raw_predictions),
        is_binary,
    )
    print(" | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))

    # Plots
    plot_confusion_matrix(all_targets, all_predictions, model_variant, is_binary)
    plot_pr_curve(all_targets, all_raw_predictions, model_variant, is_binary)

    plot_per_class_confusion_matrix(
        all_targets, all_predictions, model_variant, is_binary
    )

    test_results = {
        "test_loss": float(test_loss),
        "metrics": {k: float(v) for k, v in metrics.items()},
    }
    train_type = "binary" if is_binary else "multi"
    save_metrics_to_json(test_results, model_variant, train_type, phase="test")
    return test_results


def run_experiments(
    n_runs,
    model_variant,
    train_loader,
    val_loader,
    test_loader,
    criterion,
    learning_rate,
    device,
    num_epochs,
    is_binary,
    config,
):
    """
    Run multiple training/evaluation runs, compute mean & std of all metrics (train/val/test),
    and save results automatically under ./out/logs.
    """

    all_metrics = {}  # dynamic collection of all metrics

    for i in range(1, n_runs + 1):
        print(f"\n{'='*80}")
        print(f"🚀 RUN {i}/{n_runs} — {model_variant.upper()}")
        print(f"{'='*80}")

        # model = ModelFactory.create(model_variant, **simple_gnn_config)
        # optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        model = ModelFactory.create(model_variant, **config)

        is_gnn_model = args.model_variant in ["simple_gnn", "hybrid_gnn"]
        # warm-up one mini-batch to instantiate lazy layers on the correct device
        if is_gnn_model:

            model.to(device)

            xb, yb = next(iter(train_loader))
            with torch.no_grad():
                _ = model(xb.to(device))

        # NOW create the optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # --- TRAIN ---
        model, history = train_model(
            model,
            f"{model_variant}_run_{i}",  # pass run counter
            train_loader,
            val_loader,
            criterion,
            optimizer,
            device,
            num_epochs=num_epochs,
            is_binary=is_binary,
        )

        # --- Aggregate last-epoch train/val metrics ---
        for key, values in history.items():
            if len(values) > 0:
                metric_key = f"{key}"  # e.g., "val_f1_score"
                last_val = values[-1]
                if metric_key not in all_metrics:
                    all_metrics[metric_key] = []
                all_metrics[metric_key].append(last_val)

        # --- EVALUATE ---
        test_results = evaluate_model(
            model,
            f"{model_variant}_run_{i}",
            test_loader,
            criterion,
            device,
            is_binary=is_binary,
        )

        # --- Aggregate test metrics ---
        for key, val in test_results["metrics"].items():
            metric_key = f"test_{key}"  # prefix to avoid name clash
            if metric_key not in all_metrics:
                all_metrics[metric_key] = []
            all_metrics[metric_key].append(val)

        # Also include test loss
        if "test_loss" not in all_metrics:
            all_metrics["test_loss"] = []
        all_metrics["test_loss"].append(test_results["test_loss"])

    # --- Compute mean & std dynamically ---
    summary_stats = {}
    for metric, values in all_metrics.items():
        if len(values) > 0:
            summary_stats[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }

    # --- Save everything ---
    os.makedirs("out/logs", exist_ok=True)
    summary_path = os.path.join("out/logs", f"{model_variant}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_stats, f, indent=4)

    print(f"\n📁 Saved summary statistics to {summary_path}")
    print(json.dumps(summary_stats, indent=4))

    return summary_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run optimized training pipeline.")
    parser.add_argument(
        "--train_type",
        type=str,
        choices=["binary", "multi"],
        default="multi",
        help="Choose training type: 'binary' or 'multi'",
    )

    parser.add_argument(
        "--model_variant",
        type=str,
        choices=[
            "cnn_lstm",
            "deepresidual_cnn_lstm",
            "bilstm_cnn",
            "cnn_lstm_attention",
            "cnn_gru",
            "simple_gnn",
            "hybrid_gnn",
            "transformer",
        ],
        required=True,
        help="Choose model variant architecture",
    )

    parser.add_argument(
        "--sampling",
        type=str,
        choices=["normal", "smote"],
        default="normal",
        help="Choose data sampling method: 'normal' (no oversampling) or 'smote' (oversampling)",
    )

    parser.add_argument(
        "--n_runs",
        type=int,
        default=1,
        help="Choose number of runs",
    )

    args = parser.parse_args()

    # ---- enforce rules ----
    if args.sampling == "smote" and args.train_type != "multi":
        parser.error(
            "SMOTE can only be used with multi-class training. Use --train_type multi."
        )

    set_seed(42)

    filepath = "../data/X-IIoTID dataset.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    learning_rate = 0.001727656981651963
    num_epochs = 2  # Increased from 2 for better visualization
    batch_size = 64
    dropout_rate = 0.2729953238135602

    # hybrid_gnn_config["num_classes"] = 1 if args.train_type == "binary" else 19
    # hybrid_gnn_config["dropout_rate"] = 0.2729953238135602

    # Run pipeline
    if args.sampling == "normal":
        results = run_optimized_pipeline(filepath)
    else:  # smote
        results = run_optimized_pipeline_with_smote(filepath)

    if args.train_type == "binary":
        data = results["binary"]
        label_tensor_type = torch.FloatTensor
        criterion = nn.BCELoss()

        config = create_config(
            args.model_variant, train_type=args.train_type, dropout_rate=dropout_rate
        )
    else:  # multi
        data = results["multi"]
        label_tensor_type = torch.LongTensor
        criterion = nn.CrossEntropyLoss()

        config = create_config(
            args.model_variant,
            train_type=args.train_type,
            num_classes=data["num_classes"],
            dropout_rate=dropout_rate,
        )

    # # Common preprocessing
    # X_train_tensor = torch.FloatTensor(data["X_train"]).unsqueeze(2)
    # X_val_tensor = torch.FloatTensor(data["X_val"]).unsqueeze(2)
    # X_test_tensor = torch.FloatTensor(data["X_test"]).unsqueeze(2)

    # y_train_tensor = label_tensor_type(data["y_train"])
    # y_val_tensor = label_tensor_type(data["y_val"])
    # y_test_tensor = label_tensor_type(data["y_test"])

    # # Create data loaders
    # train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    # val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    # test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size)

    is_gnn_model = args.model_variant in ["simple_gnn", "hybrid_gnn"]

    if is_gnn_model:
        # Reuse your processed splits from pipeline
        train_loader, val_loader, test_loader, edge_index = (
            example_integration_with_pipeline(
                results,
                classification=args.train_type,
                num_nodes=16,
                device=device,
            )
        )

    else:
        # Existing CNN/LSTM pipeline
        X_train_tensor = torch.FloatTensor(data["X_train"]).unsqueeze(2)
        X_val_tensor = torch.FloatTensor(data["X_val"]).unsqueeze(2)
        X_test_tensor = torch.FloatTensor(data["X_test"]).unsqueeze(2)

        y_train_tensor = label_tensor_type(data["y_train"])
        y_val_tensor = label_tensor_type(data["y_val"])
        y_test_tensor = label_tensor_type(data["y_test"])

        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

    run_experiments(
        n_runs=args.n_runs,
        model_variant=args.model_variant,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        learning_rate=learning_rate,
        device=device,
        num_epochs=num_epochs,
        is_binary=True if args.train_type == "binary" else False,
        config=config,
    )
