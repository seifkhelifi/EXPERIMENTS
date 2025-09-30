import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

import wandb
import argparse

import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.models.factory import ModelFactory
from src.config.archi_config import archi_config
from src.config.hyper_param_config import create_sweep_config
from src.utils.metrics import calculate_metrics
from src.utils.seed import set_seed

from src.data.data_process import run_optimized_pipeline


model = ModelFactory.create("cnn_lstm_attention", **archi_config)
print(model.__class__.__name__)


def train_epoch(model, train_loader, criterion, optimizer, device, is_binary=True):
    """Train the model for one epoch"""
    model.train()
    total_loss = 0.0
    all_predictions = []
    all_targets = []

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(data)

        # Calculate loss
        if is_binary:
            output = (
                output.squeeze()
            )  # Remove extra dimension for binary classification
        loss = criterion(output, target)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Store predictions and targets for metrics
        all_predictions.append(output.detach())
        all_targets.append(target.detach())

    # Concatenate all predictions and targets
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)

    # Calculate metrics
    metrics = calculate_metrics(all_targets, all_predictions, is_binary)
    avg_loss = total_loss / len(train_loader)

    return avg_loss, metrics


def validate_epoch(model, val_loader, criterion, device, is_binary=True):
    """Validate the model for one epoch"""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)

            # Forward pass
            output = model(data)

            # Calculate loss
            if is_binary:
                output = (
                    output.squeeze()
                )  # Remove extra dimension for binary classification
            loss = criterion(output, target)

            total_loss += loss.item()

            # Store predictions and targets for metrics
            all_predictions.append(output)
            all_targets.append(target)

    # Concatenate all predictions and targets
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)

    # Calculate metrics
    metrics = calculate_metrics(all_targets, all_predictions, is_binary)
    avg_loss = total_loss / len(val_loader)

    return avg_loss, metrics


def train_model(
    model,
    archi_config,
    config,
    train_loader,
    val_loader,
    device,
    is_binary=True,
):
    """Complete training function for one configuration"""

    archi_config["dropout_rate"] = config["dropout_rate"]
    archi_config["num_classes"] = 1 if args.train_type == "binary" else 19
    model = ModelFactory.create(model, **archi_config)

    model.to(device)

    # Define loss function
    if is_binary:
        criterion = nn.BCELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Training metrics tracking
    best_f1 = 0.0
    best_model_state = None

    # Training loop
    for epoch in range(config["epochs"]):
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, is_binary
        )

        # Validate
        val_loss, val_metrics = validate_epoch(
            model, val_loader, criterion, device, is_binary
        )

        # Log metrics to W&B
        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_metrics["accuracy"],
                "train_precision": train_metrics["precision"],
                "train_recall": train_metrics["recall"],
                "train_f1": train_metrics["f1_score"],
                "val_loss": val_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1_score"],
            }
        )

        # Save best model based on F1 score
        if val_metrics["f1_score"] > best_f1:
            best_f1 = val_metrics["f1_score"]
            best_model_state = model.state_dict().copy()

        print(f'Epoch [{epoch+1}/{config["epochs"]}]:')
        print(
            f'  Train Loss: {train_loss:.4f}, Train F1: {train_metrics["f1_score"]:.4f}'
        )
        print(f'  Val Loss: {val_loss:.4f}, Val F1: {val_metrics["f1_score"]:.4f}')

    # Log best F1 score for sweep optimization
    wandb.log({"best_f1_accuracy": best_f1})

    return model, best_model_state, best_f1


def sweep_train(train_dataset, val_dataset, model_variant, archi_config):
    """Training function for binary classification sweep"""
    # Initialize W&B run
    wandb.init()
    config = wandb.config

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader_bin = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True
    )
    val_loader_bin = DataLoader(val_dataset, batch_size=config.batch_size)

    # Train model
    model, best_state, best_f1 = train_model(
        model=model_variant,
        archi_config=archi_config,
        config=config,
        train_loader=train_loader_bin,
        val_loader=val_loader_bin,
        device=device,
        is_binary=True,
    )

    wandb.finish()


def run_hyperparameter_sweep(
    train_dataset,
    val_dataset,
    archi_config,
    classification_type="binary",
    model_variant="vanilla",
    sweep_count=20,
):
    """
    Run hyperparameter sweep for the specified classification type

    Args:
        classification_type (str): 'binary' or 'multi'
        sweep_count (int): Number of sweep runs
    """

    # Create sweep configuration
    sweep_config = create_sweep_config()

    # Initialize sweep
    project_name = f"X-IIoTID-{model_variant}-{classification_type}-classification-{sweep_config['method']}-search"
    sweep_id = wandb.sweep(sweep_config, project=project_name)

    # Choose training function based on classification type

    train_function = lambda: sweep_train(
        train_dataset, val_dataset, model_variant, archi_config
    )

    # Run sweep
    print(f"Starting {classification_type} classification sweep...")
    print(f"Sweep ID: {sweep_id}")
    print(f"W&B Project: {project_name}")

    wandb.agent(sweep_id, train_function, count=sweep_count)

    print(
        f"Sweep completed! Check your results at: https://wandb.ai/your-username/{project_name}"
    )


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
        ],
        required=True,
        help="Choose model variant architecture",
    )

    args = parser.parse_args()

    set_seed(42)

    filepath = "../data/X-IIoTID dataset.csv"

    # Run pipeline
    results = run_optimized_pipeline(
        filepath,
    )

    # Pick dataset and label tensor type
    if args.train_type == "binary":
        data = results["binary"]
        label_tensor_type = torch.FloatTensor
    else:  # multi-class
        data = results["multi"]
        label_tensor_type = torch.LongTensor

    # Common preprocessing
    X_train_tensor = torch.FloatTensor(data["X_train"]).unsqueeze(2)
    X_val_tensor = torch.FloatTensor(data["X_val"]).unsqueeze(2)
    X_test_tensor = torch.FloatTensor(data["X_test"]).unsqueeze(2)

    y_train_tensor = label_tensor_type(data["y_train"])
    y_val_tensor = label_tensor_type(data["y_val"])
    y_test_tensor = label_tensor_type(data["y_test"])

    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    # Run hyperparameter sweep
    run_hyperparameter_sweep(
        train_dataset,
        val_dataset,
        archi_config,
        args.train_type,
        model_variant=args.model_variant,
        sweep_count=20,
    )
