import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import wandb
import argparse

import sys, os

from dotenv import load_dotenv


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.models.factory import ModelFactory
from src.config.archi_config import archi_config
from src.config.hyper_param_config import create_sweep_config
from src.utils.metrics import calculate_metrics
from src.utils.seed import set_seed

from src.data.data_process import run_optimized_pipeline
from src.data.construct_graph_x_iiot_d import example_integration_with_pipeline


from src.config.archi_config import create_config


def train_epoch(model, train_loader, criterion, optimizer, device, is_binary=True):
    """Train the model for one epoch"""
    model.train()
    total_loss = 0.0
    all_predictions = []
    all_targets = []

    for batch_idx, (data, target) in enumerate(train_loader):
        target = target.float()
        data, target = data.to(device), target.to(device)

        # Zero gradients
        optimizer.zero_grad()

        if hasattr(model, "edge_index") and model.edge_index is None:
            model.edge_index = edge_index.to(device)

        # Forward pass
        output = model(data)

        # Calculate loss
        # if is_binary:
        #     output = (
        #         output.squeeze()
        #     )  # Remove extra dimension for binary classification

        if is_binary:
            # BCE expects floats; use [B,1] for both
            output = output.view(-1, 1)
            target = target.view(-1, 1).float()
        else:
            # CrossEntropy: logits [B, C], targets int64 [B]
            # If your labels are one-hot: argmax -> indices
            if target.dim() > 1 and target.size(-1) > 1:
                target = target.argmax(dim=-1)
            else:
                target = target.view(-1)  # flatten if [B,1]

            # Ensure correct dtype
            target = target.long()

            # (Optional sanity check)
            if output.dim() != 2:
                raise ValueError(
                    f"Expected logits [B, C] for multi-class, got shape {tuple(output.shape)}"
                )

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
            # if is_binary:
            #     output = (
            #         output.squeeze()
            #     )  # Remove extra dimension for binary classification

            if is_binary:
                output = output.view(-1, 1)
                target = target.view(-1, 1).float()
            else:
                # CrossEntropy: logits [B, C], targets int64 [B]
                # If your labels are one-hot: argmax -> indices
                if target.dim() > 1 and target.size(-1) > 1:
                    target = target.argmax(dim=-1)
                else:
                    target = target.view(-1)  # flatten if [B,1]

                # Ensure correct dtype
                target = target.long()

                # (Optional sanity check)
                if output.dim() != 2:
                    raise ValueError(
                        f"Expected logits [B, C] for multi-class, got shape {tuple(output.shape)}"
                    )

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
    num_classes,
    classification_type,
):
    """Complete training function for one configuration"""

    # archi_config["dropout_rate"] = config["dropout_rate"]
    # archi_config["num_classes"] = 1 if args.train_type == "binary" else 19

    # Define loss function
    is_binary = True if classification_type == "binary" else False

    if is_binary:
        criterion = nn.BCELoss()
        archi_config = create_config(
            args.model_variant,
            train_type=classification_type,
            dropout_rate=config["dropout_rate"],
        )
    else:
        criterion = nn.CrossEntropyLoss()
        archi_config = create_config(
            args.model_variant,
            train_type=classification_type,
            num_classes=num_classes,
            dropout_rate=config["dropout_rate"],
        )

    is_gnn_model = args.model_variant in ["simple_gnn", "hybrid_gnn"]

    model = ModelFactory.create(model, **archi_config)

    model.to(device)

    # warm-up one mini-batch to instantiate lazy layers on the correct device
    if is_gnn_model:
        xb, yb = next(iter(train_loader))
        with torch.no_grad():
            _ = model(xb.to(device))

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
                "train_macro_recall": train_metrics["macro_recall"],
                "train_macro_f1": train_metrics["macro_f1"],
                "train_auprc": train_metrics["auprc"],
                "val_loss": val_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1_score"],
                "val_macro_recall": val_metrics["macro_recall"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_auprc": val_metrics["auprc"],
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


def sweep_train(
    train_dataset,
    val_dataset,
    model_variant,
    archi_config,
    num_classes,
    classification_type,
):
    """Training function for binary classification sweep"""
    # Initialize W&B run
    wandb.init()
    config = wandb.config

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_gnn_model = model_variant in ["simple_gnn", "hybrid_gnn"]

    if is_gnn_model:
        model, best_state, best_f1 = train_model(
            model=model_variant,
            archi_config=archi_config,
            config=config,
            train_loader=train_dataset,
            val_loader=val_dataset,
            device=device,
            num_classes=num_classes,
            classification_type=classification_type,
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

        # Train model
        model, best_state, best_f1 = train_model(
            model=model_variant,
            archi_config=archi_config,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_classes=num_classes,
            classification_type=classification_type,
        )

    wandb.finish()


def run_hyperparameter_sweep(
    train_dataset,
    val_dataset,
    archi_config,
    num_classes,
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
        train_dataset,
        val_dataset,
        model_variant,
        archi_config,
        num_classes,
        classification_type,
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
    
    load_dotenv()
    api_key = os.getenv("WANB_DB_KEY")
    
    wandb.login(key=api_key)


    filepath = "../data/X-IIoTID dataset.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    is_gnn_model = args.model_variant in ["simple_gnn", "hybrid_gnn"]

    if is_gnn_model:
        # Reuse your processed splits from pipeline
        train_dataset, val_dataset, test_dataset, edge_index = (
            example_integration_with_pipeline(
                results,
                classification=args.train_type,
                num_nodes=16,
                device=device,
            )
        )

    else:
        X_train_tensor = torch.FloatTensor(data["X_train"]).unsqueeze(2)
        X_val_tensor = torch.FloatTensor(data["X_val"]).unsqueeze(2)

        y_train_tensor = label_tensor_type(data["y_train"])
        y_val_tensor = label_tensor_type(data["y_val"])

        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    # Run hyperparameter sweep
    run_hyperparameter_sweep(
        train_dataset,
        val_dataset,
        archi_config,
        num_classes=data["num_classes"] if args.train_type == "multi" else 1,
        classification_type=args.train_type,
        model_variant=args.model_variant,
        sweep_count=20,
    )
