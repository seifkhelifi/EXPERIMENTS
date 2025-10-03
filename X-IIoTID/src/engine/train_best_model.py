import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
import numpy as np


import argparse

import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.models.factory import ModelFactory
from src.config.archi_config import archi_config
from src.utils.metrics import calculate_metrics
from src.utils.seed import set_seed
from src.utils.plot_figs import plot_training_history, plot_confusion_matrix
from src.utils.train_with_loss_lanscape import plot_loss_landscape_2d

from src.data.data_process import run_optimized_pipeline
from src.data.data_process_smote import run_optimized_pipeline_with_smote


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

    # Initialize history dictionary to store metrics
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "train_precision": [],
        "train_recall": [],
        "train_f1_score": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1_score": [],
    }

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_metrics = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
        }

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)

            # Reshape targets for binary classification
            if is_binary:
                targets = targets.view(-1, 1)

            # Calculate loss
            loss = criterion(outputs, targets)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update metrics
            train_loss += loss.item() * inputs.size(0)
            batch_metrics = calculate_metrics(targets, outputs, is_binary)
            train_metrics["accuracy"] += batch_metrics["accuracy"] * inputs.size(0)
            train_metrics["precision"] += batch_metrics["precision"] * inputs.size(0)
            train_metrics["recall"] += batch_metrics["recall"] * inputs.size(0)
            train_metrics["f1_score"] += batch_metrics["f1_score"] * inputs.size(0)

        # Calculate average metrics
        train_loss /= len(train_loader.dataset)
        train_metrics["accuracy"] /= len(train_loader.dataset)
        train_metrics["precision"] /= len(train_loader.dataset)
        train_metrics["recall"] /= len(train_loader.dataset)
        train_metrics["f1_score"] /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        val_metrics = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
        }

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass
                outputs = model(inputs)

                # Reshape targets for binary classification
                if is_binary:
                    targets = targets.view(-1, 1)

                # Calculate loss
                loss = criterion(outputs, targets)

                # Update metrics
                val_loss += loss.item() * inputs.size(0)
                batch_metrics = calculate_metrics(targets, outputs, is_binary)
                val_metrics["accuracy"] += batch_metrics["accuracy"] * inputs.size(0)
                val_metrics["precision"] += batch_metrics["precision"] * inputs.size(0)
                val_metrics["recall"] += batch_metrics["recall"] * inputs.size(0)
                val_metrics["f1_score"] += batch_metrics["f1_score"] * inputs.size(0)

        # Calculate average metrics
        val_loss /= len(val_loader.dataset)
        val_metrics["accuracy"] /= len(val_loader.dataset)
        val_metrics["precision"] /= len(val_loader.dataset)
        val_metrics["recall"] /= len(val_loader.dataset)
        val_metrics["f1_score"] /= len(val_loader.dataset)

        # Store metrics in history
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["train_precision"].append(train_metrics["precision"])
        history["train_recall"].append(train_metrics["recall"])
        history["train_f1_score"].append(train_metrics["f1_score"])

        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_precision"].append(val_metrics["precision"])
        history["val_recall"].append(val_metrics["recall"])
        history["val_f1_score"].append(val_metrics["f1_score"])

        # Print epoch results
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(
            f"Train Loss: {train_loss:.4f}, Accuracy: {train_metrics['accuracy']:.4f}, "
            f"Precision: {train_metrics['precision']:.4f}, Recall: {train_metrics['recall']:.4f}, "
            f"F1: {train_metrics['f1_score']:.4f}"
        )
        print(
            f"Val Loss: {val_loss:.4f}, Accuracy: {val_metrics['accuracy']:.4f}, "
            f"Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, "
            f"F1: {val_metrics['f1_score']:.4f}"
        )

        # Save the best model
        if val_metrics["f1_score"] > best_val_f1:
            best_val_f1 = val_metrics["f1_score"]

            # Create checkpoints directory if it doesn't exist
            os.makedirs("checkpoints", exist_ok=True)

            # Save model
            torch.save(
                model.state_dict(),
                os.path.join(
                    "checkpoints",
                    f'{model_variant}_best_model_{"binary" if is_binary else "multi"}.pt',
                ),
            )
            print("Saved best model!")

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

    # Plot training history
    plot_training_history(history, model_variant, is_binary)

    return model, history


# Function to evaluate model
def evaluate_model(
    model, model_variant, test_loader, criterion, device, is_binary=True
):
    model.eval()
    test_loss = 0.0
    all_targets = []
    all_predictions = []
    all_raw_predictions = []  # For ROC curve if needed later

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)

            # Reshape targets for binary classification
            if is_binary:
                targets = targets.view(-1, 1)

            # Calculate loss
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)

            # Store predictions and targets
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

    # Calculate average loss
    test_loss /= len(test_loader.dataset)

    # Convert lists to numpy arrays
    all_targets = np.array(all_targets)
    all_predictions = np.array(all_predictions)

    # Calculate metrics
    print("\nTest Results:")
    print(f"Loss: {test_loss:.4f}")
    print(classification_report(all_targets, all_predictions))

    # Plot confusion matrix
    plot_confusion_matrix(all_targets, all_predictions, model_variant, is_binary)

    # Calculate and print metrics
    # metrics = calculate_metrics(torch.tensor(all_targets), torch.tensor(all_raw_predictions) if is_binary else torch.tensor(all_predictions), is_binary)
    metrics = calculate_metrics(
        torch.tensor(all_targets),
        (
            torch.tensor(all_raw_predictions)
            if is_binary
            else torch.tensor(all_raw_predictions)
        ),
        is_binary,
    )
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test Precision: {metrics['precision']:.4f}")
    print(f"Test Recall: {metrics['recall']:.4f}")
    print(f"Test F1 Score: {metrics['f1_score']:.4f}")

    return test_loss, all_targets, all_predictions, all_raw_predictions


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

    parser.add_argument(
        "--sampling",
        type=str,
        choices=["normal", "smote"],
        default="normal",
        help="Choose data sampling method: 'normal' (no oversampling) or 'smote' (oversampling)",
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

    archi_config["num_classes"] = 1 if args.train_type == "binary" else 19
    archi_config["dropout_rate"] = 0.2729953238135602

    # Run pipeline
    if args.sampling == "normal":
        results = run_optimized_pipeline(filepath)
    else:  # smote
        results = run_optimized_pipeline_with_smote(filepath)

    if args.train_type == "binary":
        data = results["binary"]
        label_tensor_type = torch.FloatTensor
        criterion = nn.BCELoss()
    else:  # multi
        data = results["multi"]
        label_tensor_type = torch.LongTensor
        criterion = nn.CrossEntropyLoss()

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
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = ModelFactory.create(args.model_variant, **archi_config)
    print(model.__class__.__name__)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"\n{'='*60}")
    print(
        f"🚀 Training Model: {args.model_variant.upper()} | Type: {args.train_type.upper()}"
    )
    print(f"{'='*60}\n")

    train_model(
        model,
        args.model_variant,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        num_epochs,
        is_binary=True if args.train_type == "binary" else False,
        plot_every=1,
    )

    print(f"\n{'='*60}")
    print(
        f"🚀 Evaluating Model: {args.model_variant.upper()} | Type: {args.train_type.upper()}"
    )
    print(f"{'='*60}\n")

    model.load_state_dict(
        torch.load(
            f"./checkpoints/{args.model_variant}_best_model_{args.train_type}.pt"
        )
    )
    model.to(device)

    # Evaluate binary model
    print("\n📊 Evaluating Multi-Class Classification Model...")
    binary_test_loss, binary_targets, binary_predictions, binary_raw_predictions = (
        evaluate_model(
            model,
            args.model_variant,
            test_loader,
            criterion,
            device,
            is_binary=True if args.train_type == "binary" else False,
        )
    )
