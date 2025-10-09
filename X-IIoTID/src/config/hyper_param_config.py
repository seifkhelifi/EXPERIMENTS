def create_sweep_config():
    """Create improved W&B sweep configuration for CNN-LSTM on X-IIoTID"""
    sweep_config = {
        "method": "random",
        "metric": {"name": "best_f1_accuracy", "goal": "maximize"},
        "parameters": {
            "learning_rate": {
                "distribution": "log_uniform_values",
                "min": 1e-5,
                "max": 1e-2,
            },
            "batch_size": {"values": [32, 64, 128, 256]},
            "dropout_rate": {"distribution": "uniform", "min": 0.1, "max": 0.6},
            "epochs": {"value": 2},
        },
    }
    return sweep_config


def create_transformer_sweep_config():
    """W&B sweep for TransformerIDS — ONLY lr, batch_size, dropout (consistent with your CNN/LSTM)."""
    return {
        "method": "random",
        "metric": {"name": "best_f1_accuracy", "goal": "maximize"},
        "parameters": {
            "learning_rate": {
                "distribution": "log_uniform_values",
                "min": 3e-5,  # safe floor for Adam on this stack
                "max": 3e-3,  # avoids the too-hot 1e-2 corner
            },
            "batch_size": {"values": [32, 64, 128, 256]},
            "dropout_rate": {"distribution": "uniform", "min": 0.1, "max": 0.5},
            "epochs": {"value": 2},  # same as your prior sweeps
        },
    }


def create_hybrid_gnn_config():
    """W&B sweep for HybridAGNN_IDS — ONLY lr, batch_size, dropout (no arch changes)."""
    return {
        "method": "random",
        "metric": {"name": "best_f1_accuracy", "goal": "maximize"},
        "parameters": {
            "learning_rate": {
                "distribution": "log_uniform_values",
                "min": 1e-4,  # GNNs can like a touch higher lr than TF encoders
                "max": 3e-3,
            },
            "batch_size": {"values": [32, 64, 128, 256]},
            "dropout_rate": {"distribution": "uniform", "min": 0.2, "max": 0.6},
            "epochs": {"value": 2},
        },
    }
