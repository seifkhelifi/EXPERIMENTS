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
