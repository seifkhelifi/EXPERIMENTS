from copy import deepcopy


cnn_lstm_config = {
    "input_channels": 1,
    "num_classes": 2,
    "feature_layers": 64,
    "lstm_units": 128,
}

archi_config = {
    "input_channels": 1,
    "num_classes": 2,
    "feature_layers": 64,
    "lstm_units": 128,
}


transformer_config = {
    "input_channels": 1,
    "feature_dim": 128,
    "num_heads": 4,
    "num_layers": 2,
    "dropout_rate": 0.3,
    "ff_hidden": 256,
}


hybrid_gnn_config = {
    # "input_dim": 4,
    "hidden_dims": [128, 256, 128],
    "num_heads": 4,
    "graph_layers": 3,
}

simple_gnn_config = {
    # "input_dim": 4,
    "hidden_dim": 128,
    "num_graph_layers": 2,
    "num_heads": 4,
}


# ---- helper ----

_VARIANT_TO_BASE = {
    "cnn_lstm": cnn_lstm_config,
    "deepresidual_cnn_lstm": archi_config,
    "bilstm_cnn": archi_config,
    "cnn_lstm_attention": archi_config,
    "cnn_gru": archi_config,
    "simple_gnn": simple_gnn_config,
    "hybrid_gnn": hybrid_gnn_config,
    "transformer": transformer_config,
}


def create_config(
    model_variant: str,
    *,
    train_type: str | None = None,  # "binary" or e.g. "multiclass"
    num_classes: int | None = None,  # optional override; if None, infer from train_type
    dropout_rate: float | None = None,
    multiclass_count: int = 19,  # used when train_type != "binary" and num_classes is None
) -> dict:
    """
    Return a config dict for the requested model variant with num_classes and dropout_rate applied.

    Priority for num_classes:
        1) explicit num_classes argument
        2) infer from train_type: 1 if "binary" else multiclass_count

    dropout_rate:
        - if provided, it is added/overrides in the returned config (even if not present in the base).
    """
    if model_variant not in _VARIANT_TO_BASE:
        raise ValueError(
            f"Unknown model_variant '{model_variant}'. "
            f"Choose from {list(_VARIANT_TO_BASE)}"
        )

    cfg = deepcopy(_VARIANT_TO_BASE[model_variant])

    # determine num_classes
    if num_classes is None:
        if train_type is None:
            raise ValueError("Provide either num_classes or train_type.")
        num_classes = 1 if str(train_type).lower() == "binary" else multiclass_count

    cfg["num_classes"] = num_classes

    # apply dropout_rate if specified (always allowed to add/override)
    if dropout_rate is not None:
        cfg["dropout_rate"] = float(dropout_rate)

    return cfg


# ---- examples ----
# example 1: hybrid_gnn with train_type and specific dropout
# cfg = create_config("hybrid_gnn", train_type="binary", dropout_rate=0.2729953238135602)
# print(cfg)

# example 2: transformer with explicit num_classes (overrides train_type) and custom dropout
# cfg = create_config(
#     "transformer", train_type="multiclass", num_classes=7, dropout_rate=0.1
# )
# print(cfg)
