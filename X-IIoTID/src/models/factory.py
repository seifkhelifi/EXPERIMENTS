import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.models.cnn_lstm import CNN_LSTM
from src.models.cnn_bilstm import BiLSTM_CNN
from src.models.cnn_gru import CNN_GRU
from src.models.cnn_lstm_residual import DeepResidual_CNN_LSTM
from src.models.cnn_lstm_attention import CNN_LSTM_Attention
from src.models.transformer import TransformerIDS
from src.models.simplified_gnn_ids import SimplifiedGNN_IDS
from src.models.hybridagnn_ids import HybridAGNN_IDS


class ModelFactory:
    _registry = {
        "cnn_lstm": CNN_LSTM,
        "deepresidual_cnn_lstm": DeepResidual_CNN_LSTM,
        "bilstm_cnn": BiLSTM_CNN,
        "cnn_lstm_attention": CNN_LSTM_Attention,
        "cnn_gru": CNN_GRU,
        "transformer": TransformerIDS,
        "simple_gnn": SimplifiedGNN_IDS,
        "hybrid_gnn": HybridAGNN_IDS,
    }

    @classmethod
    def create(cls, name: str, **kwargs):
        """Create a model by name with the given parameters."""
        key = name.lower()
        if key not in cls._registry:
            raise ValueError(
                f"Unknown model type '{name}'. Available: {list(cls._registry.keys())}"
            )
        return cls._registry[key](**kwargs)


if __name__ == "__main__":
    cnn_lstm = {
        # CNN-LSTM models
        "input_channels": 1,
        "num_classes": 2,
        "feature_layers": 64,
        "lstm_units": 128,
        "dropout_rate": 0.4,
    }

    transformer_config = {
        "input_channels": 1,
        "num_classes": 2,
        "feature_dim": 128,
        "num_heads": 4,
        "num_layers": 2,
        "dropout_rate": 0.3,
        "ff_hidden": 256,
    }

    hybrid_gnn_config = {
        "num_classes": 2,
        "input_dim": 4,
        "hidden_dims": [128, 256, 128],
        "num_heads": 4,
        "dropout_rate": 0.4,
        "graph_layers": 3,
    }

    simple_gnn_config = {
        "num_classes": 2,
        "input_dim": 4,
        "hidden_dim": 128,
        "num_graph_layers": 2,
        "num_heads": 4,
        "dropout_rate": 0.3,
    }

    model = ModelFactory.create("cnn_lstm_attention", **cnn_lstm)
    print(model.__class__.__name__)

    model2 = ModelFactory.create("transformer", **transformer_config)
    print(model2.__class__.__name__)

    model3 = ModelFactory.create("simple_gnn", **simple_gnn_config)
    print(model3.__class__.__name__)

    model4 = ModelFactory.create("hybrid_gnn", **hybrid_gnn_config)
    print(model4.__class__.__name__)
