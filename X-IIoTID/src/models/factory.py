import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.models.cnn_lstm import CNN_LSTM
from src.models.cnn_bilstm import BiLSTM_CNN
from src.models.cnn_gru import CNN_GRU
from src.models.cnn_lstm_residual import DeepResidual_CNN_LSTM
from src.models.cnn_lstm_attention import CNN_LSTM_Attention


class ModelFactory:
    _registry = {
        "cnn_lstm": CNN_LSTM,
        "deepresidual_cnn_lstm": DeepResidual_CNN_LSTM,
        "bilstm_cnn": BiLSTM_CNN,
        "cnn_lstm_attention": CNN_LSTM_Attention,
        "cnn_gru": CNN_GRU,
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
    config = {
        "input_channels": 1,
        "num_classes": 2,
        "feature_layers": 64,
        "lstm_units": 128,
        "dropout_rate": 0.4,
    }

    model = ModelFactory.create("cnn_lstm_attention", **config)
    print(model.__class__.__name__)

    model2 = ModelFactory.create("bilstm_cnn", **config)
    print(model2.__class__.__name__)
