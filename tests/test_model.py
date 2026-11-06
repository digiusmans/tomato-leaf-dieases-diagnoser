
from pathlib import Path
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_tomato_model.keras"


def test_model_loading():
    assert MODEL_PATH.exists(), "Model file not found."

    model = tf.keras.models.load_model(MODEL_PATH)

    assert model is not None
    assert model.input_shape == (None, 224, 224, 3)
    assert model.output_shape == (None, 10)
