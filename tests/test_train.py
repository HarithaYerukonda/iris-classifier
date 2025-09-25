import os
import joblib
import pytest
from src import train


def test_training_runs(tmp_path):
    """Check if training runs and outputs a model file."""
    # Change working dir to temp path
    os.chdir(tmp_path)

    # Run training
    train.main()

    # Assert model file exists
    model_path = os.path.join("outputs", "decision_tree_model.pkl")
    assert os.path.exists(model_path)

    # Assert model can be loaded
    model = joblib.load(model_path)
    assert hasattr(model, "predict")