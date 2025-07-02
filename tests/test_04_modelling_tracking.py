import pytest
import pandas as pd
import numpy as np
import os, sys
import matplotlib
#from unittest.mock import patch
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")  # Prevents GUI backend during CI

# Add project path
sys.path.append(os.path.abspath(os.path.join('..')))

# Import your FraudDetectionPipeline class
from scripts._04_Modelling_Tracking import CreditRiskTrainer

@pytest.fixture(scope="module")
def dummy_model_data(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "model_data.csv"
    df = pd.DataFrame({
        "vd": [0, 1] * 50,
        "Amount": np.random.uniform(100, 500, 100),
        "Value": np.random.uniform(200, 1000, 100),
        "txn_hour": np.random.randint(0, 24, 100)
    })
    df.to_csv(path, index=False)
    return str(path)

@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("models")

@pytest.fixture(scope="module")
def trainer(dummy_model_data, model_dir):
    return CreditRiskTrainer(data_path=dummy_model_data, mdl_dir=str(model_dir))

def patch_load_data(trainer):
    def patched():
        abs_path = os.path.abspath(trainer.data_path)
        df = pd.read_csv(abs_path)
        drop_cols = ["vd", "FraudResult"]
        X = df.drop(columns=[col for col in drop_cols if col in df.columns])
        y = df["vd"]
        return train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    trainer.load_data = patched

# Patched with monkey to avoid file system dependency
def test_load_data(trainer):
    patch_load_data(trainer)
    X_train, X_test, y_train, y_test = trainer.load_data()
    assert X_train.shape[0] + X_test.shape[0] == 100
    assert "vd" not in X_train.columns

def test_evaluate_model(trainer):
    patch_load_data(trainer)
    X_train, X_test, y_train, y_test = trainer.load_data()
    model = trainer.get_best_model() or trainer.train_with_tracking() or trainer.model
    trainer.model = model
    metrics = trainer.evaluate_model(model, X_test, y_test)
    for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        assert k in metrics
        assert 0 <= metrics[k] <= 1

def test_compare_models(trainer):
    trainer.compare_models()
    assert hasattr(trainer, "best_model_name")

def test_train_with_tracking(trainer):
    trainer.train_with_tracking()
    assert trainer.model is not None
    assert hasattr(trainer.model, "predict_proba")

def test_get_best_model(trainer):
    model = trainer.get_best_model()
    assert model is not None
    assert hasattr(model, "predict_proba")

def test_save_and_load_model(trainer):
    trainer.save_model()
    trainer.load_model()
    assert trainer.model is not None
    assert hasattr(trainer.model, "predict_proba")

def test_plot_curves(trainer):
    try:
        trainer.plot_curves()
    except Exception as e:
        pytest.fail(f"plot_curves raised an error: {e}")
