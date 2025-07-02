import pytest
import pandas as pd
import numpy as np
import os, sys

# Add project path
sys.path.append(os.path.abspath(os.path.join('..')))

# Import your FraudDetectionPipeline class
from scripts._03_Target_Engineering import RiskTargetBuilder  # adjust import if needed

@pytest.fixture
def dummy_rfm_data(tmp_path):
    # Create a temporary CSV for testing
    df = pd.DataFrame({
    'CustomerId': ['C1'] * 10 + ['C2'] * 10 + ['C3'] * 10,
    'TransactionStartTime': pd.date_range('2024-01-01', periods=30, freq='H'),
    'TransactionId': range(30),
    'Amount': np.random.uniform(50, 500, 30)
})
    
    path = tmp_path / "rfm_test.csv"
    df.to_csv(path, index=False)
    return str(path)

def test_compute_rfm(dummy_rfm_data):
    builder = RiskTargetBuilder(dummy_rfm_data)
    rfm_df = builder.compute_rfm()

    expected_n = pd.read_csv(dummy_rfm_data)["CustomerId"].nunique()
    assert rfm_df.shape[0] == expected_n
    assert list(rfm_df.columns) == ["CustomerId", "Recency", "Frequency", "Monetary"]

def test_scale_rfm(dummy_rfm_data):
    builder = RiskTargetBuilder(dummy_rfm_data)
    rfm_df = builder.compute_rfm()
    scaled = builder.scale_rfm(rfm_df)
    assert scaled.shape == (3, 3)  # 3 customers, 3 RFM features

def test_cluster_customers(dummy_rfm_data):
    builder = RiskTargetBuilder(dummy_rfm_data, n_clusters=2)
    rfm_df = builder.compute_rfm()
    scaled = builder.scale_rfm(rfm_df)
    clusters = builder.cluster_customers(scaled)
    assert len(clusters) == 3
    assert set(clusters) <= {0, 1}

def test_assign_high_risk(dummy_rfm_data):
    builder = RiskTargetBuilder(dummy_rfm_data, n_clusters=2)
    rfm_df = builder.compute_rfm()
    scaled = builder.scale_rfm(rfm_df)
    clusters = builder.cluster_customers(scaled)
    labeled = builder.assign_high_risk(rfm_df, clusters)
    assert "is_high_risk" in labeled.columns
    assert labeled["is_high_risk"].isin([0, 1]).all()

def test_generate_target(dummy_rfm_data):
    builder = RiskTargetBuilder(dummy_rfm_data)
    target_df = builder.generate_target()
    assert target_df.shape[0] == 3
    assert "CustomerId" in target_df.columns
    assert "is_high_risk" in target_df.columns

