import pandas as pd
import numpy as np
import pytest
import os, sys
import tempfile

# Add project path
sys.path.append(os.path.abspath(os.path.join('..')))

# Import your FraudDetectionPipeline class
from scripts._02_Feature_Engineering import FraudDetectionPipeline

@pytest.fixture
def dummy_data():
    
    os.makedirs(os.path.join(os.path.dirname(__file__), 'data'), exist_ok=True)
    np.random.seed(42)  # for reproducibility

    df = pd.DataFrame({
        'CustomerId': np.random.choice(['C1', 'C2', 'C3', 'C4'], size=200),
        'FraudResult': np.random.choice([0, 1], size=200, p=[0.6, 0.4]),
        'Amount': np.random.normal(loc=300, scale=100, size=200).clip(5),  # varied, no negatives
        'Value': np.random.normal(loc=250, scale=80, size=200).clip(5),
        'TransactionStartTime': pd.date_range('2024-01-01', periods=200, freq='H'),
        'ProviderId': np.random.choice(['P1', 'P2', 'P3'], size=200),
        'ProductId': np.random.choice(['ProdA', 'ProdB', 'ProdC'], size=200),
        'ProductCategory': np.random.choice(['Cat1', 'Cat2'], size=200),
        'ChannelId': np.random.choice(['Web', 'Mobile', 'Checkout'], size=200),
        'PricingStrategy': np.random.choice(['Strategy1', 'Strategy2'], size=200),
        'CurrencyCode': ['USD'] * 200,
        'CountryCode': ['ET'] * 200,
        'TransactionId': np.arange(200),
        'BatchId': np.arange(200),
        'AccountId': ['A1'] * 200,
        'SubscriptionId': ['S1'] * 200,
    })
    
    file_path = os.path.join(os.path.dirname(__file__), 'data', 'temp.csv')
    df.to_csv(file_path, index=False)
    return file_path
    
def test_custom_split(dummy_data):
    pipeline = FraudDetectionPipeline(dummy_data)
    df = pd.read_csv(dummy_data)
    train, test = pipeline.custom_split_df(df, y_col='FraudResult', ratio=0.8)
    assert len(train) > len(test)
    assert set(train['FraudResult']) == set(test['FraudResult'])  # stratified split

def test_load_and_split(dummy_data):
    pipeline = FraudDetectionPipeline(dummy_data)
    pipeline.load_and_split_data()
    assert hasattr(pipeline, 'train')
    assert hasattr(pipeline, 'test')
    assert 'txn_hour' in pipeline.df.columns

def test_apply_woe(dummy_data):
    pipeline = FraudDetectionPipeline(dummy_data)
    pipeline.load_and_split_data()
    pipeline.compute_monotonic_breaks()
    pipeline.compute_categorical_breaks()
    pipeline.apply_woe_transformation()
    assert hasattr(pipeline, 'train_woe')
    assert hasattr(pipeline, 'test_woe')
    assert pipeline.train_woe.shape[0] == pipeline.train.shape[0]

def test_train_and_predict(dummy_data):
    pipeline = FraudDetectionPipeline(dummy_data)
    pipeline.load_and_split_data()
    pipeline.compute_monotonic_breaks()
    pipeline.compute_categorical_breaks()
    pipeline.apply_woe_transformation()
    pipeline.merge_and_clean()
    pipeline.filter_variables()
    pipeline.train_model()
    preds = pipeline.lr.predict_proba(pipeline.X_test)
    assert preds.shape[1] == 2
    assert 0 <= preds[0][1] <= 1

def test_model_save_load(dummy_data):
    with tempfile.TemporaryDirectory(dir=os.path.dirname(dummy_data)) as tmp_dir:
        pipeline = FraudDetectionPipeline(dummy_data, mdl_dir=tmp_dir)
        pipeline.load_and_split_data()
        pipeline.compute_monotonic_breaks()
        pipeline.compute_categorical_breaks()
        pipeline.apply_woe_transformation()
        pipeline.merge_and_clean()
        pipeline.filter_variables()
        pipeline.train_model()
        pipeline.save_model()
        pipeline.load_model()
        assert pipeline.lr is not None