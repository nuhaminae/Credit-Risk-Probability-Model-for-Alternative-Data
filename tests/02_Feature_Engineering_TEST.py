# tests/test_fraud_pipeline.py
import pandas as pd
import numpy as np
import pytest
import os, sys

# Add project path
sys.path.append(os.path.abspath(os.path.join('..')))

# Import your FraudDetectionPipeline class
from scripts._02_Feature_Engineering import FraudDetectionPipeline



@pytest.fixture
def dummy_data():
    df = pd.DataFrame({
        'CustomerId': ['C1'] * 50 + ['C2'] * 50,
        'FraudResult': [0] * 90 + [1] * 10,
        'Amount': np.random.uniform(10, 500, 100),
        'Value': np.random.uniform(10, 500, 100),
        'TransactionStartTime': pd.date_range('2024-01-01', periods=100, freq='H'),
        'ProviderId': ['P1'] * 100,
        'ProductId': ['ProdA'] * 100,
        'ProductCategory': ['Cat1'] * 100,
        'ChannelId': ['Web'] * 100,
        'PricingStrategy': ['Strategy1'] * 100,
        'CurrencyCode': ['USD'] * 100,
        'CountryCode': ['ET'] * 100,
        'TransactionId': list(range(100)),
        'BatchId': list(range(100)),
        'AccountId': ['A1'] * 100,
        'SubscriptionId': ['S1'] * 100,
        'CustomerId': ['C1'] * 50 + ['C2'] * 50
    })
    file_path = 'tests/data/temp.csv'
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

def test_model_save_load(tmp_path, dummy_data):
    pipeline = FraudDetectionPipeline(dummy_data, mdl_dir=tmp_path)
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