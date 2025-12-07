import pytest
import pandas as pd
import numpy as np
from src.pipeline import UserSegmentationPipeline

def test_pipeline_initialization():
    pipeline = UserSegmentationPipeline()
    assert pipeline.scaler is not None
    assert pipeline.imputer is not None
    assert pipeline.is_fitted is False

def test_pipeline_preprocess(sample_df):
    pipeline = UserSegmentationPipeline()
    processed_df = pipeline.preprocess(sample_df, is_training=True)
    
    # Check if unnecessary columns are dropped
    assert 'PlayerID' not in processed_df.columns
    
    # Check if categorical columns are encoded
    assert 'Gender_Male' in processed_df.columns or 'Gender' not in processed_df.columns
    
def test_pipeline_predict_without_fit_raises_error(sample_df):
    pipeline = UserSegmentationPipeline()
    with pytest.raises(ValueError, match="Model not trained"):
        pipeline.predict(sample_df)

def test_pipeline_load_exists():
    # Only test load if the model exists, otherwise skip or warn
    try:
        pipeline = UserSegmentationPipeline()
        pipeline.load()
        assert pipeline.is_fitted is True
    except FileNotFoundError:
        pytest.skip("Model file not found, skipping load test")
