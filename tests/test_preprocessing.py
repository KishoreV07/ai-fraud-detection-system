"""
test_preprocessing.py
------------------------
Unit tests for the data preprocessing pipeline.

Run:  pytest tests/test_preprocessing.py -v
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from preprocessing import clean_data, scale_features


def make_sample_df():
    """Creates a small synthetic dataset mimicking the real schema."""
    return pd.DataFrame({
        "Time": [0, 1, 2, 2],  # last row is a duplicate of row 2 on purpose
        "Amount": [10.0, 200.0, 50.0, 50.0],
        "V1": [0.1, -0.2, 0.3, 0.3],
        "V2": [0.5, -0.5, 0.0, 0.0],
        "Class": [0, 1, 0, 0],
    })


def test_clean_data_removes_duplicates():
    df = make_sample_df()
    cleaned = clean_data(df)
    assert len(cleaned) == 3, "Expected exactly one duplicate row to be removed"


def test_clean_data_handles_missing_values():
    df = make_sample_df()
    df.loc[0, "Amount"] = np.nan
    cleaned = clean_data(df)
    assert cleaned["Amount"].isnull().sum() == 0, "Missing values should be filled, not left as NaN"


def test_scale_features_creates_scaled_columns():
    df = make_sample_df().drop_duplicates()
    scaled = scale_features(df)
    assert "scaled_amount" in scaled.columns
    assert "scaled_time" in scaled.columns
    assert "Amount" not in scaled.columns, "Original Amount column should be dropped after scaling"
    assert "Time" not in scaled.columns, "Original Time column should be dropped after scaling"


def test_scale_features_output_is_standardized():
    df = make_sample_df().drop_duplicates()
    scaled = scale_features(df)
    # StandardScaler output should have approximately zero mean
    assert abs(scaled["scaled_amount"].mean()) < 1e-6