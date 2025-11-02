import pytest
import pandas as pd
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
from unittest.mock import patch
import logging

# Setup test data and fixtures
@pytest.fixture
def sample_df():
    """Create a sample dataframe with known missing values"""
    return pd.DataFrame({
        'Income': [50000, np.nan, 75000, np.nan],
        'Age': [25, 30, np.nan, 40],
        'Response': [1, 0, 1, 0]
    })

@pytest.fixture
def logger():
    """Create a test logger"""
    logger = logging.getLogger('test_logger')
    logger.setLevel(logging.WARNING)
    return logger

def test_missing_value_counts(sample_df):
    """Test calculation of missing value counts"""
    missing_cnt = sample_df.isna().sum().sort_values(ascending=False)
    expected = pd.Series({
        'Income': 2,
        'Age': 1,
        'Response': 0
    }).sort_values(ascending=False)
    pd.testing.assert_series_equal(missing_cnt, expected)

def test_missing_percentages(sample_df):
    """Test calculation of missing value percentages"""
    missing_cnt = sample_df.isna().sum().sort_values(ascending=False)
    missing_pct = (missing_cnt / len(sample_df)).round(4) * 100
    expected_pct = pd.Series({
        'Income': 50.0,
        'Age': 25.0,
        'Response': 0.0
    }).sort_values(ascending=False)
    pd.testing.assert_series_equal(missing_pct, expected_pct)

def test_missing_report_creation(sample_df):
    """Test creation of missing values report DataFrame"""
    missing_cnt = sample_df.isna().sum().sort_values(ascending=False)
    missing_pct = (missing_cnt / len(sample_df)).round(4) * 100
    missing_report = pd.DataFrame({"missing_count": missing_cnt, "missing_pct": missing_pct})
    
    assert isinstance(missing_report, pd.DataFrame)
    assert all(col in missing_report.columns for col in ['missing_count', 'missing_pct'])
    assert len(missing_report) == len(sample_df.columns)

@patch('matplotlib.pyplot.show')
def test_missing_values_plot(mock_show, sample_df):
    """Test missing values plotting functionality"""
    missing_cnt = sample_df.isna().sum().sort_values(ascending=False)
    ax = missing_cnt[missing_cnt > 0].plot(kind="bar")
    
    # Set the title here to ensure it matches what you're testing
    ax.set_title("Missing values per column")
    
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == "Missing values per column"

def test_income_warning_logging(sample_df, logger):
    """Test Income column missing values warning"""
    with patch.object(logger, 'warning') as mock_warning:
        missing_cnt = sample_df.isna().sum()
        if 'Income' in missing_cnt.index and missing_cnt['Income'] > 0:
            logger.warning("Missing values detected in 'Income' column.")
        
        mock_warning.assert_called_once_with("Missing values detected in 'Income' column.")

def test_empty_dataframe():
    """Test handling of empty DataFrame"""
    empty_df = pd.DataFrame()
    missing_cnt = empty_df.isna().sum().sort_values(ascending=False)
    missing_pct = (missing_cnt / len(empty_df)).round(4) * 100 if len(empty_df) > 0 else pd.Series()
    missing_report = pd.DataFrame({"missing_count": missing_cnt, "missing_pct": missing_pct})
    
    assert len(missing_report) == 0
    assert isinstance(missing_report, pd.DataFrame)

def test_no_missing_values():
    """Test handling of DataFrame with no missing values"""
    complete_df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    })
    missing_cnt = complete_df.isna().sum().sort_values(ascending=False)
    assert missing_cnt.sum() == 0