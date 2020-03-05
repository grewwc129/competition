from ..csv_to_pkl import _write_single
from scripts.config import config
import pandas as pd 

import pytest


@pytest.fixture
def get_df():
    return pd.read_csv(config.train_file_name, nrows=100, low_memory=False)


@pytest.mark.passed
def test_write_single(get_df):
    _write_single(get_df, "df.h5")


