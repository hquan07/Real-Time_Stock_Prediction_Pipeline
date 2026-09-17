import pytest
import pandas as pd
import numpy as np
from scripts.signal_generator import compute_rsi

def test_compute_rsi():
    # Simple increasing series should have RSI near 100
    prices_up = pd.Series([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])
    rsi_up = compute_rsi(prices_up, window=14)
    # The last RSI value should be 100 since there are only gains
    assert np.isclose(rsi_up.iloc[-1], 100.0)

    # Simple decreasing series should have RSI near 0
    prices_down = pd.Series([24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10])
    rsi_down = compute_rsi(prices_down, window=14)
    assert np.isclose(rsi_down.iloc[-1], 0.0)

    # Alternating series should have RSI near 50
    prices_alt = pd.Series([10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10])
    rsi_alt = compute_rsi(prices_alt, window=14)
    assert np.isclose(rsi_alt.iloc[-1], 50.0)
