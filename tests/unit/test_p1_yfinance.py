"""Research EOD fetch must use the last completed exchange session."""

from datetime import datetime, timezone

import pandas as pd
import pytest

from stock_pipeline.ingestion.market_price import publish_yfinance


def _history(day: str):
    return pd.DataFrame(
        {"Open": [100.0], "High": [105.0], "Low": [99.0],
         "Close": [101.0], "Volume": [1000]},
        index=pd.DatetimeIndex([day], tz="America/New_York"),
    )


def test_fetch_uses_completed_session_on_holiday_week(monkeypatch):
    class Ticker:
        def history(self, **_kwargs):
            return _history("2026-11-25")

    monkeypatch.setattr(publish_yfinance.yf, "Ticker", lambda _symbol: Ticker())
    now = datetime(2026, 11, 27, 15, tzinfo=timezone.utc)
    events = publish_yfinance.fetch_completed_bars(["AAPL"], now=now)
    assert len(events) == 1
    assert events[0]["source"] == "yfinance"
    assert events[0]["symbol"] == "AAPL"


def test_fetch_rejects_missing_session_without_publishing(monkeypatch):
    class Ticker:
        def history(self, **_kwargs):
            return _history("2026-11-24")

    monkeypatch.setattr(publish_yfinance.yf, "Ticker", lambda _symbol: Ticker())
    now = datetime(2026, 11, 27, 15, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="completed"):
        publish_yfinance.fetch_completed_bars(["AAPL"], now=now)
