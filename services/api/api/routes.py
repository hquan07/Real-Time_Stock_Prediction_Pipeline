from datetime import datetime, timezone
import json
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from loguru import logger
import pandas as pd

from services.data_service import fetch_stock_data
from services.redis_cache import get_cached_data, set_cached_data


router = APIRouter()


def _as_utc(value) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _base_meta(**extra) -> dict:
    return {
        "source": None,
        "as_of": None,
        "timezone": "UTC",
        "exchange_timezone": "America/New_York",
        "market_status": "unknown",
        "stale": None,
        **extra,
    }


def _market_status() -> str:
    try:
        import exchange_calendars as xcals

        now = pd.Timestamp.now(tz="UTC")
        calendar = xcals.get_calendar("XNYS")
        return "open" if calendar.is_open_on_minute(now, ignore_breaks=True) else "closed"
    except Exception as exc:
        logger.warning(f"Could not calculate market status: {exc}")
        return "unknown"


@router.get("/tickers")
async def get_active_tickers(request: Request):
    pool = getattr(request.app.state, "pool", None)
    if not pool:
        raise HTTPException(status_code=503, detail="Database unavailable")

    cache_key = "api:tickers:active:v2"
    cached = await get_cached_data(cache_key)
    if cached:
        return cached

    try:
        query = "SELECT ticker, long_name FROM companies WHERE is_active = true ORDER BY ticker"
        async with pool.acquire() as connection:
            records = await connection.fetch(query)
        result = {
            "data": [
                {"label": f"{row['long_name'] or row['ticker']} ({row['ticker']})", "value": row["ticker"]}
                for row in records
            ],
            "meta": _base_meta(source="database", market_status=_market_status()),
        }
        await set_cached_data(cache_key, result, expire=3600)
        return result
    except Exception as exc:
        logger.error(f"Failed to fetch tickers: {exc}")
        raise HTTPException(status_code=503, detail="Ticker catalog unavailable") from exc


@router.get("/stock/{ticker}")
async def get_stock_data(
    request: Request, ticker: str, period: str = Query("3M", alias="period")
):
    pool = getattr(request.app.state, "pool", None)
    if not pool:
        raise HTTPException(status_code=503, detail="Database unavailable")
    ticker = ticker.upper()
    cache_key = f"api:stock:v2:{ticker}:{period}"
    cached = await get_cached_data(cache_key)
    if cached:
        return cached

    try:
        frame = await fetch_stock_data(pool, ticker, period)
        if frame.empty:
            return {
                "data": [],
                "state": "empty",
                "meta": _base_meta(period=period, market_status=_market_status()),
            }

        latest = frame.iloc[-1]
        as_of = _as_utc(latest["event_time"])
        interval = latest.get("interval") or "1d"
        stale_after_seconds = 36 * 3600 if interval == "1d" else 15 * 60
        age_seconds = max(
            0, (datetime.now(timezone.utc) - as_of.to_pydatetime()).total_seconds()
        )
        meta = _base_meta(
            source=latest.get("source"),
            as_of=as_of.isoformat(),
            exchange_timezone=latest.get("exchange_timezone") or "America/New_York",
            market_status=_market_status(),
            stale=age_seconds > stale_after_seconds,
            age_seconds=int(age_seconds),
            interval=interval,
            schema_version=int(latest.get("schema_version") or 0),
            period=period,
        )
        serializable = frame.copy()
        for column in ("date", "event_time"):
            if column in serializable:
                serializable[column] = serializable[column].astype(str)
        serializable = serializable.where(pd.notna(serializable), None)
        result = {"data": serializable.to_dict(orient="records"), "state": "ready", "meta": meta}
        await set_cached_data(cache_key, result, expire=300)
        return result
    except Exception as exc:
        logger.error(f"Failed to fetch stock data: {exc}")
        raise HTTPException(status_code=503, detail="Market data unavailable") from exc


@router.get("/portfolio/{user_id}")
async def get_portfolio(request: Request, user_id: str):
    pool = getattr(request.app.state, "pool", None)
    if not pool:
        raise HTTPException(status_code=503, detail="Database unavailable")
    try:
        query = """
            WITH latest_prices AS (
                SELECT DISTINCT ON (ticker) ticker, close AS current_price, event_time
                FROM stock_prices_stream ORDER BY ticker, event_time DESC
            )
            SELECT p.ticker, p.shares AS quantity, p.avg_price AS average_price,
                   COALESCE(lp.current_price, p.avg_price) AS current_price,
                   lp.event_time AS data_as_of
            FROM portfolio p LEFT JOIN latest_prices lp USING (ticker)
            WHERE p.user_id = $1
        """
        async with pool.acquire() as connection:
            records = await connection.fetch(query, user_id)
        data = []
        timestamps = []
        for row in records:
            item = dict(row)
            quantity = float(item["quantity"])
            average_price = float(item["average_price"])
            current_price = float(item["current_price"])
            item.update(
                quantity=quantity,
                average_price=average_price,
                current_price=current_price,
                total_cost=quantity * average_price,
                total_value=quantity * current_price,
                unrealized_pnl=quantity * (current_price - average_price),
            )
            if item.get("data_as_of") is not None:
                as_of = _as_utc(item["data_as_of"])
                timestamps.append(as_of)
                item["data_as_of"] = as_of.isoformat()
            data.append(item)

        latest_as_of = max(timestamps).isoformat() if timestamps else None
        return {
            "data": data,
            "state": "ready" if data else "empty",
            "meta": _base_meta(
                source="database" if data else None,
                as_of=latest_as_of,
                market_status=_market_status(),
                research_only=True,
            ),
        }
    except Exception as exc:
        logger.error(f"Failed to fetch portfolio: {exc}")
        raise HTTPException(status_code=503, detail="Portfolio unavailable") from exc


@router.get("/predictions/{ticker}")
async def get_predictions(
    request: Request, ticker: str, model: str = Query("random_forest")
):
    pool = getattr(request.app.state, "pool", None)
    if not pool:
        raise HTTPException(status_code=503, detail="Database unavailable")
    try:
        query = """
            SELECT prediction_date, target_date, predicted_close, predicted_return,
                   confidence, uncertainty_validated, model_version, created_at
            FROM predictions
            WHERE ticker = $1 AND lower(model_name) = lower($2)
            ORDER BY prediction_date ASC
        """
        async with pool.acquire() as connection:
            records = await connection.fetch(query, ticker.upper(), model)
        data = []
        for row in records:
            item = dict(row)
            for field in ("prediction_date", "target_date", "created_at"):
                if item.get(field) is not None:
                    item[field] = item[field].isoformat()
            data.append(item)
        latest = data[-1] if data else {}
        return {
            "data": {"historical": data, "future": []},
            "state": "ready" if data else "empty",
            "meta": _base_meta(
                source="model_registry" if data else None,
                as_of=latest.get("created_at"),
                model_version=latest.get("model_version"),
                uncertainty_validated=bool(latest.get("uncertainty_validated", False)),
                research_only=True,
            ),
        }
    except Exception as exc:
        logger.error(f"Failed to fetch predictions: {exc}")
        raise HTTPException(status_code=503, detail="Predictions unavailable") from exc


@router.get("/feature-importance/{ticker}")
async def get_feature_importance(ticker: str):
    model_dir = Path(os.getenv("MODEL_REPORT_DIR", "artifacts/model-reports"))
    candidates = [
        model_dir / f"feature_importance_{ticker.upper()}.json",
        model_dir / "feature_importance_ALL.json",
    ]
    target = next((path for path in candidates if path.exists()), None)
    if target is None:
        return {
            "data": [],
            "state": "empty",
            "meta": _base_meta(reason="No validated feature-importance artifact"),
        }
    try:
        return {"data": json.loads(target.read_text()), "state": "ready", "meta": _base_meta(source="model_registry")}
    except Exception as exc:
        logger.error(f"Error reading feature importance: {exc}")
        raise HTTPException(status_code=500, detail="Invalid feature-importance artifact") from exc
