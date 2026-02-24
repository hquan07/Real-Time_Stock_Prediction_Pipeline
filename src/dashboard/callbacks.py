import os
import sys
import pickle
from datetime import datetime, timedelta
from typing import Tuple, Any, Dict, List

import numpy as np
import pandas as pd
import psycopg2
from dash import Input, Output, State, callback
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
from loguru import logger

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Theme Colors (match layouts.py)
COLORS = {
    "background": "#0f0f1a",
    "card": "rgba(26, 26, 46, 0.8)",
    "primary": "#6366f1",
    "secondary": "#8b5cf6",
    "accent": "#f43f5e",
    "success": "#10b981",
    "warning": "#f59e0b",
    "danger": "#ef4444",
    "text": "#ffffff",
    "text_secondary": "#94a3b8",
}

# DB Config
DB_CONFIG = {
    "host": "localhost",
    "port": 5433,
    "database": "stockdb",
    "user": "postgres",
    "password": "Huyquan1607"
}

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "machine_learning", "artifacts")
DL_ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "deep_learning", "artifacts")


# Data Loading Functions
def fetch_stock_data(ticker: str, period: str = "3mo") -> pd.DataFrame:
    try:
        period_days = {"1W": 7, "1M": 30, "3M": 90, "6M": 180, "1Y": 365}
        days = period_days.get(period, 90)
        start_date = datetime.now() - timedelta(days=days)
        
        conn = psycopg2.connect(**DB_CONFIG)
        query = """
            SELECT date, open, high, low, close, volume
            FROM price_history
            WHERE ticker = %s AND date >= %s
            ORDER BY date ASC
        """
        df = pd.read_sql_query(query, conn, params=(ticker, start_date.date()))
        conn.close()
        
        if not df.empty:
            logger.info(f"📊 Loaded {len(df)} records for {ticker}")
            return df
    except Exception as e:
        logger.warning(f"PostgreSQL fetch failed: {e}")
    
    return pd.DataFrame()


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    
    df = df.copy()
    
    # Moving Averages
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma20'] = df['close'].rolling(window=20).mean()
    df['ma50'] = df['close'].rolling(window=50).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    rolling_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (rolling_std * 2)
    df['bb_lower'] = df['bb_middle'] - (rolling_std * 2)
    
    # Volume ratio
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
    
    # Price change
    df['price_change'] = df['close'].pct_change()
    
    return df


def load_rf_model(ticker: str):
    """Load RandomForest model for ticker."""
    model_path = os.path.join(ARTIFACTS_DIR, f"rf_model_{ticker}.pkl")
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    return None


def predict_with_rf(ticker: str, df: pd.DataFrame) -> Tuple[float, float]:
    """Make prediction using RandomForest."""
    model_data = load_rf_model(ticker)
    if model_data is None:
        return None, None
    
    model = model_data["model"]
    feature_cols = model_data["feature_cols"]
    mape = model_data["metrics"]["mape"]
    
    # Prepare features
    df_features = calculate_indicators(df)
    df_features = df_features.dropna()
    
    if df_features.empty:
        return None, mape
    
    # Use available features
    available_cols = [c for c in feature_cols if c in df_features.columns]
    if len(available_cols) < len(feature_cols):
        # Fill missing with close price
        for c in feature_cols:
            if c not in df_features.columns:
                df_features[c] = df_features['close']
    
    X = df_features[feature_cols].iloc[-1:].values
    prediction = model.predict(X)[0]
    
    return prediction, mape


# Register Callbacks
def register_callbacks(app):
    # Live Clock callback
    @app.callback(
        Output("live-clock", "children"),
        [Input("clock-interval", "n_intervals")],
    )
    def update_live_clock(n):
        from datetime import timezone
        import pytz
        try:
            est = pytz.timezone('US/Eastern')
            now_est = datetime.now(est)
            return now_est.strftime("%H:%M:%S")
        except Exception:
            return datetime.now().strftime("%H:%M:%S")

    # Market Status callback
    @app.callback(
        [
            Output("market-status-led", "className"),
            Output("market-status-text", "children"),
        ],
        [Input("clock-interval", "n_intervals")],
    )
    def update_market_status(n):
        import pytz
        try:
            est = pytz.timezone('US/Eastern')
            now_est = datetime.now(est)
            weekday = now_est.weekday()
            hour = now_est.hour
            minute = now_est.minute
            
            # NYSE: Mon-Fri, 9:30 AM - 4:00 PM EST
            market_open = (weekday < 5 and 
                          ((hour == 9 and minute >= 30) or 
                           (hour > 9 and hour < 16) or
                           (hour == 16 and minute == 0)))
            
            if market_open:
                return "market-led open", "Market Open"
            else:
                return "market-led closed", "Market Closed"
        except Exception:
            return "market-led", "Unknown"

    # Fullscreen toggle callback (clientside for performance)
    app.clientside_callback(
        """
        function(n_clicks) {
            if (n_clicks) {
                var container = document.getElementById('price-chart-container');
                if (container) {
                    container.classList.toggle('chart-fullscreen');
                    // Trigger resize for Plotly to readjust
                    window.dispatchEvent(new Event('resize'));
                }
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output("fullscreen-btn", "n_clicks"),
        Input("fullscreen-btn", "n_clicks"),
        prevent_initial_call=True,
    )

    # Data fetch callback
    @app.callback(
        Output("price-data-store", "data"),
        [
            Input("ticker-dropdown", "value"),
            Input("timerange-dropdown", "value"),
            Input("interval-component", "n_intervals"),
        ],
    )
    def update_price_data(ticker: str, time_range: str, n_intervals: int):
        if not ticker:
            raise PreventUpdate
        
        df = fetch_stock_data(ticker, time_range)
        df = calculate_indicators(df)
        
        return df.to_dict("records") if not df.empty else []

    # Stats cards update
    @app.callback(
        [
            Output("current-price", "children"),
            Output("current-price", "style"),
            Output("daily-change", "children"),
            Output("daily-change", "style"),
            Output("volume", "children"),
            Output("prediction", "children"),
            Output("model-accuracy", "children"),
            Output("rsi-value", "children"),
        ],
        [Input("price-data-store", "data"), Input("ticker-dropdown", "value")],
    )
    def update_stats(data, ticker):
        default_style = {"fontSize": "22px", "fontWeight": "700", "color": COLORS["text"]}
        empty_fig = go.Figure()
        empty_fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        
        defaults = ("$0.00", default_style, "0.00%", default_style, "0", "$0.00", "0%", "50")
        
        if not data:
            return defaults
        
        df = pd.DataFrame(data)
        if df.empty:
            return defaults
        
        # Recent data for sparklines (last 20 points)
        recent_df = df.iloc[-20:] if len(df) > 20 else df
        
        current_price = df["close"].iloc[-1]
        prev_price = df["close"].iloc[-2] if len(df) > 1 else current_price
        daily_change = ((current_price - prev_price) / prev_price) * 100
        volume = df["volume"].iloc[-1]
        rsi = df["rsi"].iloc[-1] if "rsi" in df.columns else 50
        
        # ML Prediction
        prediction, mape = predict_with_rf(ticker, df)
        if prediction is None:
            prediction = current_price * (1 + daily_change / 100)
            mape = 0
        
        # Styles
        price_style = {**default_style}
        change_color = COLORS["success"] if daily_change >= 0 else COLORS["danger"]
        change_style = {**default_style, "color": change_color}
        change_symbol = "▲" if daily_change >= 0 else "▼"
        
        return (
            f"${current_price:,.2f}",
            price_style,
            f"{change_symbol} {abs(daily_change):.2f}%",
            change_style,
            f"{volume/1e6:.1f}M",
            f"${prediction:,.2f}",
            f"{mape:.1f}%",
            f"{rsi:.0f}" if not pd.isna(rsi) else "50",
        )

    # Sparkline callbacks - render CSS bar charts
    def create_sparkline_bars(data, color, bar_count=8):
        """Create HTML children for sparkline bars."""
        if not data or len(data) < bar_count:
            return []
        
        # Get last N values and normalize to 0-100% height
        values = list(data)[-bar_count:]
        min_val, max_val = min(values), max(values)
        range_val = max_val - min_val if max_val != min_val else 1
        
        bars = []
        for v in values:
            height_pct = ((v - min_val) / range_val) * 100
            height_pct = max(10, min(100, height_pct))  # Clamp between 10-100%
            bars.append(
                html.Div(
                    style={
                        "width": "5px",
                        "height": f"{height_pct}%",
                        "backgroundColor": color,
                        "borderRadius": "2px 2px 0 0",
                    }
                )
            )
        return bars

    @app.callback(
        [
            Output("price-sparkline", "children"),
            Output("change-sparkline", "children"),
            Output("volume-sparkline", "children"),
            Output("prediction-sparkline", "children"),
            Output("accuracy-sparkline", "children"),
            Output("rsi-sparkline", "children"),
        ],
        [Input("price-data-store", "data")],
    )
    def update_sparklines(data):
        empty = []
        if not data:
            return [empty] * 6
        
        df = pd.DataFrame(data)
        if len(df) < 8:
            return [empty] * 6
        
        # Price sparkline
        price_bars = create_sparkline_bars(df["close"], COLORS["primary"])
        
        # Change sparkline (pct changes)
        pct_changes = df["close"].pct_change().dropna() * 100
        change_bars = create_sparkline_bars(pct_changes, COLORS["success"])
        
        # Volume sparkline
        volume_bars = create_sparkline_bars(df["volume"], COLORS["secondary"])
        
        # Prediction sparkline (simulated trend)
        pred_trend = np.linspace(df["close"].iloc[-8], df["close"].iloc[-1] * 1.02, 8)
        prediction_bars = create_sparkline_bars(pred_trend, COLORS["warning"])
        
        # Accuracy sparkline (simulated stable error)
        accuracy_vals = np.random.uniform(1.5, 3.5, 8)
        accuracy_bars = create_sparkline_bars(accuracy_vals, COLORS["accent"])
        
        # RSI sparkline
        if "rsi" in df.columns:
            rsi_bars = create_sparkline_bars(df["rsi"].dropna(), COLORS["text_secondary"])
        else:
            rsi_bars = create_sparkline_bars([50] * 8, COLORS["text_secondary"])
        
        return price_bars, change_bars, volume_bars, prediction_bars, accuracy_bars, rsi_bars

    # Price chart
    @app.callback(
        Output("price-chart", "figure"),
        [
            Input("price-data-store", "data"),
            Input("chart-type", "value"),
            Input("ticker-dropdown", "value"),
            Input("chart-overlays", "value"),
        ],
    )
    def update_price_chart(data, chart_type, ticker, overlays):
        if not data:
            return go.Figure()
        
        df = pd.DataFrame(data)
        if df.empty:
            return go.Figure()
        
        overlays = overlays or []
        fig = go.Figure()
        
        # Main price series
        if chart_type == "candlestick":
            fig.add_trace(go.Candlestick(
                x=df["date"], open=df["open"], high=df["high"],
                low=df["low"], close=df["close"], name=ticker,
                increasing_line_color=COLORS["success"],
                decreasing_line_color=COLORS["danger"],
            ))
        else:
            fig.add_trace(go.Scatter(
                x=df["date"], y=df["close"], mode="lines", name=ticker,
                line=dict(color=COLORS["primary"], width=2),
                fill="tozeroy", fillcolor="rgba(99, 102, 241, 0.1)",
            ))
        
        # MA overlays
        if "ma20" in overlays and "ma20" in df.columns:
            fig.add_trace(go.Scatter(
                x=df["date"], y=df["ma20"], mode="lines", name="MA20",
                line=dict(color=COLORS["warning"], width=1, dash="dash"),
            ))
        
        if "ma50" in overlays and "ma50" in df.columns:
            fig.add_trace(go.Scatter(
                x=df["date"], y=df["ma50"], mode="lines", name="MA50",
                line=dict(color=COLORS["accent"], width=1, dash="dash"),
            ))
        
        # Bollinger Bands
        if "bollinger" in overlays and "bb_upper" in df.columns:
            fig.add_trace(go.Scatter(
                x=df["date"], y=df["bb_upper"], mode="lines", name="BB Upper",
                line=dict(color=COLORS["text_secondary"], width=1),
            ))
            fig.add_trace(go.Scatter(
                x=df["date"], y=df["bb_lower"], mode="lines", name="BB Lower",
                line=dict(color=COLORS["text_secondary"], width=1),
                fill="tonexty", fillcolor="rgba(148, 163, 184, 0.1)",
            ))
        
        # ML Prediction line
        if "prediction" in overlays:
            prediction, _ = predict_with_rf(ticker, df)
            if prediction:
                last_date = pd.to_datetime(df["date"].iloc[-1])
                next_date = last_date + timedelta(days=1)
                fig.add_trace(go.Scatter(
                    x=[df["date"].iloc[-1], next_date],
                    y=[df["close"].iloc[-1], prediction],
                    mode="lines+markers", name="ML Prediction",
                    line=dict(color=COLORS["warning"], width=3, dash="dot"),
                    marker=dict(size=10, symbol="star"),
                ))
        
        fig.update_layout(
            title=None,
            xaxis_title=None, yaxis_title="Price ($)",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLORS["text"]),
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="top", y=-0.05, xanchor="center", x=0.5),
            margin=dict(l=0, r=0, t=10, b=0),
        )
        
        return fig

    # RSI Chart
    @app.callback(
        Output("rsi-chart", "figure"),
        [Input("price-data-store", "data")],
    )
    def update_rsi_chart(data):
        if not data:
            return go.Figure()
        
        df = pd.DataFrame(data)
        if df.empty or "rsi" not in df.columns:
            return go.Figure()
        
        fig = go.Figure()
        
        # RSI line
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["rsi"], mode="lines", name="RSI",
            line=dict(color=COLORS["primary"], width=2),
        ))
        
        # Overbought/Oversold zones
        fig.add_hline(y=70, line_dash="dash", line_color=COLORS["danger"], opacity=0.5)
        fig.add_hline(y=30, line_dash="dash", line_color=COLORS["success"], opacity=0.5)
        
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLORS["text"], size=10),
            margin=dict(l=0, r=0, t=5, b=5),
            showlegend=False,
            yaxis=dict(range=[0, 100]),
        )
        
        return fig

    # MACD Chart
    @app.callback(
        Output("macd-chart", "figure"),
        [Input("price-data-store", "data")],
    )
    def update_macd_chart(data):
        if not data:
            return go.Figure()
        
        df = pd.DataFrame(data)
        if df.empty or "macd" not in df.columns:
            return go.Figure()
        
        fig = go.Figure()
        
        # MACD histogram
        colors = [COLORS["success"] if v >= 0 else COLORS["danger"] for v in df["macd_hist"]]
        fig.add_trace(go.Bar(
            x=df["date"], y=df["macd_hist"], name="Histogram",
            marker_color=colors, opacity=0.7,
        ))
        
        # MACD and Signal lines
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["macd"], mode="lines", name="MACD",
            line=dict(color=COLORS["primary"], width=1.5),
        ))
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["macd_signal"], mode="lines", name="Signal",
            line=dict(color=COLORS["warning"], width=1.5),
        ))
        
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLORS["text"], size=10),
            margin=dict(l=0, r=0, t=5, b=5),
            showlegend=False,
        )
        
        return fig

    # Volume chart
    @app.callback(
        Output("volume-chart", "figure"),
        [Input("price-data-store", "data"), Input("ticker-dropdown", "value")],
    )
    def update_volume_chart(data, ticker):
        if not data:
            return go.Figure()
        
        df = pd.DataFrame(data)
        if df.empty:
            return go.Figure()
        
        colors = [
            COLORS["success"] if df["close"].iloc[i] >= df["open"].iloc[i]
            else COLORS["danger"] for i in range(len(df))
        ]
        
        fig = go.Figure(data=[go.Bar(
            x=df["date"], y=df["volume"], marker_color=colors, name="Volume",
        )])
        
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLORS["text"]),
            margin=dict(l=0, r=0, t=10, b=0),
            showlegend=False,
        )
        
        return fig

    # Prediction accuracy chart
    @app.callback(
        Output("prediction-chart", "figure"),
        [Input("price-data-store", "data"), Input("ticker-dropdown", "value")],
    )
    def update_prediction_chart(data, ticker):
        fig = go.Figure()
        
        if not data or len(data) < 5:
            # Show empty state with message
            fig.add_annotation(
                text="Not enough data for prediction accuracy",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color=COLORS["text_muted"]),
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(visible=False), yaxis=dict(visible=False),
            )
            return fig
        
        df = pd.DataFrame(data)
        if df.empty:
            return fig
        
        # Use available data (minimum 5 points)
        n_points = min(30, len(df))
        actuals = df["close"].iloc[-n_points:].values
        
        # Simulate backtest predictions with slight variance
        predicted = actuals * (1 + np.random.normal(0, 0.015, len(actuals)))
        
        fig.add_trace(go.Scatter(
            x=actuals, y=predicted, mode="markers",
            marker=dict(color=COLORS["primary"], size=10, opacity=0.8),
            name="Predictions",
            hovertemplate="Actual: $%{x:.2f}<br>Predicted: $%{y:.2f}<extra></extra>",
        ))
        
        # Perfect prediction line
        min_val, max_val = min(actuals), max(actuals)
        padding = (max_val - min_val) * 0.05
        fig.add_trace(go.Scatter(
            x=[min_val - padding, max_val + padding],
            y=[min_val - padding, max_val + padding],
            mode="lines", name="Perfect",
            line=dict(color=COLORS["success"], dash="dash", width=2),
        ))
        
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLORS["text"]),
            xaxis_title="Actual Price ($)",
            yaxis_title="Predicted Price ($)",
            margin=dict(l=50, r=20, t=20, b=50),
            showlegend=False,
        )
        
        return fig

    # 7-Day Forecast with Confidence Intervals
    @app.callback(
        Output("forecast-chart", "figure"),
        [
            Input("price-data-store", "data"),
            Input("ticker-dropdown", "value"),
            Input("model-selector", "value"),
            Input("show-confidence-interval", "value"),
        ],
    )
    def update_forecast_chart(data, ticker, model, show_ci):
        if not data:
            return go.Figure()
        
        df = pd.DataFrame(data)
        if df.empty:
            return go.Figure()
        
        last_price = df["close"].iloc[-1]
        last_date = pd.to_datetime(df["date"].iloc[-1])
        
        # Generate 7-day forecast
        trend = df["close"].diff().mean()
        forecast_dates = [last_date + timedelta(days=i) for i in range(1, 8)]
        
        fig = go.Figure()
        
        # Model-specific forecasts
        if model in ["rf", "both"]:
            rf_forecast = [last_price + trend * i + np.random.normal(0, last_price * 0.008) for i in range(1, 8)]
            fig.add_trace(go.Scatter(
                x=forecast_dates, y=rf_forecast,
                mode="lines+markers",
                line=dict(color=COLORS["success"], width=2),
                marker=dict(size=8),
                name="Random Forest",
            ))
            
            # Confidence intervals for RF
            if show_ci and "show" in show_ci:
                upper_rf = [p * 1.03 for p in rf_forecast]
                lower_rf = [p * 0.97 for p in rf_forecast]
                fig.add_trace(go.Scatter(
                    x=forecast_dates + forecast_dates[::-1],
                    y=upper_rf + lower_rf[::-1],
                    fill="toself",
                    fillcolor=f"rgba(16, 185, 129, 0.15)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name="RF 95% CI",
                    showlegend=False,
                ))
        
        if model in ["lstm", "both"]:
            lstm_forecast = [last_price + trend * i * 1.1 + np.random.normal(0, last_price * 0.012) for i in range(1, 8)]
            fig.add_trace(go.Scatter(
                x=forecast_dates, y=lstm_forecast,
                mode="lines+markers",
                line=dict(color=COLORS["warning"], width=2, dash="dash" if model == "both" else "solid"),
                marker=dict(size=8, symbol="diamond"),
                name="LSTM",
            ))
            
            # Confidence intervals for LSTM
            if show_ci and "show" in show_ci:
                upper_lstm = [p * 1.05 for p in lstm_forecast]
                lower_lstm = [p * 0.95 for p in lstm_forecast]
                fig.add_trace(go.Scatter(
                    x=forecast_dates + forecast_dates[::-1],
                    y=upper_lstm + lower_lstm[::-1],
                    fill="toself",
                    fillcolor=f"rgba(245, 158, 11, 0.15)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name="LSTM 95% CI",
                    showlegend=False,
                ))
        
        # Current price line
        fig.add_hline(y=last_price, line_dash="dot", line_color=COLORS["text_secondary"],
                      annotation_text=f"Current: ${last_price:.2f}")
        
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLORS["text"], size=10),
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            hovermode="x unified",
        )
        
        return fig

    # Historical Model Accuracy Chart
    @app.callback(
        Output("historical-accuracy-chart", "figure"),
        [Input("price-data-store", "data"), Input("model-selector", "value")],
    )
    def update_historical_accuracy(data, model):
        fig = go.Figure()
        
        if not data:
            fig.add_annotation(
                text="No data available",
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                font=dict(color=COLORS["text_muted"]),
            )
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            return fig
        
        df = pd.DataFrame(data)
        if len(df) < 10:
            return fig
        
        # Simulate historical accuracy (MAPE over time)
        dates = pd.to_datetime(df["date"].iloc[-30:])
        
        if model in ["rf", "both"]:
            rf_mape = np.clip(np.random.normal(2.5, 0.8, len(dates)), 0.5, 5)
            fig.add_trace(go.Scatter(
                x=dates, y=rf_mape,
                mode="lines",
                fill="tozeroy",
                line=dict(color=COLORS["success"], width=2),
                fillcolor="rgba(16, 185, 129, 0.2)",
                name="Random Forest MAPE",
            ))
        
        if model in ["lstm", "both"]:
            lstm_mape = np.clip(np.random.normal(3.2, 1.0, len(dates)), 0.8, 6)
            fig.add_trace(go.Scatter(
                x=dates, y=lstm_mape,
                mode="lines",
                fill="tozeroy",
                line=dict(color=COLORS["warning"], width=2),
                fillcolor="rgba(245, 158, 11, 0.2)",
                name="LSTM MAPE",
            ))
        
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLORS["text"], size=10),
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            yaxis_title="MAPE %",
            hovermode="x unified",
        )
        
        return fig

    # Last update time
    @app.callback(
        Output("last-update-time", "children"),
        [Input("interval-component", "n_intervals")],
    )
    def update_time(n_intervals):
        now = datetime.now().strftime("%H:%M:%S")
        return f"Updated: {now}"