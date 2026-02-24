from typing import List, Dict, Any, Optional
import plotly.graph_objs as go
from dash import dcc
import pandas as pd

COLORS = {
    "background": "#1a1a2e",
    "card": "#16213e",
    "primary": "#0f3460",
    "accent": "#e94560",
    "text": "#ffffff",
    "success": "#00ff88",
    "danger": "#ff4444",
}

# Candlestick Chart

def create_candlestick_chart(
    df: pd.DataFrame,
    ticker: str = "",
    height: int = 500,
) -> dcc.Graph:
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df["date"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name=ticker,
                increasing_line_color=COLORS["success"],
                decreasing_line_color=COLORS["danger"],
            )
        ]
    )

    fig.update_layout(
        title=f"{ticker} Price Chart",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        template="plotly_dark",
        paper_bgcolor=COLORS["card"],
        plot_bgcolor=COLORS["card"],
        font=dict(color=COLORS["text"]),
        xaxis_rangeslider_visible=False,
        height=height,
    )

    return dcc.Graph(figure=fig, config={"displayModeBar": True})


# Line Chart

def create_line_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str = "",
    color: str = None,
    fill: bool = True,
    height: int = 400,
) -> dcc.Graph:
    color = color or COLORS["accent"]

    fig = go.Figure(
        data=[
            go.Scatter(
                x=df[x_col],
                y=df[y_col],
                mode="lines",
                line=dict(color=color, width=2),
                fill="tozeroy" if fill else None,
                fillcolor=f"rgba{tuple(list(bytes.fromhex(color[1:])) + [0.1])}",
            )
        ]
    )

    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor=COLORS["card"],
        plot_bgcolor=COLORS["card"],
        font=dict(color=COLORS["text"]),
        height=height,
    )

    return dcc.Graph(figure=fig, config={"displayModeBar": True})


# Volume Bar Chart

def create_volume_chart(
    df: pd.DataFrame,
    height: int = 250,
) -> dcc.Graph:
    colors = [
        COLORS["success"] if df["close"].iloc[i] >= df["open"].iloc[i]
        else COLORS["danger"]
        for i in range(len(df))
    ]

    fig = go.Figure(
        data=[
            go.Bar(
                x=df["date"],
                y=df["volume"],
                marker_color=colors,
            )
        ]
    )

    fig.update_layout(
        title="Trading Volume",
        xaxis_title="Date",
        yaxis_title="Volume",
        template="plotly_dark",
        paper_bgcolor=COLORS["card"],
        plot_bgcolor=COLORS["card"],
        font=dict(color=COLORS["text"]),
        height=height,
        showlegend=False,
    )

    return dcc.Graph(figure=fig, config={"displayModeBar": True})


# Prediction Comparison Chart

def create_prediction_chart(
    actual: pd.Series,
    predicted: pd.Series,
    dates: pd.Series,
    height: int = 350,
) -> dcc.Graph:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=actual,
            mode="lines",
            name="Actual",
            line=dict(color=COLORS["accent"], width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=predicted,
            mode="lines",
            name="Predicted",
            line=dict(color=COLORS["success"], width=2, dash="dash"),
        )
    )

    fig.update_layout(
        title="Prediction vs Actual",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        template="plotly_dark",
        paper_bgcolor=COLORS["card"],
        plot_bgcolor=COLORS["card"],
        font=dict(color=COLORS["text"]),
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    return dcc.Graph(figure=fig, config={"displayModeBar": True})


# Metrics Gauge

def create_gauge(
    value: float,
    title: str,
    min_val: float = 0,
    max_val: float = 100,
    height: int = 200,
) -> dcc.Graph:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": title, "font": {"color": COLORS["text"]}},
            gauge={
                "axis": {"range": [min_val, max_val]},
                "bar": {"color": COLORS["accent"]},
                "bgcolor": COLORS["card"],
                "borderwidth": 0,
                "steps": [
                    {"range": [min_val, max_val * 0.3], "color": COLORS["danger"]},
                    {"range": [max_val * 0.3, max_val * 0.7], "color": COLORS["primary"]},
                    {"range": [max_val * 0.7, max_val], "color": COLORS["success"]},
                ],
            },
        )
    )

    fig.update_layout(
        paper_bgcolor=COLORS["card"],
        font=dict(color=COLORS["text"]),
        height=height,
    )

    return dcc.Graph(figure=fig, config={"displayModeBar": False})