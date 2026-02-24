from typing import List, Dict, Any, Optional
from dash import html, dash_table
import pandas as pd


# Theme Colors

COLORS = {
    "background": "#1a1a2e",
    "card": "#16213e",
    "primary": "#0f3460",
    "text": "#ffffff",
    "text_secondary": "#a0a0a0",
    "success": "#00ff88",
    "danger": "#ff4444",
}


# Stock Data Table

def create_stock_table(
    df: pd.DataFrame,
    columns: List[str] = None,
    page_size: int = 10,
    sortable: bool = True,
) -> dash_table.DataTable:
    if columns:
        df = df[columns]

    return dash_table.DataTable(
        data=df.to_dict("records"),
        columns=[{"name": col, "id": col} for col in df.columns],
        page_size=page_size,
        sort_action="native" if sortable else "none",
        filter_action="native",
        style_table={
            "overflowX": "auto",
        },
        style_header={
            "backgroundColor": COLORS["primary"],
            "color": COLORS["text"],
            "fontWeight": "bold",
            "textAlign": "center",
        },
        style_cell={
            "backgroundColor": COLORS["card"],
            "color": COLORS["text"],
            "textAlign": "center",
            "padding": "10px",
            "border": f"1px solid {COLORS['primary']}",
        },
        style_data_conditional=[
            {
                "if": {"row_index": "odd"},
                "backgroundColor": COLORS["background"],
            }
        ],
    )


# Prediction Results Table

def create_prediction_table(predictions: List[Dict]) -> dash_table.DataTable:
    df = pd.DataFrame(predictions)

    return dash_table.DataTable(
        data=df.to_dict("records"),
        columns=[
            {"name": "Ticker", "id": "ticker"},
            {"name": "Predicted", "id": "predicted_close", "type": "numeric", "format": {"specifier": "$.2f"}},
            {"name": "Confidence", "id": "confidence", "type": "numeric", "format": {"specifier": ".0%"}},
            {"name": "Model", "id": "model_name"},
            {"name": "Time", "id": "prediction_time"},
        ],
        style_header={
            "backgroundColor": COLORS["primary"],
            "color": COLORS["text"],
            "fontWeight": "bold",
        },
        style_cell={
            "backgroundColor": COLORS["card"],
            "color": COLORS["text"],
            "textAlign": "center",
            "padding": "8px",
        },
    )


# Summary Stats Table

def create_stats_table(stats: Dict[str, Any]) -> html.Div:
    rows = []

    for key, value in stats.items():
        rows.append(
            html.Tr([
                html.Td(
                    key,
                    style={
                        "color": COLORS["text_secondary"],
                        "padding": "8px 16px",
                        "borderBottom": f"1px solid {COLORS['primary']}",
                    },
                ),
                html.Td(
                    str(value),
                    style={
                        "color": COLORS["text"],
                        "fontWeight": "bold",
                        "padding": "8px 16px",
                        "borderBottom": f"1px solid {COLORS['primary']}",
                        "textAlign": "right",
                    },
                ),
            ])
        )

    return html.Table(
        html.Tbody(rows),
        style={
            "width": "100%",
            "backgroundColor": COLORS["card"],
            "borderRadius": "8px",
        },
    )


# Ticker List Table

def create_ticker_list(
    tickers: List[Dict[str, Any]],
    on_click_id: str = "ticker-click",
) -> html.Div:
    items = []

    for ticker in tickers:
        change = ticker.get("change", 0)
        change_color = COLORS["success"] if change >= 0 else COLORS["danger"]
        change_symbol = "▲" if change >= 0 else "▼"

        items.append(
            html.Div(
                [
                    html.Span(
                        ticker.get("symbol", ""),
                        style={
                            "fontWeight": "bold",
                            "color": COLORS["text"],
                        },
                    ),
                    html.Span(
                        f" ${ticker.get('price', 0):.2f}",
                        style={"color": COLORS["text_secondary"], "marginLeft": "10px"},
                    ),
                    html.Span(
                        f" {change_symbol} {abs(change):.2f}%",
                        style={"color": change_color, "marginLeft": "10px"},
                    ),
                ],
                style={
                    "padding": "10px 15px",
                    "borderBottom": f"1px solid {COLORS['primary']}",
                    "cursor": "pointer",
                },
                className="ticker-item",
                id={"type": on_click_id, "index": ticker.get("symbol", "")},
            )
        )

    return html.Div(
        items,
        style={
            "backgroundColor": COLORS["card"],
            "borderRadius": "8px",
            "maxHeight": "400px",
            "overflowY": "auto",
        },
    )