from dash import html, dcc
import plotly.graph_objs as go

# Theme Colors - Premium Dark Theme
COLORS = {
    "background": "#0f0f1a",
    "card": "rgba(26, 26, 46, 0.8)",
    "card_border": "rgba(255, 255, 255, 0.1)",
    "primary": "#6366f1",
    "secondary": "#8b5cf6",
    "accent": "#f43f5e",
    "success": "#10b981",
    "warning": "#f59e0b",
    "danger": "#ef4444",
    "text": "#ffffff",
    "text_secondary": "#94a3b8",
    "text_muted": "#64748b",
    "tab_active": "#6366f1",
    "tab_inactive": "rgba(26, 26, 46, 0.6)",
}


# Common Styles
GLASS_CARD_STYLE = {
    "backgroundColor": COLORS["card"],
    "borderRadius": "16px",
    "padding": "20px",
    "marginBottom": "16px",
    "border": f"1px solid {COLORS['card_border']}",
    "backdropFilter": "blur(10px)",
    "boxShadow": "0 8px 32px rgba(0, 0, 0, 0.3)",
}

HEADER_STYLE = {
    "color": COLORS["text"],
    "marginBottom": "16px",
    "fontWeight": "600",
    "fontSize": "16px",
    "display": "flex",
    "alignItems": "center",
    "gap": "8px",
}

TAB_STYLE = {
    "backgroundColor": COLORS["tab_inactive"],
    "color": COLORS["text_secondary"],
    "border": "none",
    "borderRadius": "12px 12px 0 0",
    "padding": "14px 28px",
    "fontWeight": "600",
    "fontSize": "14px",
    "cursor": "pointer",
    "marginRight": "4px",
}

TAB_SELECTED_STYLE = {
    "backgroundColor": COLORS["tab_active"],
    "color": COLORS["text"],
    "border": "none",
    "borderRadius": "12px 12px 0 0",
    "padding": "14px 28px",
    "fontWeight": "600",
    "fontSize": "14px",
    "cursor": "pointer",
    "marginRight": "4px",
    "boxShadow": "0 -4px 20px rgba(99, 102, 241, 0.3)",
}


# Header Component - Premium Design
def create_header():
    return html.Div(
        [
            # Animated gradient background layer
            html.Div(className="header-gradient-bg"),
            
            # Left section - Market Status
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(
                                id="market-status-led",
                                className="market-led",
                            ),
                            html.Span(
                                id="market-status-text",
                                style={
                                    "color": COLORS["text_secondary"],
                                    "fontSize": "12px",
                                    "fontWeight": "500",
                                },
                            ),
                        ],
                        style={"display": "flex", "alignItems": "center", "gap": "8px"},
                    ),
                ],
                style={"position": "absolute", "left": "24px", "top": "50%", "transform": "translateY(-50%)"},
            ),
            
            # Center - Title with glow
            html.Div(
                [
                    html.H1(
                        [
                            html.Span("📈 ", style={"marginRight": "8px"}),
                            html.Span("Stock Prediction", className="title-glow"),
                            html.Span(" Dashboard", style={"fontWeight": "400", "opacity": "0.9"}),
                        ],
                        className="header-title",
                        style={
                            "color": COLORS["text"],
                            "margin": "0",
                            "fontSize": "26px",
                            "fontWeight": "700",
                            "textAlign": "center",
                            "letterSpacing": "-0.5px",
                        },
                    ),
                ],
                style={"flex": "1", "display": "flex", "justifyContent": "center"},
            ),
            
            # Right section - Live Clock, Last Update, Theme Toggle
            html.Div(
                [
                    # Live Clock
                    html.Div(
                        [
                            html.Span("🕐", style={"marginRight": "6px", "fontSize": "14px"}),
                            html.Span(
                                id="live-clock",
                                style={
                                    "color": COLORS["text"],
                                    "fontSize": "14px",
                                    "fontWeight": "600",
                                    "fontFamily": "'JetBrains Mono', monospace",
                                },
                            ),
                            html.Span(
                                " EST",
                                style={"color": COLORS["text_muted"], "fontSize": "11px", "marginLeft": "4px"},
                            ),
                        ],
                        style={
                            "display": "flex",
                            "alignItems": "center",
                            "backgroundColor": "rgba(255, 255, 255, 0.05)",
                            "padding": "6px 12px",
                            "borderRadius": "8px",
                            "marginRight": "12px",
                        },
                    ),
                    # Last Update with Refresh Indicator
                    html.Div(
                        [
                            html.Span(className="refresh-indicator"),
                            html.Span(id="last-update-time", children="Updated just now"),
                        ],
                        style={
                            "color": COLORS["text_muted"],
                            "fontSize": "11px",
                            "display": "flex",
                            "alignItems": "center",
                        },
                    ),
                ],
                style={
                    "position": "absolute",
                    "right": "24px",
                    "top": "50%",
                    "transform": "translateY(-50%)",
                    "display": "flex",
                    "alignItems": "center",
                },
            ),
            
            # Interval for live clock
            dcc.Interval(id="clock-interval", interval=1000, n_intervals=0),
        ],
        className="premium-header",
        style={
            "position": "relative",
            "padding": "20px 24px",
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "center",
            "borderBottom": f"1px solid {COLORS['card_border']}",
            "overflow": "hidden",
        },
    )


# Controls Component - Premium Redesign
def create_controls():
    """Create premium control panel with modern styling."""
    
    # Control item wrapper style
    control_item_style = {
        "backgroundColor": "rgba(15, 15, 26, 0.6)",
        "borderRadius": "12px",
        "padding": "12px 16px",
        "border": f"1px solid {COLORS['card_border']}",
        "marginRight": "12px",
    }
    
    label_style = {
        "color": COLORS["text_muted"],
        "fontSize": "11px",
        "fontWeight": "600",
        "textTransform": "uppercase",
        "letterSpacing": "0.5px",
        "marginBottom": "8px",
        "display": "block",
    }
    
    dropdown_style = {
        "backgroundColor": COLORS["background"],
        "border": "none",
        "borderRadius": "8px",
        "color": COLORS["text"],
        "minWidth": "160px",
    }
    
    return html.Div(
        [
            # Left side - Dropdowns
            html.Div(
                [
                    # Ticker Dropdown
                    html.Div(
                        [
                            html.Label("STOCK", style=label_style),
                            dcc.Dropdown(
                                id="ticker-dropdown",
                                options=[
                                    {"label": "Apple (AAPL)", "value": "AAPL"},
                                    {"label": "Google (GOOGL)", "value": "GOOGL"},
                                    {"label": "Microsoft (MSFT)", "value": "MSFT"},
                                    {"label": "Amazon (AMZN)", "value": "AMZN"},
                                    {"label": "Meta (META)", "value": "META"},
                                    {"label": "NVIDIA (NVDA)", "value": "NVDA"},
                                    {"label": "Tesla (TSLA)", "value": "TSLA"},
                                    {"label": "JPMorgan (JPM)", "value": "JPM"},
                                    {"label": "Visa (V)", "value": "V"},
                                    {"label": "Walmart (WMT)", "value": "WMT"},
                                ],
                                value="AAPL",
                                clearable=False,
                                style={**dropdown_style, "zIndex": 1000},
                            ),
                        ],
                        style=control_item_style,
                    ),
                    # Time Range Dropdown
                    html.Div(
                        [
                            html.Label("PERIOD", style=label_style),
                            dcc.Dropdown(
                                id="timerange-dropdown",
                                options=[
                                    {"label": "1W", "value": "1W"},
                                    {"label": "1M", "value": "1M"},
                                    {"label": "3M", "value": "3M"},
                                    {"label": "6M", "value": "6M"},
                                    {"label": "1Y", "value": "1Y"},
                                ],
                                value="3M",
                                clearable=False,
                                style={**dropdown_style, "minWidth": "80px"},
                            ),
                        ],
                        style={**control_item_style, "minWidth": "100px"},
                    ),
                ],
                style={"display": "flex", "alignItems": "stretch"},
            ),
            # Right side - Chart Type Toggle
            html.Div(
                [
                    html.Label("CHART TYPE", style=label_style),
                    html.Div(
                        [
                            dcc.RadioItems(
                                id="chart-type",
                                options=[
                                    {"label": "📊 Candlestick", "value": "candlestick"},
                                    {"label": "📈 Line", "value": "line"},
                                ],
                                value="candlestick",
                                inline=True,
                                style={"display": "flex", "gap": "8px"},
                                inputStyle={"display": "none"},
                                labelStyle={
                                    "backgroundColor": "rgba(99, 102, 241, 0.1)",
                                    "color": COLORS["text_secondary"],
                                    "padding": "8px 16px",
                                    "borderRadius": "8px",
                                    "cursor": "pointer",
                                    "fontSize": "13px",
                                    "fontWeight": "500",
                                    "border": f"1px solid {COLORS['card_border']}",
                                    "transition": "all 0.2s ease",
                                },
                                className="chart-type-toggle",
                            ),
                        ],
                    ),
                ],
                style={**control_item_style, "marginRight": "0", "marginLeft": "auto"},
            ),
        ],
        style={
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "space-between",
            "padding": "16px 20px",
            "marginBottom": "16px",
            "background": f"linear-gradient(135deg, {COLORS['card']} 0%, rgba(99, 102, 241, 0.05) 100%)",
            "borderRadius": "16px",
            "border": f"1px solid {COLORS['card_border']}",
            "backdropFilter": "blur(10px)",
            "position": "relative",
            "zIndex": 100,
        },
    )



# Watchlist Panel
def create_watchlist_panel():
    """Create a collapsible watchlist panel with quick search and favorites."""
    default_watchlist = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "META"]
    
    return html.Div(
        [
            # Panel Header
            html.Div(
                [
                    html.Span("⭐", style={"fontSize": "16px"}),
                    html.Span("Watchlist", style={"fontWeight": "600", "color": COLORS["text"], "fontSize": "14px"}),
                    html.Button(
                        "−",
                        id="watchlist-toggle",
                        style={
                            "marginLeft": "auto",
                            "background": "transparent",
                            "border": "none",
                            "color": COLORS["text_muted"],
                            "fontSize": "18px",
                            "cursor": "pointer",
                        },
                    ),
                ],
                style={"display": "flex", "alignItems": "center", "gap": "8px", "marginBottom": "12px"},
            ),
            # Quick Search
            html.Div(
                [
                    dcc.Input(
                        id="watchlist-search",
                        type="text",
                        placeholder="🔍 Search ticker...",
                        style={
                            "width": "100%",
                            "padding": "8px 12px",
                            "borderRadius": "8px",
                            "border": f"1px solid {COLORS['card_border']}",
                            "backgroundColor": "rgba(15, 15, 26, 0.8)",
                            "color": COLORS["text"],
                            "fontSize": "12px",
                        },
                    ),
                ],
                style={"marginBottom": "12px"},
            ),
            # Watchlist Body
            html.Div(
                id="watchlist-body",
                children=[
                    create_watchlist_item(ticker, i == 0) for i, ticker in enumerate(default_watchlist)
                ],
                style={"maxHeight": "250px", "overflowY": "auto"},
            ),
            # Store for favorites
            dcc.Store(id="favorites-store", data=["AAPL", "TSLA"]),
        ],
        id="watchlist-panel",
        style={
            **GLASS_CARD_STYLE,
            "padding": "14px",
            "marginBottom": "16px",
            "transition": "all 0.3s ease",
        },
    )


def create_watchlist_item(ticker: str, is_active: bool = False):
    """Create a single watchlist item row."""
    return html.Div(
        [
            # Favorite star
            html.Button(
                "★",
                id={"type": "favorite-btn", "ticker": ticker},
                n_clicks=0,
                style={
                    "background": "transparent",
                    "border": "none",
                    "color": COLORS["warning"],
                    "fontSize": "14px",
                    "cursor": "pointer",
                    "padding": "0",
                    "opacity": "0.4",
                },
                className="favorite-star",
            ),
            # Ticker name
            html.Span(
                ticker,
                style={
                    "fontWeight": "600",
                    "color": COLORS["text"] if is_active else COLORS["text_secondary"],
                    "fontSize": "13px",
                    "cursor": "pointer",
                },
                id={"type": "watchlist-ticker", "ticker": ticker},
            ),
            # Mini price (placeholder)
            html.Span(
                id={"type": "watchlist-price", "ticker": ticker},
                children="—",
                style={
                    "marginLeft": "auto",
                    "color": COLORS["text_muted"],
                    "fontSize": "11px",
                },
            ),
            # Change indicator
            html.Span(
                id={"type": "watchlist-change", "ticker": ticker},
                children="",
                style={"fontSize": "10px", "marginLeft": "4px"},
            ),
        ],
        style={
            "display": "flex",
            "alignItems": "center",
            "gap": "8px",
            "padding": "8px 6px",
            "borderRadius": "6px",
            "transition": "background 0.2s ease",
            "cursor": "pointer",
            "backgroundColor": "rgba(99, 102, 241, 0.1)" if is_active else "transparent",
        },
        className="watchlist-item",
    )


# Stats Cards - 6 Cards Layout
def create_stats_cards():
    """Create 6 modern stats cards with sparklines and tooltips."""
    return html.Div(
        [
            create_stat_card("current-price", "Current Price", "$0.00", "💰", COLORS["primary"], "price-sparkline", "Latest closing price"),
            create_stat_card("daily-change", "Daily Change", "0.00%", "📊", COLORS["success"], "change-sparkline", "Percentage change from previous close"),
            create_stat_card("volume", "Volume", "0", "📈", COLORS["secondary"], "volume-sparkline", "Trading volume in millions"),
            create_stat_card("prediction", "ML Prediction", "$0.00", "🤖", COLORS["warning"], "prediction-sparkline", "Random Forest model prediction"),
            create_stat_card("model-accuracy", "Model MAPE", "0%", "🎯", COLORS["accent"], "accuracy-sparkline", "Mean Absolute Percentage Error"),
            create_stat_card("rsi-value", "RSI (14)", "50", "📉", COLORS["text_secondary"], "rsi-sparkline", "Relative Strength Index (14-day)"),
        ],
        style={
            "display": "grid",
            "gridTemplateColumns": "repeat(6, 1fr)",
            "gap": "12px",
            "marginBottom": "16px",
        },
    )


def create_stat_card(card_id: str, title: str, default_value: str, icon: str, accent_color: str, sparkline_id: str = None, tooltip_text: str = ""):
    """Create a single modern stat card with sparkline and tooltip."""
    return html.Div(
        [
            # Header row
            html.Div(
                [
                    html.Span(icon, style={"fontSize": "18px"}),
                    html.Span(title, style={"color": COLORS["text_muted"], "fontSize": "11px", "textTransform": "uppercase", "fontWeight": "500"}),
                ],
                style={"display": "flex", "alignItems": "center", "gap": "6px", "marginBottom": "6px"},
            ),
            # Value and Sparkline row
            html.Div(
                [
                    # Value with tooltip trigger
                    html.Div(
                        id=card_id,
                        children=default_value,
                        className="stats-value animated-counter",
                        title=tooltip_text,
                        style={
                            "fontSize": "20px",
                            "fontWeight": "700",
                            "color": COLORS["text"],
                            "whiteSpace": "nowrap",
                        },
                    ),
                    # Mini sparkline chart
                    html.Div(
                        id=sparkline_id,
                        className="sparkline-container",
                        style={
                            "width": "60px",
                            "height": "30px",
                            "marginLeft": "auto",
                        },
                    ) if sparkline_id else None,
                ],
                style={"display": "flex", "alignItems": "flex-end", "justifyContent": "space-between"},
            ),
        ],
        style={
            **GLASS_CARD_STYLE,
            "borderLeft": f"3px solid {accent_color}",
            "padding": "12px 14px",
            "transition": "all 0.3s ease",
            "cursor": "default",
        },
        className="stats-card-hover",
    )


# TAB 1: Overview Content
def create_overview_tab():
    """Create Overview tab content - Price chart + Volume."""
    return html.Div(
        [
            # Price Chart
            html.Div(
                [
                    html.H3(["📈 ", "Price Chart"], style=HEADER_STYLE),
                    html.Div(
                        [
                            html.Span("Overlays: ", style={"color": COLORS["text_muted"], "fontSize": "12px", "marginRight": "8px"}),
                            dcc.Checklist(
                                id="chart-overlays",
                                options=[
                                    {"label": " MA20", "value": "ma20"},
                                    {"label": " MA50", "value": "ma50"},
                                    {"label": " Bollinger", "value": "bollinger"},
                                    {"label": " Prediction", "value": "prediction"},
                                ],
                                value=["ma20", "prediction"],
                                inline=True,
                                style={"color": COLORS["text_secondary"], "fontSize": "12px", "display": "inline-flex"},
                                inputStyle={"marginRight": "4px"},
                                labelStyle={"marginRight": "16px"},
                            ),
                        ],
                        style={"marginBottom": "10px", "display": "flex", "alignItems": "center"},
                    ),
                    # Chart toolbar
                    html.Div(
                        [
                            html.Button(
                                "⛶ Fullscreen",
                                id="fullscreen-btn",
                                className="chart-toolbar-btn",
                                style={
                                    "background": "rgba(99, 102, 241, 0.2)",
                                    "border": f"1px solid {COLORS['primary']}",
                                    "borderRadius": "6px",
                                    "color": COLORS["text"],
                                    "padding": "6px 12px",
                                    "fontSize": "11px",
                                    "cursor": "pointer",
                                    "marginLeft": "auto",
                                },
                            ),
                        ],
                        style={"display": "flex", "alignItems": "center", "marginBottom": "8px"},
                    ),
                    dcc.Graph(
                        id="price-chart",
                        config={
                            "displayModeBar": True,
                            "scrollZoom": True,
                            "modeBarButtonsToAdd": [
                                "drawline",
                                "drawopenpath",
                                "drawclosedpath",
                                "drawrect",
                                "eraseshape",
                            ],
                            "toImageButtonOptions": {
                                "format": "png",
                                "filename": "stock_chart",
                                "height": 800,
                                "width": 1200,
                                "scale": 2,
                            },
                        },
                        style={"height": "450px"},
                    ),
                ],
                style={**GLASS_CARD_STYLE, "position": "relative"},
                id="price-chart-container",
            ),
            # Volume Chart
            html.Div(
                [
                    html.H3(["📊 ", "Trading Volume"], style=HEADER_STYLE),
                    dcc.Graph(
                        id="volume-chart",
                        config={"displayModeBar": False},
                        style={"height": "180px"},
                    ),
                ],
                style=GLASS_CARD_STYLE,
            ),
        ],
    )


# TAB 2: Technical Analysis Content
def create_technical_tab():
    return html.Div(
        [
            # RSI Chart
            html.Div(
                [
                    html.H3(["📉 ", "RSI (Relative Strength Index)"], style=HEADER_STYLE),
                    html.P(
                        "RSI > 70: Overbought | RSI < 30: Oversold",
                        style={"color": COLORS["text_muted"], "fontSize": "12px", "marginBottom": "10px"},
                    ),
                    dcc.Graph(id="rsi-chart", config={"displayModeBar": False}, style={"height": "250px"}),
                ],
                style=GLASS_CARD_STYLE,
            ),
            # MACD Chart
            html.Div(
                [
                    html.H3(["📊 ", "MACD (Moving Average Convergence Divergence)"], style=HEADER_STYLE),
                    html.P(
                        "MACD crossover signal line = Buy/Sell signal",
                        style={"color": COLORS["text_muted"], "fontSize": "12px", "marginBottom": "10px"},
                    ),
                    dcc.Graph(id="macd-chart", config={"displayModeBar": False}, style={"height": "250px"}),
                ],
                style=GLASS_CARD_STYLE,
            ),
        ],
    )


# TAB 3: ML Predictions Content
def create_predictions_tab():
    return html.Div(
        [
            # Model Selection Row
            html.Div(
                [
                    html.Span("Model: ", style={"color": COLORS["text_muted"], "fontSize": "12px", "marginRight": "8px"}),
                    dcc.Dropdown(
                        id="model-selector",
                        options=[
                            {"label": "🌲 Random Forest", "value": "rf"},
                            {"label": "🧠 LSTM Neural Network", "value": "lstm"},
                            {"label": "📊 Compare Both", "value": "both"},
                        ],
                        value="rf",
                        clearable=False,
                        style={
                            "width": "200px",
                            "backgroundColor": "rgba(26, 26, 46, 0.8)",
                            "border": "1px solid rgba(99, 102, 241, 0.3)",
                            "borderRadius": "8px",
                        },
                    ),
                    html.Div(
                        [
                            dcc.Checklist(
                                id="show-confidence-interval",
                                options=[{"label": " Show 95% Confidence Interval", "value": "show"}],
                                value=["show"],
                                style={"color": COLORS["text_secondary"], "fontSize": "12px"},
                                inputStyle={"marginRight": "6px"},
                            ),
                        ],
                        style={"marginLeft": "24px"},
                    ),
                ],
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "marginBottom": "16px",
                    "padding": "12px 16px",
                    "overflow": "visible",
                    "zIndex": "1000",
                    "position": "relative",
                    **GLASS_CARD_STYLE,
                },
            ),
            # Row with 2 charts
            html.Div(
                [
                    # Prediction Accuracy
                    html.Div(
                        [
                            html.H3(["🎯 ", "Prediction vs Actual"], style=HEADER_STYLE),
                            html.P(
                                "Actual prices overlaid with model predictions",
                                style={"color": COLORS["text_muted"], "fontSize": "12px", "marginBottom": "10px"},
                            ),
                            dcc.Graph(
                                id="prediction-chart",
                                config={"displayModeBar": False},
                                style={"height": "350px"},
                            ),
                        ],
                        style={**GLASS_CARD_STYLE, "flex": "1"},
                    ),
                    # 7-Day Forecast with Confidence Intervals
                    html.Div(
                        [
                            html.H3(["🔮 ", "7-Day Forecast"], style=HEADER_STYLE),
                            html.P(
                                "Next 7 trading days with confidence bands",
                                style={"color": COLORS["text_muted"], "fontSize": "12px", "marginBottom": "10px"},
                            ),
                            dcc.Graph(
                                id="forecast-chart",
                                config={"displayModeBar": False},
                                style={"height": "350px"},
                            ),
                        ],
                        style={**GLASS_CARD_STYLE, "flex": "1", "marginLeft": "16px"},
                    ),
                ],
                style={"display": "flex", "marginBottom": "16px"},
            ),
            # Historical Accuracy Chart Row
            html.Div(
                [
                    html.H3(["📈 ", "Historical Model Accuracy"], style=HEADER_STYLE),
                    html.P(
                        "Model performance over time (MAPE % - lower is better)",
                        style={"color": COLORS["text_muted"], "fontSize": "12px", "marginBottom": "10px"},
                    ),
                    dcc.Graph(
                        id="historical-accuracy-chart",
                        config={"displayModeBar": False},
                        style={"height": "250px"},
                    ),
                ],
                style=GLASS_CARD_STYLE,
            ),
        ],
    )


# Main Layout with Tabs
def create_layout():
    return html.Div(
        [
            # Auto-refresh interval
            dcc.Interval(id="interval-component", interval=60 * 1000, n_intervals=0),
            
            # Data stores
            dcc.Store(id="price-data-store"),
            dcc.Store(id="indicators-data-store"),
            dcc.Store(id="prediction-data-store"),
            
            # Header
            create_header(),
            
            # Main Content
            html.Div(
                [
                    # Controls
                    create_controls(),
                    
                    # Stats Cards Row
                    create_stats_cards(),
                    
                    # Tabs
                    dcc.Tabs(
                        id="dashboard-tabs",
                        value="overview",
                        children=[
                            dcc.Tab(
                                label="📈 Overview",
                                value="overview",
                                style=TAB_STYLE,
                                selected_style=TAB_SELECTED_STYLE,
                                children=create_overview_tab(),
                            ),
                            dcc.Tab(
                                label="📊 Technical Analysis",
                                value="technical",
                                style=TAB_STYLE,
                                selected_style=TAB_SELECTED_STYLE,
                                children=create_technical_tab(),
                            ),
                            dcc.Tab(
                                label="🔮 ML Predictions",
                                value="predictions",
                                style=TAB_STYLE,
                                selected_style=TAB_SELECTED_STYLE,
                                children=create_predictions_tab(),
                            ),
                        ],
                        style={"marginBottom": "16px"},
                    ),
                ],
                style={"padding": "16px 24px"},
            ),
        ],
        style={
            "backgroundColor": COLORS["background"],
            "minHeight": "100vh",
            "fontFamily": "'Inter', 'Segoe UI', sans-serif",
        },
    )