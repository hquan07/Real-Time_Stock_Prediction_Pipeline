import os
import sys
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
from loguru import logger

from dashboard.layouts import create_layout
from dashboard.callbacks import register_callbacks


# Configuration
DEBUG = os.getenv("DASHBOARD_DEBUG", "False").lower() == "true"
HOST = os.getenv("DASHBOARD_HOST", "0.0.0.0")
PORT = int(os.getenv("DASHBOARD_PORT", "8050"))


# Initialize Dash App
app = Dash(
    __name__,
    title="Stock Prediction Dashboard",
    update_title="Loading...",
    suppress_callback_exceptions=True,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ],
)

# Create layout
app.layout = create_layout()

# Register callbacks
register_callbacks(app)

# Server for deployment
server = app.server


# Main Entry Point
def run_dashboard(debug: bool = None, host: str = None, port: int = None):
    """
    Run the dashboard server.
    
    Args:
        debug: Enable debug mode
        host: Host address
        port: Port number
    """
    _debug = debug if debug is not None else DEBUG
    _host = host if host is not None else HOST
    _port = port if port is not None else PORT

    logger.info(f"🚀 Starting dashboard on http://{_host}:{_port}")
    app.run(debug=_debug, host=_host, port=_port)


if __name__ == "__main__":
    run_dashboard()
