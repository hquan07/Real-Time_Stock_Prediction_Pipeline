import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from loguru import logger
from utils.helper import load_env


def run_data_ingestion(args):
    """Run data ingestion pipeline."""
    from data_ingestion.main import run_pipeline
    logger.info("🚀 Starting data ingestion pipeline...")
    run_pipeline()


def run_dashboard(args):
    """Run the Dash dashboard."""
    from dashboard.app import run_dashboard as start_dashboard
    logger.info(f"🚀 Starting dashboard on port {args.port}...")
    start_dashboard(debug=args.debug, port=args.port)


def run_train(args):
    """Run model training."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from scripts.run_train import run_train as train
    logger.info(f"🚀 Starting {args.model} training...")
    result = train(model_type=args.model, lookback_days=args.days)
    logger.info(f"Training result: {result}")


def run_inference(args):
    """Run batch inference."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from scripts.run_inference import run_inference as inference
    logger.info("🚀 Starting inference pipeline...")
    result = inference(model_type=args.model)
    logger.info(f"Inference result: {result}")


def run_spark_consumer(args):
    """Run Spark streaming consumer."""
    from streaming.jobs.spark_consumer_job import run_consumer
    logger.info("🚀 Starting Spark consumer...")
    run_consumer()


def init_database(args):
    """Initialize database tables."""
    from database.db_connection import init_database as init_db, check_connection
    
    logger.info("🔍 Checking database connection...")
    if check_connection():
        logger.info("✅ Connection OK, initializing tables...")
        init_db()
    else:
        logger.error("❌ Database connection failed")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    # Load environment variables
    load_env()
    
    parser = argparse.ArgumentParser(
        description="Real-Time Stock Prediction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py ingest          # Run data ingestion
  python main.py dashboard       # Start web dashboard
  python main.py train           # Train ML model
  python main.py inference       # Run predictions
  python main.py init-db         # Initialize database
        """,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Data ingestion
    ingest_parser = subparsers.add_parser("ingest", help="Run data ingestion")
    ingest_parser.set_defaults(func=run_data_ingestion)
    
    # Dashboard
    dashboard_parser = subparsers.add_parser("dashboard", help="Start dashboard")
    dashboard_parser.add_argument("--port", type=int, default=8050, help="Port number")
    dashboard_parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    dashboard_parser.set_defaults(func=run_dashboard)
    
    # Training
    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument(
        "--model",
        choices=["random_forest", "lstm"],
        default="random_forest",
        help="Model type",
    )
    train_parser.add_argument("--days", type=int, default=365, help="Days of data")
    train_parser.set_defaults(func=run_train)
    
    # Inference
    inference_parser = subparsers.add_parser("inference", help="Run inference")
    inference_parser.add_argument(
        "--model",
        choices=["random_forest", "lstm"],
        default="random_forest",
        help="Model type",
    )
    inference_parser.set_defaults(func=run_inference)
    
    # Spark consumer
    spark_parser = subparsers.add_parser("spark", help="Run Spark consumer")
    spark_parser.set_defaults(func=run_spark_consumer)
    
    # Database init
    db_parser = subparsers.add_parser("init-db", help="Initialize database")
    db_parser.set_defaults(func=init_database)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    try:
        args.func(args)
    except KeyboardInterrupt:
        logger.warning("🛑 Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"🔥 Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
