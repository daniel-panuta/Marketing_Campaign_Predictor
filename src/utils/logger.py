"""
Logger configuration for Marketing Campaign Predictor project.

This module centralizes logging configuration using Loguru,
so that every Python module in the project can import the same logger.
Example usage:
    from src.utils.logger import logger
    logger.info("Starting preprocessing phase...")
"""

from loguru import logger
import os

from datetime import datetime

# --- Create logs directory if it doesn't exist ---
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# --- Define log file path ---
LOG_FILE = os.path.join(LOG_DIR, "app.log")

# --- Remove default handlers (avoid duplicate console logs) ---
logger.remove()

# --- Add console handler (for real-time visibility) ---
logger.add(
    sink=lambda msg: print(msg, end=""),
    level="INFO",
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
           "<level>{level: <8}</level> | "
           "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
           "<level>{message}</level>",
)

# --- Add file handler (persistent logs with rotation & retention) ---
logger.add(
    LOG_FILE,
    level="DEBUG",
    rotation="1 day",          # new file every day
    retention="7 days",        # keep logs for 7 days
    compression="zip",         # compress old logs
    encoding="utf-8",
    enqueue=True,              # thread-safe
    backtrace=True,            # include stack traces
    diagnose=True,             # detailed errors
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
)

# --- Example startup log ---
logger.info(f"Logger initialized at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} — writing to {LOG_FILE}")
