"""
Logging Module
Configured with loguru for structured logging
"""

import sys
import os
from loguru import logger
from config import settings

# Remove default handler
logger.remove()

# Ensure logs directory exists
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Add console logger (colored output for development)
logger.add(
    sys.stderr,
    level=settings.LOG_LEVEL,
    format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level:<8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True
)

# Add file logger (structured JSON for production)
logger.add(
    os.path.join(LOG_DIR, "app.log"),
    level=settings.LOG_LEVEL,
    rotation="10 MB",      # Create new file every 10 MB
    retention="7 days",    # Keep logs for 7 days
    serialize=True,        # Output as JSON
    enqueue=True          # Make it async-safe
)

logger.info("Logger initialized.")