"""
Logging configuration for TrendStack
"""

from loguru import logger
import sys


def configure_logging():
    """Configure loguru with custom format."""
    # Remove default handler
    logger.remove()
    
    # Add custom handler with your desired format
    # Format: LEVEL -> file:function:line -> message (all colored by level)
    # Colors: DEBUG=white, INFO=green, WARNING=yellow, ERROR=red, CRITICAL=magenta
    logger.add(
        sys.stderr,
        format="<level>{level}</level> -> <cyan>{file.name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> -> <level>{message}</level>",
        level="DEBUG",
        colorize=True
    )


# Auto-configure when imported
configure_logging()
