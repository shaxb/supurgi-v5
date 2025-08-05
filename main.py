"""
Main entry point for trading system
"""

# Configure logging first
import logging_config  # This will configure loguru format

from orchestrator import run_orchestrator


if __name__ == "__main__":
    print("=== Trading System Starting ===")
    run_orchestrator()
    print("=== Trading System Stopped ===")
