"""
Main entry point for trading system
"""

from orchestrator import run_orchestrator


if __name__ == "__main__":
    print("=== Trading System Starting ===")
    run_orchestrator()
    print("=== Trading System Stopped ===")
