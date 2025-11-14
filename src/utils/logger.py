"""
Centralized logging configuration for the churn prediction project.
"""
import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(
    name: str,
    log_file: str = None,
    level: int = logging.INFO,
    format_string: str = None
) -> logging.Logger:
    """
    Create and configure a logger.

    Args:
        name: Logger name (typically __name__)
        log_file: Optional path to log file
        level: Logging level (default: INFO)
        format_string: Custom format string

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # Default format
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    formatter = logging.Formatter(format_string, datefmt='%Y-%m-%d %H:%M:%S')

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if log_file provided)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_project_logger(module_name: str) -> logging.Logger:
    """
    Get a logger for a specific module with project-wide settings.

    Args:
        module_name: Name of the module (__name__)

    Returns:
        Configured logger
    """
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d")
    log_file = logs_dir / f"churn_pipeline_{timestamp}.log"

    return setup_logger(
        name=module_name,
        log_file=str(log_file),
        level=logging.INFO
    )


class PipelineLogger:
    """
    Context manager for logging pipeline steps.
    """
    def __init__(self, logger: logging.Logger, step_name: str):
        self.logger = logger
        self.step_name = step_name
        self.start_time = None

    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info("=" * 60)
        self.logger.info(f"Starting: {self.step_name}")
        self.logger.info("=" * 60)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()

        if exc_type is None:
            self.logger.info(f"Completed: {self.step_name} ({duration:.2f}s)")
        else:
            self.logger.error(f"Failed: {self.step_name} ({duration:.2f}s)")
            self.logger.error(f"Error: {exc_val}")

        self.logger.info("=" * 60 + "\n")
        return False  # Don't suppress exceptions


# Example usage functions
def log_dataframe_info(logger: logging.Logger, df, name: str = "DataFrame"):
    """Log useful information about a DataFrame."""
    logger.info(f"{name} shape: {df.shape}")
    logger.info(f"{name} columns: {list(df.columns)}")
    logger.info(f"{name} memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    if df.isnull().any().any():
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        logger.warning(f"Missing values:\n{missing}")


def log_model_metrics(logger: logging.Logger, model_name: str, metrics: dict):
    """Log model performance metrics in a formatted way."""
    logger.info(f"Model: {model_name}")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")