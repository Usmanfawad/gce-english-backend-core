"""Logging configuration for the GCE English backend."""

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import Request
from loguru import logger


class LoguruConfig:
    """Loguru configuration class for the application."""

    def __init__(self, app_name: str = "gce-english-be"):
        self.app_name = app_name
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)
        
    def setup_logger(self, log_level: str = "INFO") -> None:
        """Configure Loguru logger for the application."""
        
        # Remove default handler
        logger.remove()
        
        # Console handler with colors and formatting
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level=log_level,
            colorize=True,
            backtrace=True,
            diagnose=True
        )
        
        # General application logs
        logger.add(
            self.logs_dir / "app.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="DEBUG",
            rotation="10 MB",
            retention="7 days",
            compression="zip",
            encoding="utf-8",
            backtrace=True,
            diagnose=True
        )
        
        # Error logs only
        logger.add(
            self.logs_dir / "errors.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="ERROR",
            rotation="5 MB",
            retention="30 days",
            compression="zip",
            encoding="utf-8",
            backtrace=True,
            diagnose=True
        )
        
        # Request logs
        logger.add(
            self.logs_dir / "requests.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
            level="INFO",
            rotation="20 MB",
            retention="14 days",
            compression="zip",
            encoding="utf-8",
            filter=lambda record: "REQUEST" in record["message"]
        )
        
        # Performance logs
        logger.add(
            self.logs_dir / "performance.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
            level="INFO",
            rotation="10 MB",
            retention="7 days",
            compression="zip",
            encoding="utf-8",
            filter=lambda record: "PERFORMANCE" in record["message"]
        )


def log_request_start(request: Request) -> None:
    """Log the start of a request using Loguru."""
    logger.info(
        "REQUEST START: {method} {path}",
        method=request.method,
        path=request.url.path,
        extra={
            "query_params": str(request.query_params),
            "client_ip": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent"),
            "timestamp": datetime.now().isoformat(),
        }
    )


def log_request_end(request: Request, status_code: int, process_time: float) -> None:
    """Log the completion of a request using Loguru."""
    logger.info(
        "REQUEST END: {method} {path} - {status_code} ({process_time:.4f}s)",
        method=request.method,
        path=request.url.path,
        status_code=status_code,
        process_time=round(process_time, 4),
        extra={
            "client_ip": request.client.host if request.client else None,
            "timestamp": datetime.now().isoformat(),
        }
    )


def log_request_error(request: Request, error: Exception, process_time: float) -> None:
    """Log a request error using Loguru."""
    logger.error(
        "REQUEST ERROR: {method} {path} - {error} ({process_time:.4f}s)",
        method=request.method,
        path=request.url.path,
        error=str(error),
        process_time=round(process_time, 4),
        extra={
            "error_type": type(error).__name__,
            "client_ip": request.client.host if request.client else None,
            "timestamp": datetime.now().isoformat(),
        }
    )


def log_performance(operation: str, duration: float, **kwargs) -> None:
    """Log performance metrics using Loguru."""
    logger.info(
        "PERFORMANCE: {operation} completed in {duration:.4f}s",
        operation=operation,
        duration=duration,
        **kwargs
    )


# Initialize Loguru configuration
loguru_config = LoguruConfig()
loguru_config.setup_logger()

# Export logger for use in other modules
app_logger = logger
