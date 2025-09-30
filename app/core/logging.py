"""
Production logging configuration for CreativeIQ
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
import structlog
import os

from app.core.config import settings


def setup_logging(log_level: str = "INFO", log_dir: Optional[str] = None) -> None:
    """
    Configure structured logging for production
    """
    # Create log directory
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
    else:
        log_dir = "logs"
        Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper())
    )

    # Get root logger
    root_logger = logging.getLogger()

    # File handlers
    if log_dir:
        # Application log
        app_handler = logging.handlers.TimedRotatingFileHandler(
            filename=f"{log_dir}/app.log",
            when="midnight",
            interval=1,
            backupCount=7,
            encoding="utf-8"
        )
        app_handler.setLevel(logging.INFO)

        # Error log
        error_handler = logging.handlers.TimedRotatingFileHandler(
            filename=f"{log_dir}/error.log",
            when="midnight",
            interval=1,
            backupCount=30,
            encoding="utf-8"
        )
        error_handler.setLevel(logging.ERROR)

        # Performance log
        perf_handler = logging.handlers.TimedRotatingFileHandler(
            filename=f"{log_dir}/performance.log",
            when="midnight",
            interval=1,
            backupCount=7,
            encoding="utf-8"
        )
        perf_handler.setLevel(logging.INFO)

        # Add handlers
        root_logger.addHandler(app_handler)
        root_logger.addHandler(error_handler)

        # Configure specific loggers
        logging.getLogger("app.performance").addHandler(perf_handler)

    # Configure third-party loggers
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger
    """
    return structlog.get_logger(name)


class LoggingMiddleware:
    """
    Middleware to log requests and responses
    """

    def __init__(self, app):
        self.app = app
        self.logger = get_logger("api.requests")

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            start_time = time.time()

            # Log request
            self.logger.info(
                "request_started",
                method=scope["method"],
                path=scope["path"],
                client=scope.get("client", ["unknown", 0])[0],
                user_agent=dict(scope.get("headers", {})).get(b"user-agent", b"").decode()
            )

            # Process request
            async def send_wrapper(message):
                if message["type"] == "http.response.start":
                    duration = time.time() - start_time
                    self.logger.info(
                        "request_completed",
                        method=scope["method"],
                        path=scope["path"],
                        status_code=message["status"],
                        duration_ms=round(duration * 1000, 2)
                    )
                await send(message)

            await self.app(scope, receive, send_wrapper)
        else:
            await self.app(scope, receive, send)


# Performance monitoring
import time
import functools

def log_performance(operation: str):
    """
    Decorator to log operation performance
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            logger = get_logger("app.performance")
            start_time = time.time()

            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time

                logger.info(
                    "operation_completed",
                    operation=operation,
                    duration_ms=round(duration * 1000, 2),
                    status="success"
                )

                return result
            except Exception as e:
                duration = time.time() - start_time

                logger.error(
                    "operation_failed",
                    operation=operation,
                    duration_ms=round(duration * 1000, 2),
                    error=str(e),
                    status="error"
                )
                raise

        return wrapper
    return decorator


# Error tracking with Sentry (if configured)
if hasattr(settings, 'SENTRY_DSN') and settings.SENTRY_DSN:
    import sentry_sdk
    from sentry_sdk.integrations.fastapi import FastApiIntegration
    from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration

    sentry_sdk.init(
        dsn=settings.SENTRY_DSN,
        integrations=[
            FastApiIntegration(auto_enabling_integrations=False),
            SqlalchemyIntegration(),
        ],
        traces_sample_rate=0.1,
        environment=getattr(settings, 'ENV', 'production')
    )