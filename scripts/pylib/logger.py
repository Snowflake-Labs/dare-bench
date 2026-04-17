# https://betterstack.com/community/guides/logging/loguru/#creating-a-request-logging-middleware

import asyncio
import functools
import json
import sys
import time

import loguru


logger = loguru.logger
logger.remove(0)
logger.add(
    sys.stderr,
    # adds color to each log line for that level
    format="<level>{extra[serialized]}</level>",
    # sets the minimum level for visible logs
    level="TRACE",
)


def get_logger():
    """Returns the loguru logger"""
    return logger


def log_function(func):
    """Decorate a function with this to provide timing information, useful for tracing"""

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        logger.trace(f"Starting execution of {func.__name__}")
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.trace(
            f"Finished execution of {func.__name__}. Took {end_time - start_time:.10f} seconds."
        )
        return result

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        logger.trace(f"Starting execution of {func.__name__}")
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        logger.trace(
            f"Finished execution of {func.__name__}. Took {end_time - start_time:.10f} seconds."
        )
        return result

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper