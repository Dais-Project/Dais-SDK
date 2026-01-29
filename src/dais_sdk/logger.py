import sys
import logging

logger = logging.getLogger("LiteAI-SDK")
logger.addHandler(logging.NullHandler())

def enable_logging(level=logging.INFO):
    """
    Enable logging for the LiteAI SDK.

    Args:
        level: The logging level (default: logging.INFO).

               Common values: logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR
    """
    logger.setLevel(level)

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
