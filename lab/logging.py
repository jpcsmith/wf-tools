"""Configure logging for experiments."""
import logging
from typing import Union


def add_console(logger: Union[logging.Logger, str], level: int = logging.DEBUG):
    """Configures the logging to the console."""
    if isinstance(logger, str):
        logger = logging.getLogger(logger)

    logger.setLevel(level)

    console = logging.StreamHandler()
    console.setLevel(level)

    formatter = logging.Formatter(
        '[%(asctime)s] %(name)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logger.addHandler(console)
