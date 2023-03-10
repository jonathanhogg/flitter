"""
Common Flitter initialisation
"""

import sys

from loguru import logger
import pyximport
pyximport.install()


LOGGING_LEVEL = "SUCCESS"
LOGGING_FORMAT = "{time:HH:mm:ss.SSS} {process}:{extra[shortname]:14s} | <level>{level}: {message}</level>"


def configure_logger(level=None):
    global LOGGING_LEVEL
    if level is None:
        level = LOGGING_LEVEL
    else:
        LOGGING_LEVEL = level
    logger.configure(handlers=[dict(sink=sys.stderr, format=LOGGING_FORMAT, level=level, enqueue=True)],
                     patcher=lambda record: record['extra'].update(shortname=record['name'].removeprefix('flitter.')))
    return logger
