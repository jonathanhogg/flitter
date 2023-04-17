"""
Common Flitter initialisation
"""

import sys

from loguru import logger

try:
    import pyximport
    pyximport.install()
except ImportError:
    pass


LOGGING_LEVEL = "SUCCESS"
LOGGING_FORMAT = "{time:HH:mm:ss.SSS} {process}:{extra[shortname]:16s} | <level>{level}: {message}</level>"


def configure_logger(level=None):
    global LOGGING_LEVEL
    if level is None:
        level = LOGGING_LEVEL
    else:
        LOGGING_LEVEL = level
    logger.configure(handlers=[dict(sink=sys.stderr, format=LOGGING_FORMAT, level=level, enqueue=True)],
                     patcher=lambda record: record['extra'].update(shortname=record['name'].removeprefix('flitter')))
    return logger


def name_patch(logger, name):
    return logger.patch(lambda record, name=name: (record.update(name=name),
                                                   record['extra'].update(shortname=name.removeprefix('flitter'))))
