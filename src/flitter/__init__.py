"""
Common Flitter initialisation
"""

import logging
import sys

from loguru import logger

try:
    import pyximport
    pyximport.install()
except ImportError:
    pass


__version__ = "1.0.0b10"

LOGGING_LEVEL = "SUCCESS"
LOGGING_FORMAT = "{time:HH:mm:ss.SSS} {process}:{extra[shortname]:16s} | <level>{level}: {message}</level>"


class LoguruInterceptHandler(logging.Handler):
    @classmethod
    def install(cls):
        handler = cls()
        logging.basicConfig(handlers=[handler], level=0)
        return handler

    def uninstall(self):
        logging.getLogger().removeHandler(self)

    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = logging.currentframe().f_back, 1
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def configure_logger(level=None):
    global LOGGING_LEVEL
    if level is None:
        level = LOGGING_LEVEL
    else:
        LOGGING_LEVEL = level
    logger.configure(handlers=[dict(sink=sys.stderr, format=LOGGING_FORMAT, level=level, enqueue=True)],
                     patcher=lambda record: record['extra'].update(shortname=record['name'].removeprefix('flitter')))
    LoguruInterceptHandler.install()
    return logger


def name_patch(logger, name):
    return logger.patch(lambda record, name=name: (record.update(name=name),
                                                   record['extra'].update(shortname=name.removeprefix('flitter'))))
