"""
Common Flitter initialisation
"""

import logging
import sys

from loguru import logger

try:
    import pyximport
    pyximport.install(language_level=3)
except ImportError:
    pass

try:
    from setproctitle import setproctitle
except ImportError:
    def setproctitle(title):
        pass


__version__ = "1.0.0b28"

LOGGING_LEVEL = "SUCCESS"
LOGGING_FORMAT = "{time:HH:mm:ss.SSS} {extra[shortname]:25} | <level>{level}: {message}</level>"


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


def shorten_name(pkg_name, n=25):
    if len(pkg_name) <= n:
        return pkg_name
    over = len(pkg_name) - n
    parts = pkg_name.split('.')
    for i in range(len(parts)-1):
        part = parts[i]
        if len(part) > 2:
            over -= len(part) - 2
            part = part[0] + '-'
            parts[i] = part
            if over <= 0:
                break
    return '.'.join(parts)


def configure_logger(level=None):
    global LOGGING_LEVEL
    if level is None:
        level = LOGGING_LEVEL
    else:
        LOGGING_LEVEL = level
    logger.configure(handlers=[dict(sink=sys.stderr, format=LOGGING_FORMAT, level=level)],
                     patcher=lambda record: record['extra'].update(shortname=shorten_name(record['name'])))
    LoguruInterceptHandler.install()
    return logger


def name_patch(logger, name):
    return logger.patch(lambda record, name=name: (record.update(name=name),
                                                   record['extra'].update(shortname=shorten_name(name))))
