"""
Common Flitter initialisation
"""

import pyximport
pyximport.install()


LOGGING_LEVEL = "WARNING"
LOGGING_FORMAT = "{time:HH:mm:ss.SSS} | {process}:{name} | <level>{level}: {message}</level>"
