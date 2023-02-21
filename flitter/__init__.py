"""
Common Flitter initialisation
"""

import logging
logging.TRACE = 9
logging.addLevelName(logging.TRACE, "TRACE")
def trace(self, message, *args, **kwargs):
    if self.isEnabledFor(logging.TRACE):
        self._log(logging.TRACE, message, args, **kwargs)
logging.Logger.trace = trace
del trace

import pyximport
pyximport.install()
