"""Logging config. Every line is prefixed with its source, e.g. [scraping], so output
from a background job is distinguishable from request handling."""

import logging
import sys

LOGGERS = ('scraping', 'server')


def setup(level):
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('[%(name)s] %(message)s'))

    for name in LOGGERS:
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
