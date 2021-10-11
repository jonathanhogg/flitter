"""
Flitter main entry point
"""

import argparse
import asyncio
import logging
import sys

from .control import Controller


parser = argparse.ArgumentParser(description="Flitter")
parser.add_argument('--debug', action='store_true', default=False, help="Debug logging")
parser.add_argument('--verbose', action='store_true', default=False, help="Informational logging")
parser.add_argument('script', help="Script to execute")
args = parser.parse_args()
logging.basicConfig(level=logging.DEBUG if args.debug else (logging.INFO if args.verbose else logging.WARNING), stream=sys.stderr)

controller = Controller('.')
controller.load(args.script)
asyncio.run(controller.run())
