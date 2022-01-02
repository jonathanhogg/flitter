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
parser.add_argument('--profile', action='store_true', default=False, help="Run with profiling")
parser.add_argument('script', nargs='+', help="Script to execute")
args = parser.parse_args()
logging.basicConfig(level=logging.DEBUG if args.debug else (logging.INFO if args.verbose else logging.WARNING), stream=sys.stderr)

controller = Controller('.')
for script in args.script:
    controller.load_page(script)
controller.switch_to_page(0)
if args.profile:
    import cProfile
    cProfile.run('asyncio.run(controller.run())', sort='tottime')
else:
    asyncio.run(controller.run())
