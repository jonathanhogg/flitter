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
parser.add_argument('--throttle', type=int, default=60, help="Framerate throttle")
parser.add_argument('--screen', type=int, default=0, help="Default screen number")
parser.add_argument('--fullscreen', action='store_true', default=False, help="Default to full screen")
parser.add_argument('--state', type=str, help="State save/restore file")
parser.add_argument('script', nargs='+', help="Script to execute")
args = parser.parse_args()
logging.basicConfig(level=logging.DEBUG if args.debug else (logging.INFO if args.verbose else logging.WARNING), stream=sys.stderr)

controller = Controller('.', max_fps=args.throttle, screen=args.screen, fullscreen=args.fullscreen, state_file=args.state)
for script in args.script:
    controller.load_page(script)
controller.switch_to_page(0)
if args.profile:
    import cProfile
    cProfile.run('asyncio.run(controller.run())', sort='tottime')
else:
    asyncio.run(controller.run())
