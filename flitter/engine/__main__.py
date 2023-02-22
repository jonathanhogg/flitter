"""
Flitter main entry point
"""

import argparse
import asyncio
import subprocess
import sys

from loguru import logger

import flitter
from .control import Controller


parser = argparse.ArgumentParser(description="Flitter")
parser.set_defaults(level=flitter.LOGGING_LEVEL)
levels = parser.add_mutually_exclusive_group()
levels.add_argument('--trace', action='store_const', const='TRACE', dest='level', help="Trace logging")
levels.add_argument('--debug', action='store_const', const='DEBUG', dest='level', help="Debug logging")
levels.add_argument('--verbose', action='store_const', const='INFO', dest='level', help="Informational logging")
parser.add_argument('--profile', action='store_true', default=False, help="Run with profiling")
parser.add_argument('--fps', type=int, default=60, help="Target framerate")
parser.add_argument('--screen', type=int, default=0, help="Default screen number")
parser.add_argument('--fullscreen', action='store_true', default=False, help="Default to full screen")
parser.add_argument('--vsync', action='store_true', default=False, help="Default to winow vsync")
parser.add_argument('--state', type=str, help="State save/restore file")
parser.add_argument('--multiprocess', action='store_true', default=False, help="Use multiprocessing")
parser.add_argument('--autoreset', type=float, help="Auto-reset state on idle")
parser.add_argument('--evalstate', type=float, default=0, help="Partially-evaluate on state after stable period")
parser.add_argument('--push', action='store_true', default=False, help="Start Ableton Push 2 interface")
parser.add_argument('script', nargs='+', help="Script to execute")
args = parser.parse_args()
logger.configure(handlers=[{'sink': sys.stderr, 'format': flitter.LOGGING_FORMAT, 'level': args.level, 'enqueue': True}])
flitter.LOGGING_LEVEL = args.level
controller = Controller('.', target_fps=args.fps, screen=args.screen, fullscreen=args.fullscreen, vsync=args.vsync,
                        state_file=args.state, multiprocess=args.multiprocess and not args.profile, autoreset=args.autoreset,
                        state_eval_wait=args.evalstate)
for script in args.script:
    controller.load_page(script)
controller.switch_to_page(0)

if args.push:
    arguments = [sys.executable, '-u', '-m', 'flitter.interface.push']
    if args.level == 'DEBUG':
        arguments.append('--debug')
    elif args.level == 'VERBOSE':
        arguments.append('--verbose')
    push = subprocess.Popen(arguments)
    logger.success("Started Push 2 interface sub-process")
else:
    push = None

try:
    if args.profile:
        import cProfile
        cProfile.run('asyncio.run(controller.run())', sort='tottime')
    else:
        asyncio.run(controller.run())
except KeyboardInterrupt:
    logger.info("Exiting on keyboard interrupt")
except Exception:
    logger.exception("Unexpected error in flitter")
finally:
    if push is not None:
        push.kill()
        push.wait()
    logger.complete()
