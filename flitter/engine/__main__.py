"""
Flitter main entry point
"""

import argparse
import asyncio
import logging
import subprocess
import sys

from .control import Controller


parser = argparse.ArgumentParser(description="Flitter")
parser.add_argument('--debug', action='store_true', default=False, help="Debug logging")
parser.add_argument('--verbose', action='store_true', default=False, help="Informational logging")
parser.add_argument('--profile', action='store_true', default=False, help="Run with profiling")
parser.add_argument('--fps', type=int, default=60, help="Target framerate")
parser.add_argument('--screen', type=int, default=0, help="Default screen number")
parser.add_argument('--fullscreen', action='store_true', default=False, help="Default to full screen")
parser.add_argument('--vsync', action='store_true', default=False, help="Default to winow vsync")
parser.add_argument('--state', type=str, help="State save/restore file")
parser.add_argument('--multiprocess', action='store_true', default=False, help="Use multiprocessing")
parser.add_argument('--autoreset', type=float, help="Auto-reset state on idle")
parser.add_argument('--push', action='store_true', default=False, help="Start Ableton Push 2 interface")
parser.add_argument('script', nargs='+', help="Script to execute")
args = parser.parse_args()
logging.basicConfig(level=logging.DEBUG if args.debug else (logging.INFO if args.verbose else logging.WARNING), stream=sys.stderr)

controller = Controller('.', target_fps=args.fps, screen=args.screen, fullscreen=args.fullscreen, vsync=args.vsync,
                        state_file=args.state, multiprocess=args.multiprocess and not args.profile, autoreset=args.autoreset)
for script in args.script:
    controller.load_page(script)
controller.switch_to_page(0)

if args.push:
    arguments = [sys.executable, '-u', '-m', 'flitter.interface.push']
    level = logging.getLogger().level
    if args.debug:
        arguments.append('--debug')
    elif args.verbose:
        arguments.append('--verbose')
    push = subprocess.Popen(arguments)
else:
    push = None

try:
    if args.profile:
        import cProfile
        cProfile.run('asyncio.run(controller.run())', sort='tottime')
    else:
        asyncio.run(controller.run())
finally:
    if push is not None:
        push.kill()
        push.wait()
