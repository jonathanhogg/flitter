"""
Multi-processing for window rendering
"""

# pylama:ignore=R0903,R1732,R0913

import argparse
import logging
from multiprocessing.shared_memory import SharedMemory
import pickle
import struct
import subprocess
import sys
import time

from posix_ipc import Semaphore, O_CREX

from . import scene


HEADER_SIZE = 4
BUFFER_SIZE = 1 << 20

Log = logging.getLogger(__package__)


class Window:
    def __init__(self, screen=0, fullscreen=False, vsync=False):
        self.shared_memory = SharedMemory(create=True, size=BUFFER_SIZE)
        self.ready = Semaphore(None, O_CREX)
        self.done = Semaphore(None, O_CREX)
        self.position = HEADER_SIZE
        arguments = [sys.executable, '-m', 'flitter.render.process', f'--screen={screen}']
        if fullscreen:
            arguments.append('--fullscreen')
        if vsync:
            arguments.append('--vsync')
        level = logging.getLogger().level
        if level == logging.INFO:
            arguments.append('--verbose')
        elif level == logging.DEBUG:
            arguments.append('--debug')
        arguments.extend([self.shared_memory.name, self.ready.name, self.done.name])
        self.process = subprocess.Popen(arguments)

    def write(self, data):
        end = self.position + len(data)
        self.shared_memory.buf[self.position:end] = data
        self.position = end

    def update(self, node, **kwargs):
        pickle.dump((node, kwargs), self, protocol=pickle.HIGHEST_PROTOCOL)
        self.shared_memory.buf[0:4] = struct.pack('>L', self.position)
        self.ready.release()
        self.done.acquire()
        self.position = HEADER_SIZE

    def purge(self):
        pass

    def destroy(self):
        self.shared_memory.close()
        self.ready.close()
        self.done.close()
        self.process.kill()
        self.process.wait()
        self.shared_memory.unlink()
        self.ready.unlink()
        self.done.unlink()


class Server:
    def __init__(self, buffer, ready, done, screen=0, fullscreen=False, vsync=False):
        Log.info("Starting render node %s", buffer)
        self.window = scene.Window(screen=screen, fullscreen=fullscreen, vsync=vsync)
        self.shared_memory = SharedMemory(name=buffer)
        self.ready = Semaphore(ready)
        self.done = Semaphore(done)

    def run(self):
        try:
            decode = 0
            draw = 0
            size = 0
            nframes = 0
            stats_time = time.monotonic()
            while True:
                self.ready.acquire()
                decode -= time.monotonic()
                end, = struct.unpack_from('>L', self.shared_memory.buf)
                node, kwargs = pickle.loads(self.shared_memory.buf[HEADER_SIZE:end])
                decode += time.monotonic()
                self.done.release()
                draw -= time.monotonic()
                self.window.update(node, **kwargs)
                draw += time.monotonic()
                nframes += 1
                size += end - HEADER_SIZE
                if time.monotonic() > stats_time + 5:
                    Log.info("Render stats - decode %.1fms, draw %.1fms, data size %.0f bytes", 1000*decode/nframes, 1000*draw/nframes, 1000*size/nframes)
                    nframes = decode = draw = size = 0
                    stats_time += 5
        finally:
            self.shared_memory.close()
            self.ready.close()
            self.done.close()
            Log.info("Stopped render node %s", self.shared_memory.name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Flitter renderer")
    parser.add_argument('--debug', action='store_true', default=False, help="Debug logging")
    parser.add_argument('--verbose', action='store_true', default=False, help="Informational logging")
    parser.add_argument('--screen', type=int, default=0, help="Default screen number")
    parser.add_argument('--fullscreen', action='store_true', default=False, help="Default to full screen")
    parser.add_argument('--vsync', action='store_true', default=False, help="Default to winow vsync")
    parser.add_argument('objects', type=str, nargs='+', help="Shared objects")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else (logging.INFO if args.verbose else logging.WARNING), stream=sys.stderr)
    server = Server(*args.objects, screen=args.screen, fullscreen=args.fullscreen, vsync=args.vsync)
    server.run()
