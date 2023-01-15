"""
Multi-processing for window rendering
"""

# pylama:ignore=R0903,R1732,R0913

import argparse
import asyncio
import importlib
import logging
from multiprocessing.shared_memory import SharedMemory
import os
import pickle
import struct
import subprocess
import sys
import time

from posix_ipc import Semaphore, O_CREX

from .. import model


HEADER_SIZE = 4
BUFFER_SIZE = 1 << 20

Log = logging.getLogger('flitter.render.process')


class Proxy:
    def __init__(self, class_name, **kwargs):
        self.shared_memory = SharedMemory(create=True, size=BUFFER_SIZE)
        self.ready = Semaphore(None, O_CREX)
        self.done = Semaphore(None, O_CREX)
        self.position = HEADER_SIZE
        arguments = [sys.executable, '-u', '-m', 'flitter.render.process']
        level = logging.getLogger().level
        if level == logging.INFO:
            arguments.append('--verbose')
        elif level == logging.DEBUG:
            arguments.append('--debug')
        arguments.extend([self.shared_memory.name, self.ready.name, self.done.name, class_name])
        pickle.dump(kwargs, self, protocol=pickle.HIGHEST_PROTOCOL)
        self.shared_memory.buf[0:4] = struct.pack('>L', self.position)
        self.process = subprocess.Popen(arguments)
        self.position = HEADER_SIZE

    def write(self, data):
        end = self.position + len(data)
        self.shared_memory.buf[self.position:end] = data
        self.position = end

    async def update(self, node, **kwargs):
        await asyncio.to_thread(self.done.acquire)
        pickle.dump((node, kwargs), self, protocol=pickle.HIGHEST_PROTOCOL)
        self.shared_memory.buf[0:4] = struct.pack('>L', self.position)
        self.ready.release()
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
    def __init__(self, buffer, ready, done, class_name):
        self.pid = os.getpid()
        self.shared_memory = SharedMemory(name=buffer)
        self.ready = Semaphore(ready)
        self.done = Semaphore(done)
        self.class_name = class_name
        parts = class_name.split('.')
        module = importlib.import_module('.'.join(parts[:-1]))
        cls = getattr(module, parts[-1])
        end, = struct.unpack_from('>L', self.shared_memory.buf)
        kwargs = pickle.loads(self.shared_memory.buf[HEADER_SIZE:end])
        self.obj = cls(**kwargs)

    async def run(self):
        Log.info("Started %s render process %d", self.class_name, self.pid)
        try:
            decode = 0
            draw = 0
            size = 0
            nframes = 0
            stats_time = time.perf_counter()
            self.done.release()
            while True:
                self.ready.acquire()
                decode -= time.perf_counter()
                end, = struct.unpack_from('>L', self.shared_memory.buf)
                node, kwargs = pickle.loads(self.shared_memory.buf[HEADER_SIZE:end])
                decode += time.perf_counter()
                self.done.release()
                draw -= time.perf_counter()
                await self.obj.update(node, **kwargs)
                draw += time.perf_counter()
                nframes += 1
                size += end - HEADER_SIZE
                if time.perf_counter() > stats_time + 5:
                    Log.info("Process %d /frame - decode %.1fms, render %.1fms, data size %d bytes", self.pid,
                             1000*decode/nframes, 1000*draw/nframes, size//nframes)
                    nframes = decode = draw = size = 0
                    stats_time += 5
        finally:
            self.shared_memory.close()
            self.ready.close()
            self.done.close()
            Log.info("Stopped %s render process %d", self.class_name, self.pid)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Flitter renderer")
    parser.add_argument('--debug', action='store_true', default=False, help="Debug logging")
    parser.add_argument('--verbose', action='store_true', default=False, help="Informational logging")
    parser.add_argument('names', type=str, nargs='+', help="Object/class names")
    args = parser.parse_args()
    server = Server(*args.names)
    logging.basicConfig(level=logging.DEBUG if args.debug else (logging.INFO if args.verbose else logging.WARNING), stream=sys.stderr)
    asyncio.run(server.run())
