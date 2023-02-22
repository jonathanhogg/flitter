"""
Multi-processing for rendering
"""

# pylama:ignore=R0903,R1732,R0913

import asyncio
from multiprocessing import Process, Queue
import os
import sys
import time

from loguru import logger

import flitter


class Proxy:
    def __init__(self, cls, **kwargs):
        self.queue = Queue(1)
        self.process = Process(target=Proxy.run, args=(cls, kwargs, self.queue, flitter.LOGGING_LEVEL))
        self.process.start()

    async def update(self, *args, **kwargs):
        await asyncio.to_thread(self.queue.put, ('update', args, kwargs))

    def purge(self):
        self.queue.put(('purge', (), {}))

    def destroy(self):
        self.queue.put(('purge', (), {}))
        self.queue.put(('destroy', (), {}))
        self.queue.close()
        self.queue.join_thread()
        self.process.join()
        self.process.close()
        self.queue = self.process = None

    @staticmethod
    def run(cls, kwargs, queue, log_level):
        logger.configure(handlers=[{'sink': sys.stderr, 'format': flitter.LOGGING_FORMAT, 'level': log_level, 'enqueue': True}])
        logger.info("Started {} render process", cls.__name__)
        try:
            asyncio.run(Proxy.loop(queue, cls(**kwargs)))
        except Exception:
            logger.exception("Unhandled exception in {} render process", cls.__name__)
        finally:
            logger.info("Stopped {} render process", cls.__name__)
            logger.complete()

    @staticmethod
    async def loop(queue, obj):
        nframes = render = 0
        stats_time = time.perf_counter()
        while True:
            method, args, kwargs = queue.get()
            if method == 'update':
                render -= time.perf_counter()
                await obj.update(*args, **kwargs)
                render += time.perf_counter()
            elif method == 'purge':
                obj.purge()
            elif method == 'destroy':
                obj.destroy()
                return
            nframes += 1
            if time.perf_counter() > stats_time + 5:
                logger.info("{} render {:.1f}ms", obj.__class__.__name__, 1000*render/nframes)
                nframes = render = 0
                stats_time += 5
