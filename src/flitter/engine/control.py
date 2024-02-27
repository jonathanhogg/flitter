"""
The main Flitter engine
"""

import asyncio
import gc
from pathlib import Path
import pickle

from loguru import logger

from ..cache import SharedCache
from ..clock import BeatCounter, system_clock
from ..language.vm import log_vm_stats
from ..model import Vector, StateDict, Context, null, numbers_cache_counts, empty_numbers_cache
from ..render import get_renderer


class EngineController:
    def __init__(self, target_fps=60, screen=0, fullscreen=False, vsync=False, state_file=None,
                 autoreset=None, state_simplify_wait=0, realtime=True, defined_names=None, vm_stats=False,
                 run_time=None, offscreen=False, window_gamma=1):
        self.default_fps = target_fps
        self.target_fps = target_fps
        self.realtime = realtime
        self.screen = screen
        self.fullscreen = fullscreen
        self.vsync = vsync
        self.offscreen = offscreen
        self.window_gamma = window_gamma
        self.autoreset = autoreset
        self.state_simplify_wait = state_simplify_wait / 2
        if defined_names:
            self.defined_names = {key: Vector.coerce(value) for key, value in defined_names.items()}
        else:
            self.defined_names = {}
        self.vm_stats = vm_stats
        self.run_time = run_time
        self.state_file = Path(state_file) if state_file is not None else None
        if self.state_file is not None and self.state_file.exists():
            logger.info("Recover state from state file: {}", self.state_file)
            with open(self.state_file, 'rb') as file:
                self.global_state = pickle.load(file)
        else:
            self.global_state = {}
        self.global_state_dirty = False
        self.state = None
        self.state_timestamp = None
        self.state_generation0 = None
        self.state_generation1 = None
        self.state_generation2 = None
        self.renderers = {}
        self.counter = BeatCounter()
        self.pages = []
        self.switch_page = None
        self.current_page = None
        self.current_path = None
        self._references = {}

    def load_page(self, filename):
        page_number = len(self.pages)
        path = SharedCache.get_with_root(filename, '.')
        self.pages.append((path, self.global_state.setdefault(str(path), StateDict())))
        logger.info("Added code page {}: {}", page_number, path)
        if page_number == 0:
            self.switch_to_page(0)

    def switch_to_page(self, page_number):
        if self.pages is not None and 0 <= page_number < len(self.pages):
            path, state = self.pages[page_number]
            self.state = state
            self.state_timestamp = system_clock()
            self.state_generation0 = set()
            self.state_generation1 = set()
            self.state_generation2 = set()
            self.current_path = path
            self.current_page = page_number
            SharedCache.set_root(self.current_path)
            logger.info("Switched to page {}: {}", page_number, self.current_path)
            if counter_state := self.state['_counter']:
                tempo, quantum, start = counter_state
                self.counter.reset(tempo, int(quantum), start)
                logger.info("Restore counter at beat {:.1f}, tempo {:.1f}, quantum {}", self.counter.beat, self.counter.tempo, self.counter.quantum)
            else:
                self.counter.reset()
            for renderers in self.renderers.values():
                for renderer in renderers:
                    renderer.purge()

    def has_next_page(self):
        return self.current_page < len(self.pages) - 1

    def next_page(self):
        if self.current_page < len(self.pages) - 1:
            self.switch_page = self.current_page + 1

    def has_previous_page(self):
        return self.current_page > 0

    def previous_page(self):
        if self.current_page > 0:
            self.switch_page = self.current_page - 1

    async def update_renderers(self, root, **kwargs):
        nodes_by_kind = {}
        for node in root.children:
            nodes_by_kind.setdefault(node.kind, []).append(node)
        tasks = []
        references = dict(self._references)
        for kind, nodes in nodes_by_kind.items():
            renderer_class = get_renderer(kind)
            if renderer_class is not None:
                renderers = self.renderers.setdefault(kind, [])
                count = 0
                for node in nodes:
                    if count == len(renderers):
                        renderer = renderer_class(**kwargs)
                        renderers.append(renderer)
                    tasks.append(asyncio.create_task(renderers[count].update(self, node, references=references, **kwargs)))
                    count += 1
                while len(renderers) > count:
                    renderers.pop().destroy()
        for kind in list(self.renderers):
            if kind not in nodes_by_kind:
                for renderer in self.renderers.pop(kind):
                    renderer.destroy()
        try:
            await asyncio.gather(*tasks)
        except Exception:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        return references

    def handle_pragmas(self, pragmas, timestamp):
        tempo = pragmas.get('tempo', null).match(1, float, 120)
        if tempo != self.counter.tempo:
            self.counter.set_tempo(tempo, timestamp)
        quantum = pragmas.get('quantum', null).match(1, float, 4)
        if quantum != self.counter.quantum:
            self.counter.set_quantum(quantum, timestamp)
        self.target_fps = pragmas.get('fps', null).match(1, float, self.default_fps)

    def reset_state(self):
        self.state.clear()
        self.state_timestamp = None
        self.state_generation0 = set()
        self.state_generation1 = set()
        self.state_generation2 = set()
        self.global_state_dirty = True

    async def run(self):
        try:
            frames = []
            start_time = frame_time = system_clock() if self.realtime else self.counter.start
            last = self.counter.beat_at_time(frame_time)
            dump_time = frame_time
            execution = render = housekeeping = 0
            slow_frame = False
            performance = 1
            run_program = current_program = errors = logs = None
            simplify_state_time = system_clock() + self.state_simplify_wait
            gc.disable()
            while self.run_time is None or int(round((frame_time - start_time) * self.target_fps)) < int(round(self.run_time * self.target_fps)):
                housekeeping -= system_clock()

                beat = self.counter.beat_at_time(frame_time)
                delta = beat - last
                last = beat
                names = {'beat': beat, 'quantum': self.counter.quantum, 'tempo': self.counter.tempo,
                         'delta': delta, 'clock': frame_time, 'performance': performance, 'slow_frame': slow_frame,
                         'fps': self.target_fps, 'realtime': self.realtime, 'window_gamma': self.window_gamma,
                         'screen': self.screen, 'fullscreen': self.fullscreen, 'vsync': self.vsync, 'offscreen': self.offscreen}

                program = self.current_path.read_flitter_program(static=self.defined_names, dynamic=names)
                if program is not current_program:
                    level = 'SUCCESS' if current_program is None else 'INFO'
                    logger.log(level, "Loaded page {}: {}", self.current_page, self.current_path)
                    run_program = current_program = program
                    errors = set()
                    logs = set()

                if self.state.changed:
                    if self.state_simplify_wait:
                        changed_keys = self.state.changed_keys - self.state_generation0
                        self.state_generation0.update(changed_keys)
                        if changed_keys:
                            generation1to0 = self.state_generation1 & changed_keys
                            changed_keys = changed_keys - generation1to0
                            self.state_generation1.difference_update(generation1to0)
                            generation2to0 = self.state_generation2 & changed_keys
                            if generation2to0:
                                if run_program is not current_program:
                                    run_program = current_program
                                    logger.debug("Undo simplification on state; original program with {} instructions", len(run_program))
                                self.state_generation1.update(self.state_generation2 - generation2to0)
                                self.state_generation2 = set()
                                simplify_state_time = system_clock() + self.state_simplify_wait
                    self.global_state_dirty = True
                    self.state_timestamp = system_clock()
                    self.state.clear_changed()

                if self.state_simplify_wait and system_clock() > simplify_state_time:
                    if current_program is not None and self.state_generation1:
                        simplify_time = -system_clock()
                        simplify_state = self.state.with_keys(self.state_generation2 ^ self.state_generation1)
                        generation2 = set(simplify_state)
                        if generation2 != self.state_generation2:
                            self.state_generation2 = generation2
                            top = current_program.top.simplify(state=simplify_state, dynamic=names)
                            now = system_clock()
                            simplify_time += now
                            compile_time = -now
                            run_program = top.compile(initial_lnames=tuple(names))
                            run_program.set_path(current_program.path)
                            run_program.set_top(top)
                            compile_time += system_clock()
                            logger.debug("Simplified on {} static state keys to {} instructions in -/{:.1f}/{:.1f}ms",
                                         len(self.state_generation2), len(run_program), simplify_time*1000, compile_time*1000)
                    self.state_generation1 = self.state_generation0
                    self.state_generation0 = set()
                    simplify_state_time = system_clock() + self.state_simplify_wait

                now = system_clock()
                housekeeping += now
                execution -= now
                if run_program is not None:
                    context = Context(names={key: Vector.coerce(value) for key, value in names.items()},
                                      state=self.state, references=self._references)
                    run_program.run(context, record_stats=self.vm_stats)
                else:
                    context = Context()
                self.handle_pragmas(context.pragmas, frame_time)

                new_errors = context.errors.difference(errors) if errors is not None else context.errors
                errors = context.errors
                for error in new_errors:
                    logger.error("Execution error: {}", error)
                new_logs = context.logs.difference(logs) if logs is not None else context.logs
                logs = context.logs
                for log in new_logs:
                    print(log)
                now = system_clock()
                execution += now
                render -= now

                self._references = await self.update_renderers(context.root, **names)

                now = system_clock()
                render += now
                housekeeping -= now

                del context
                SharedCache.clean()

                self.state['_counter'] = self.counter.tempo, self.counter.quantum, self.counter.start

                if self.autoreset and self.state_timestamp is not None and system_clock() > self.state_timestamp + self.autoreset:
                    logger.debug("Auto-reset state")
                    self.reset_state()
                    current_program = program

                if self.global_state_dirty and self.state_file is not None and frame_time > dump_time + 1:
                    logger.debug("Saving state")
                    with open(self.state_file, 'wb') as file:
                        pickle.dump(self.global_state, file)
                    self.global_state_dirty = False
                    dump_time = frame_time

                if self.switch_page is not None:
                    if self.autoreset:
                        self.reset_state()
                    self.switch_to_page(self.switch_page)
                    self.switch_page = None
                    run_program = current_program = None
                    performance = 1
                    count = gc.collect(2)
                    logger.trace("Collected {} objects (full collection)", count)

                elif count := gc.collect(0):
                    logger.trace("Collected {} objects", count)

                now = system_clock()
                frames.append(now)
                frame_period = now - frame_time
                housekeeping += now
                frame_time += 1 / self.target_fps
                if self.realtime:
                    wait_time = frame_time - now
                    performance = min(performance + 0.001, 2) if wait_time > 0.001 else max(0.5, performance - 0.01)
                else:
                    wait_time = 0.001
                    performance = 1
                if wait_time > 0:
                    slow_frame = False
                    await asyncio.sleep(wait_time)
                else:
                    slow_frame = True
                    logger.trace("Slow frame - {:.0f}ms", frame_period * 1000)
                    await asyncio.sleep(0)
                    frame_time = system_clock()

                if len(frames) > 1 and frames[-1] - frames[0] > 5:
                    nframes = len(frames) - 1
                    fps = nframes / (frames[-1] - frames[0])
                    logger.info("{:4.1f}fps; {:4.1f}/{:4.1f}/{:4.1f}ms (run/render/sys); perf {:.2f}",
                                fps, 1000 * execution / nframes, 1000 * render / nframes, 1000 * housekeeping / nframes, performance)
                    frames = frames[-1:]
                    execution = render = housekeeping = 0
                    logger.trace("State dictionary size: {} keys", len(self.state))
                    if run_program is not None:
                        logger.trace("VM stack size: {:d}", run_program.stack.size)

        finally:
            self.global_state = {}
            self._references = {}
            self.pages = []
            program = run_program = current_program = context = None
            SharedCache.clean(0)
            for renderers in self.renderers.values():
                while renderers:
                    renderers.pop().destroy()
            if self.vm_stats:
                log_vm_stats()
            count = gc.collect(2)
            logger.trace("Collected {} objects (full collection)", count)
            counts = numbers_cache_counts()
            if counts:
                logger.debug("Numbers cache: {}", ", ".join(f"{size}x{count}" for size, count in counts.items()))
            empty_numbers_cache()
            gc.enable()
