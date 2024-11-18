"""
The main Flitter engine
"""

import asyncio
import gc
from pathlib import Path
import pickle
import time

from loguru import logger

from .. import setproctitle
from ..cache import SharedCache
from ..clock import BeatCounter, system_clock
from ..language.vm import log_vm_stats
from ..model import Vector, StateDict, Context, null, numbers_cache_counts, empty_numbers_cache
from ..plugins import get_plugin
from ..render.window.models import Model


class EngineController:
    STATE_SAVE_PERIOD = 1

    def __init__(self, target_fps=60, screen=0, fullscreen=False, vsync=False, state_file=None,
                 reset_on_switch=False, state_simplify_wait=0, realtime=True, defined_names=None, vm_stats=False,
                 run_time=None, offscreen=False, disable_simplifier=False, opengl_es=False):
        self.default_fps = target_fps
        self.target_fps = target_fps
        self.realtime = realtime
        self.screen = screen
        self.fullscreen = fullscreen
        self.vsync = vsync
        self.offscreen = offscreen
        self.opengl_es = opengl_es
        self.reset_on_switch = reset_on_switch
        self.disable_simplifier = disable_simplifier
        self.state_simplify_wait = 0 if self.disable_simplifier else state_simplify_wait / 2
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
            if self.state is not None:
                if self.reset_on_switch:
                    self.state.clear()
                self.global_state_dirty = self.global_state_dirty or self.state.changed
                self.state.clear_changed()
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
            setproctitle(f'flitter {self.current_path}')

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
        updated_references = []
        for kind, nodes in nodes_by_kind.items():
            renderer_class = get_plugin('flitter.render', kind)
            if renderer_class is not None:
                renderers = self.renderers.setdefault(kind, [])
                count = 0
                for node in nodes:
                    if count == len(renderers):
                        renderer = renderer_class(**kwargs)
                        renderers.append(renderer)
                    references = dict(self._references)
                    tasks.append(asyncio.create_task(renderers[count].update(self, node, references=references, **kwargs)))
                    updated_references.append(references)
                    count += 1
                while len(renderers) > count:
                    await renderers.pop().destroy()
        for kind in list(self.renderers):
            if kind not in nodes_by_kind:
                for renderer in self.renderers.pop(kind):
                    await renderer.destroy()
        try:
            await asyncio.gather(*tasks)
        except Exception:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        for references in updated_references:
            self._references.update(references)

    def handle_pragmas(self, pragmas, timestamp):
        if '_counter' not in self.state:
            tempo = pragmas.get('tempo', null).match(1, float, 120)
            if tempo != self.counter.tempo:
                self.counter.set_tempo(tempo, timestamp)
            quantum = pragmas.get('quantum', null).match(1, float, 4)
            if quantum != self.counter.quantum:
                self.counter.set_quantum(quantum, timestamp)
        self.target_fps = pragmas.get('fps', null).match(1, float, self.default_fps)

    async def run(self):
        try:
            frame_count = 0
            frames = []
            start_time = frame_time = system_clock() if self.realtime else self.counter.start
            last = self.counter.beat_at_time(frame_time)
            save_state_time = system_clock() + self.STATE_SAVE_PERIOD
            execution = render = housekeeping = 0
            slow_frame = False
            performance = 1 if self.realtime else 2
            run_program = current_program = errors = logs = None
            simplify_state_time = system_clock() + self.state_simplify_wait
            static = dict(self.defined_names)
            static.update({'realtime': self.realtime, 'screen': self.screen, 'fullscreen': self.fullscreen,
                           'vsync': self.vsync, 'offscreen': self.offscreen, 'opengl_es': self.opengl_es, 'run_time': self.run_time})
            gc.disable()
            while self.run_time is None or int(round((frame_time - start_time) * self.target_fps)) < int(round(self.run_time * self.target_fps)):
                housekeeping -= system_clock()

                beat = self.counter.beat_at_time(frame_time)
                delta = beat - last
                last = beat
                dynamic = {'beat': beat, 'quantum': self.counter.quantum, 'tempo': self.counter.tempo, 'fps': self.target_fps,
                           'delta': delta, 'time': frame_time, 'frame': frame_count, 'performance': performance, 'slow_frame': slow_frame,
                           'clock': time.time()}
                names = dict(static)
                names.update(dynamic)

                program = self.current_path.read_flitter_program(static=static, dynamic=dynamic, simplify=not self.disable_simplifier)
                if program is not current_program:
                    level = 'SUCCESS' if current_program is None else 'INFO'
                    logger.log(level, "Loaded page {}: {}", self.current_page, self.current_path)
                    run_program = current_program = program
                    self.handle_pragmas(program.pragmas, frame_time)
                    errors = logs = None
                    self.state_generation0 ^= self.state_generation1
                    self.state_generation1 = self.state_generation2
                    self.state_generation2 = set()
                    simplify_state_time = system_clock() + self.state_simplify_wait

                now = system_clock()
                housekeeping += now
                execution -= now

                if run_program is not None:
                    context = Context(names={key: Vector.coerce(value) for key, value in dynamic.items()},
                                      state=self.state, references=self._references)
                    run_program.run(context, record_stats=self.vm_stats)
                else:
                    context = Context()

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

                await self.update_renderers(context.root, **names)

                now = system_clock()
                render += now
                housekeeping -= now

                del context
                SharedCache.clean()

                self.state['_counter'] = self.counter.tempo, self.counter.quantum, self.counter.start

                if self.state.changed:
                    if self.state_simplify_wait:
                        changed_keys = self.state.changed_keys - self.state_generation0
                        self.state_generation0 ^= changed_keys
                        self.state_generation0 &= self.state.keys()
                        if changed_keys:
                            generation1to0 = self.state_generation1 & changed_keys
                            changed_keys -= generation1to0
                            self.state_generation1 -= generation1to0
                            generation2to0 = self.state_generation2 & changed_keys
                            if generation2to0:
                                if run_program is not current_program:
                                    run_program = current_program
                                    logger.debug("Undo simplification on state; original program with {} instructions", len(run_program))
                                self.state_generation1 ^= self.state_generation2 - generation2to0
                                self.state_generation2 = set()
                                simplify_state_time = system_clock() + self.state_simplify_wait
                    self.global_state_dirty = True
                    self.state_timestamp = system_clock()
                    self.state.clear_changed()

                if self.state_simplify_wait and system_clock() > simplify_state_time:
                    if current_program is not None and self.state_generation1:
                        if self.state_generation1:
                            self.state_generation2 ^= self.state_generation1
                            simplify_state = self.state.with_keys(self.state_generation2)
                            simplify_time = -system_clock()
                            top = current_program.top.simplify(state=simplify_state, dynamic=dynamic, path=current_program.path)
                            now = system_clock()
                            simplify_time += now
                            if top is current_program.top:
                                logger.trace("Program unchanged after simplification on {} static state keys in {:.1f}ms",
                                             len(self.state_generation2), simplify_time*1000)
                            else:
                                compile_time = -now
                                run_program = top.compile(initial_lnames=tuple(dynamic))
                                run_program.set_path(current_program.path)
                                run_program.set_top(top)
                                compile_time += system_clock()
                                logger.debug("Simplified on {} static state keys to {} instructions in {:.1f}/{:.1f}ms",
                                             len(self.state_generation2), len(run_program), simplify_time*1000, compile_time*1000)
                    self.state_generation1 = self.state_generation0
                    self.state_generation0 = set()
                    simplify_state_time = system_clock() + self.state_simplify_wait

                if self.global_state_dirty and self.state_file is not None and system_clock() > save_state_time:
                    logger.trace("Saving state")
                    with open(self.state_file, 'wb') as file:
                        pickle.dump(self.global_state, file)
                    self.global_state_dirty = False
                    save_state_time = system_clock() + self.STATE_SAVE_PERIOD

                if self.switch_page is not None:
                    self.switch_to_page(self.switch_page)
                    self.switch_page = None
                    run_program = current_program = None
                    performance = 1
                    for renderers in self.renderers.values():
                        for renderer in renderers:
                            await renderer.purge()

                frame_count += 1
                frames.append(frame_time if self.realtime else now)
                frame_period = now - frame_time
                frame_time += 1 / self.target_fps
                if self.realtime:
                    wait_time = frame_time - now
                    performance = min(performance + 0.001, 2) if wait_time > 0.001 else max(0.5, performance - 0.01)
                else:
                    wait_time = 0.001
                if wait_time >= 0.001:
                    slow_frame = False
                    housekeeping -= wait_time - 0.001
                    await asyncio.sleep(wait_time - 0.001)
                else:
                    slow_frame = True
                    logger.trace("Slow frame - {:.0f}ms", frame_period * 1000)
                    await asyncio.sleep(0)
                    frame_time = system_clock()

                now = system_clock()
                housekeeping += now

                if len(frames) > 1 and frames[-1] - frames[0] > 5:
                    nframes = len(frames) - 1
                    fps = nframes / (frames[-1] - frames[0])
                    logger.info("{:4.1f}fps; {:4.1f}/{:4.1f}/{:4.1f}ms (run/render/sys); perf {:.2f}",
                                fps, 1000 * execution / nframes, 1000 * render / nframes, 1000 * housekeeping / nframes, performance)
                    frames = frames[-1:]
                    execution = render = housekeeping = 0
                    logger.trace("State dictionary size: {} keys", len(self.state))
                    if run_program is not None and run_program.stack is not None:
                        logger.trace("VM stack size: {:d}", run_program.stack.size)
                    Model.flush_caches()
                else:
                    gc.collect(0)

        finally:
            self.global_state = {}
            self._references = {}
            self.pages = []
            program = run_program = current_program = context = None
            Model.flush_caches(0, 0)
            SharedCache.clean(0)
            for renderers in self.renderers.values():
                while renderers:
                    await renderers.pop().destroy()
            if self.vm_stats:
                log_vm_stats()
            count = gc.collect(2)
            logger.trace("Collected {} objects (full collection)", count)
            counts = numbers_cache_counts()
            if counts:
                logger.debug("Numbers cache: {}", ", ".join(f"{size}x{count}" for size, count in counts.items()))
            empty_numbers_cache()
            gc.enable()
