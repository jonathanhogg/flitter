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
from ..language.vm import log_vm_stats, StableChanged
from ..model import Vector, StateDict, Context, null, numbers_cache_counts, empty_numbers_cache
from ..plugins import get_plugin
from ..render.window.models import Model


class EngineController:
    STATE_SAVE_PERIOD = 1
    MINIMUM_GC_INTERVAL = 10

    def __init__(self, target_fps=60, screen=0, fullscreen=False, vsync=False, state_file=None,
                 reset_on_switch=False, realtime=True, defined_names=None, vm_stats=False,
                 run_time=None, offscreen=False, disable_simplifier=False, opengl_es=False, model_cache_time=300):
        self.default_fps = target_fps
        self.target_fps = target_fps
        self.realtime = realtime
        self.screen = screen
        self.fullscreen = fullscreen
        self.vsync = vsync
        self.offscreen = offscreen
        self.opengl_es = opengl_es
        self.model_cache_time = model_cache_time
        self.reset_on_switch = reset_on_switch
        self.disable_simplifier = disable_simplifier
        if defined_names:
            self.defined_names = {key: Vector.coerce(value) for key, value in defined_names.items()}
            logger.trace("Defined names: {!r}", self.defined_names)
        else:
            self.defined_names = {}
        self.vm_stats = vm_stats
        self.run_time = run_time
        self.state_file = Path(state_file) if state_file is not None else None
        if self.state_file is not None and self.state_file.exists():
            try:
                with open(self.state_file, 'rb') as file:
                    self.global_state = pickle.load(file)
            except Exception:
                logger.exception("Unable to use state file: {}", self.state_file)
                self.state_file = None
                self.global_state = {}
            else:
                logger.info("Recovered state from state file: {}", self.state_file)
        else:
            self.global_state = {}
        self.global_state_dirty = False
        self.state = None
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
            self.current_path = path
            self.current_page = page_number
            SharedCache.set_root(self.current_path)
            logger.info("Switched to page {}: {}", page_number, self.current_path)
            if counter_state := self.state['_counter']:
                tempo, quantum, start = counter_state
                self.counter.reset(tempo, int(quantum), start)
                logger.info("Restore counter at beat {:.1f}, tempo {:.1f}, quantum {}", self.counter.beat, self.counter.tempo, self.counter.quantum)
            else:
                self.counter.reset(start=system_clock() if self.realtime else 0)
            setproctitle(f'flitter [{self.current_path}]')

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
            start_time = frame_time = system_clock() if self.realtime else 0
            last = self.counter.beat_at_time(frame_time)
            save_state_time = frame_time + self.STATE_SAVE_PERIOD
            execution = render = housekeeping = 0
            slow_frame = False
            performance = 1 if self.realtime else 2
            gc_pending = False
            last_gc = None
            run_program = current_program = errors = logs = None
            stables = set()
            stable_cache = {}
            static = dict(self.defined_names)
            static.update({'realtime': self.realtime, 'screen': self.screen, 'fullscreen': self.fullscreen,
                           'vsync': self.vsync, 'offscreen': self.offscreen, 'opengl_es': self.opengl_es, 'run_time': self.run_time})
            gc.disable()
            housekeeping -= system_clock()
            while self.run_time is None or int(round((frame_time - start_time) * self.target_fps)) < int(round(self.run_time * self.target_fps)):
                beat = self.counter.beat_at_time(frame_time)
                delta = beat - last
                last = beat
                dynamic = {'beat': beat, 'quantum': self.counter.quantum, 'tempo': self.counter.tempo, 'fps': self.target_fps,
                           'delta': delta, 'time': frame_time, 'frame': frame_count, 'performance': performance, 'slow_frame': slow_frame,
                           'clock': time.time()}
                names = dict(static)
                names.update(dynamic)
                if self.disable_simplifier:
                    dynamic.update(static)

                program = self.current_path.read_flitter_program(static=static, dynamic=dynamic, simplify=not self.disable_simplifier)
                if program is not current_program:
                    level = 'SUCCESS' if current_program is None else 'INFO'
                    logger.log(level, "Loaded page {}: {}", self.current_page, self.current_path)
                    run_program = current_program = program
                    self.handle_pragmas(program.pragmas, frame_time)
                    errors = logs = None
                    stables = set()
                    stable_cache = {}

                now = system_clock()
                housekeeping += now
                execution -= now

                if run_program is not None:
                    while True:
                        context = Context(names={key: Vector.coerce(value) for key, value in dynamic.items()},
                                          state=self.state, references=self._references, stables=stables, stable_cache=stable_cache)
                        try:
                            run_program.run(context, record_stats=self.vm_stats)
                        except StableChanged as exc:
                            if run_program is not current_program:
                                logger.debug("Simplified program invalidated due to change of stable value")
                                run_program = current_program
                            else:
                                raise RuntimeError("Compiled program contains unexpected StableAssert") from exc
                        else:
                            break
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

                self.state['_counter'] = self.counter.tempo, self.counter.quantum, self.counter.start

                if not self.disable_simplifier and context.stables_changed and current_program is not None:
                    simplify_time = -system_clock()
                    top = current_program.top.simplify(dynamic=dynamic, path=current_program.path, stables=stables, stable_cache=stable_cache)
                    now = system_clock()
                    simplify_time += now
                    if top is current_program.top:
                        logger.trace("Program unchanged after simplification on stable values in {:.1f}ms", simplify_time*1000)
                    else:
                        compile_time = -now
                        run_program = top.compile(initial_lnames=tuple(dynamic), stables=stables)
                        run_program.set_path(current_program.path)
                        run_program.set_top(top)
                        compile_time += system_clock()
                        logger.debug("Resimplified on stable values to {} instructions in {:.1f}/{:.1f}ms",
                                     len(run_program), simplify_time*1000, compile_time*1000)

                if self.global_state_dirty and self.state_file is not None and frame_time > save_state_time:
                    logger.trace("Saving state")
                    with open(self.state_file, 'wb') as file:
                        pickle.dump(self.global_state, file)
                    self.global_state_dirty = False
                    save_state_time = frame_time + self.STATE_SAVE_PERIOD

                if self.switch_page is not None:
                    self.switch_to_page(self.switch_page)
                    self.switch_page = None
                    run_program = current_program = None
                    performance = 1
                    for renderers in self.renderers.values():
                        for renderer in renderers:
                            await renderer.purge()

                del context
                gc_pending |= SharedCache.clean()
                if self.model_cache_time > 0:
                    gc_pending |= Model.flush_caches(max_age=self.model_cache_time)
                if gc_pending and (last_gc is None or now > last_gc + self.MINIMUM_GC_INTERVAL):
                    count = gc.collect(2)
                    gc_pending = False
                    last_gc = now
                    if count:
                        logger.trace("Collected {} objects (full collection)", count)
                else:
                    gc.collect(0)

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

                if len(frames) > 1 and frames[-1] - frames[0] > 5:
                    now = system_clock()
                    nframes = len(frames) - 1
                    fps = nframes / (frames[-1] - frames[0])
                    logger.info("{:4.1f}fps; {:4.1f}/{:4.1f}/{:4.1f}ms (run/render/sys); perf {:.2f}",
                                fps, 1000 * execution / nframes, 1000 * render / nframes, 1000 * (housekeeping + now) / nframes, performance)
                    frames = frames[-1:]
                    execution = render = 0
                    housekeeping = -now
                    logger.trace("State dictionary size: {} keys", len(self.state))
                    if run_program is not None and run_program.stack is not None:
                        logger.trace("VM stack size: {:d}", run_program.stack.size)
                    count = Model.cache_size()
                    if count:
                        logger.trace("Model cache size: {}", count)

        finally:
            self.global_state = {}
            self._references = {}
            self.pages = []
            program = run_program = current_program = context = None
            Model.flush_caches(max_size=0)
            SharedCache.clean(0)
            for renderers in self.renderers.values():
                while renderers:
                    await renderers.pop().destroy()
            if self.vm_stats:
                log_vm_stats()
            count = gc.collect(2)
            if count:
                logger.debug("Collected {} objects (full collection)", count)
            counts = numbers_cache_counts()
            if counts:
                logger.debug("Numbers cache: {}", ", ".join(f"{size}x{count}" for size, count in counts.items()))
            empty_numbers_cache()
            gc.enable()
