"""
The main Flitter engine
"""

# pylama:ignore=W0703,R0902,R0912,R0913,R0914,R0915,R1702,C901

import asyncio
import csv
import gc
import logging
from pathlib import Path
import pickle

from ..clock import BeatCounter
from ..interface.controls import Pad, Encoder
from ..interface.osc import OSCReceiver, OSCSender, OSCMessage, OSCBundle
from ..language.parser import parse
from ..model import Context, Vector, null
from ..render import process, window, laser, dmx


Log = logging.getLogger(__name__)


class Controller:
    SEND_PORT = 47177
    RECEIVE_PORT = 47178

    def __init__(self, root_dir, max_fps=60, screen=0, fullscreen=False, vsync=False, state_file=None, multiprocess=True, autoreset=None):
        self.root_dir = Path(root_dir)
        self.max_fps = max_fps
        self.screen = screen
        self.fullscreen = fullscreen
        self.vsync = vsync
        self.multiprocess = multiprocess
        self.autoreset = autoreset
        self.state_timestamp = None
        self.state_file = Path(state_file) if state_file is not None else None
        if self.state_file is not None and self.state_file.exists():
            Log.info("Recover state from state file: %s", self.state_file)
            with open(self.state_file, 'rb') as file:
                self.global_state = pickle.load(file)
        else:
            self.global_state = {}
        self.global_state_dirty = False
        self.state = None
        self.program_top = None
        self.windows = []
        self.lasers = []
        self.dmx = []
        self.counter = BeatCounter()
        self.pads = {}
        self.encoders = {}
        self.osc_sender = OSCSender('localhost', self.SEND_PORT)
        self.osc_receiver = OSCReceiver('localhost', self.RECEIVE_PORT)
        self.queue = []
        self.read_cache = {}
        self.csv_cache = {}
        self.pages = []
        self.next_page = None
        self.current_page = None
        self.current_filename = None
        self.current_mtime = None

    def load_page(self, filename):
        page_number = len(self.pages)
        filename = Path(filename)
        mtime = filename.stat().st_mtime
        program_top = self.load_source(filename)
        Log.info("Loaded page %i: %s", page_number, filename)
        self.pages.append((filename, mtime, program_top, self.global_state.setdefault(page_number, {})))

    def switch_to_page(self, page_number):
        if self.pages is not None and 0 <= page_number < len(self.pages):
            self.pads = {}
            self.encoders = {}
            filename, mtime, program_top, state = self.pages[page_number]
            self.state = state
            self.program_top = program_top
            self.current_page = page_number
            self.current_filename = filename
            self.current_mtime = mtime
            Log.info("Switched to page %i: %s", page_number, filename)
            self.enqueue_reset()
            counter_state = self.get(('_counter',))
            if counter_state is not None:
                tempo, quantum, start = counter_state
                self.counter.update(tempo, int(quantum), start)
                Log.info("Restore counter at beat %.1f, tempo %.1f, quantum %d", self.counter.beat, self.counter.tempo, self.counter.quantum)
                self.enqueue_tempo()
            self.enqueue_page_status()
            for window in self.windows:
                window.purge()

    @staticmethod
    def load_source(filename):
        with open(filename, encoding='utf8') as file:
            return parse(file.read()).partially_evaluate(Context())

    def get(self, key, default=None):
        key = Vector.coerce(key)
        if key not in self.state:
            return default
        value = self.state[key]
        if len(value) == 0:
            return default
        if len(value) == 1:
            a, = value
            return a
        return tuple(value)

    def __contains__(self, key):
        return Vector.coerce(key) in self.state

    def __getitem__(self, key):
        key = Vector.coerce(key)
        value = self.state[key]
        if len(value) == 0:
            return default
        if len(value) == 1:
            a, = value
            return a
        return tuple(value)

    def __setitem__(self, key, value):
        key = Vector.coerce(key)
        value = Vector.coerce(value)
        if key not in self.state or value != self.state[key]:
            self.state[key] = value
            self.global_state_dirty = True
            self.state_timestamp = self.counter.clock()
            Log.debug("State changed: %r = %r", key, value)

    def read(self, filename):
        filename = str(filename)
        if filename:
            path = self.root_dir / filename
            if path.exists():
                path_mtime = path.stat().st_mtime
                if path in self.read_cache:
                    text, mtime = self.read_cache[path]
                    if path_mtime == mtime:
                        return text
                text = Vector((path.open(encoding='utf8').read(),))
                self.read_cache[path] = text, path_mtime
                Log.info("Read: %s", filename)
                return text
        return null

    def csv(self, filename, line_number):
        filename = str(filename)
        line_number = line_number.match(1, int)
        lines = reader = None
        if filename and line_number is not None:
            path = self.root_dir / filename
            if path.exists():
                path_mtime = path.stat().st_mtime
                if path in self.csv_cache:
                    cached_lines, cached_reader, mtime = self.csv_cache[path]
                    if path_mtime == mtime:
                        lines, reader = cached_lines, cached_reader
                if lines is None:
                    lines = []
                    reader = csv.reader(path.open(encoding='utf8'))
                    self.csv_cache[path] = lines, reader, path_mtime
                    Log.info("Read CSV: %s", filename)
            if lines is not None:
                while reader is not None and line_number >= len(lines):
                    try:
                        row = next(reader)
                        values = []
                        for value in row:
                            try:
                                value = float(value)
                            except ValueError:
                                pass
                            values.append(value)
                        lines.append(Vector(values))
                    except StopIteration:
                        self.csv_cache[path] = lines, None, path_mtime
                        break
                if line_number < len(lines):
                    return lines[line_number]
        return null

    async def update_windows(self, graph, **kwargs):
        count = 0
        async with asyncio.TaskGroup() as group:
            references = {}
            for i, node in enumerate(graph.select_below('window.')):
                if i == len(self.windows):
                    if self.multiprocess:
                        w = process.Proxy('flitter.render.window.Window', screen=self.screen, fullscreen=self.fullscreen, vsync=self.vsync)
                    else:
                        w = window.Window(screen=self.screen, fullscreen=self.fullscreen, vsync=self.vsync)
                    self.windows.append(w)
                group.create_task(self.windows[i].update(node, references=references, **kwargs))
                count += 1
        while len(self.windows) > count:
            self.windows.pop().destroy()

    async def update_lasers(self, graph):
        count = 0
        async with asyncio.TaskGroup() as group:
            for i, node in enumerate(graph.select_below('laser.')):
                if i == len(self.lasers):
                    if self.multiprocess:
                        l = process.Proxy('flitter.render.laser.Laser')
                    else:
                        l = laser.Laser()
                    self.lasers.append(l)
                group.create_task(self.lasers[i].update(node))
                count += 1
        while len(self.lasers) > count:
            self.lasers.pop().destroy()

    async def update_dmx(self, graph):
        count = 0
        async with asyncio.TaskGroup() as group:
            for i, node in enumerate(graph.select_below('dmx.')):
                if i == len(self.dmx):
                    if self.multiprocess:
                        d = process.Proxy('flitter.render.dmx.DMX')
                    else:
                        d = dmx.DMX()
                    self.dmx.append(d)
                group.create_task(self.dmx[i].update(node))
                count += 1
        while len(self.dmx) > count:
            self.dmx.pop().destroy()

    def update_controls(self, graph):
        remaining = set(self.pads)
        for node in graph.select_below('pad.'):
            number = node.get('number', 2, int)
            if number is not None:
                number = tuple(number)
                if number not in self.pads:
                    Log.debug("New pad @ %r", number)
                    pad = self.pads[number] = Pad(number)
                elif number in remaining:
                    pad = self.pads[number]
                    remaining.remove(number)
                else:
                    continue
                if pad.update(node, self):
                    self.enqueue_pad_status(pad)
        for number in remaining:
            self.enqueue_pad_status(self.pads[number], deleted=True)
            del self.pads[number]
        remaining = set(self.encoders)
        for node in graph.select_below('encoder.'):
            number = node.get('number', 1, int)
            if number is not None:
                if number not in self.encoders:
                    Log.debug("New encoder @ %r", number)
                    encoder = self.encoders[number] = Encoder(number)
                elif number in remaining:
                    encoder = self.encoders[number]
                    remaining.remove(number)
                else:
                    continue
                if encoder.update(node, self):
                    self.enqueue_encoder_status(encoder)
        for number in remaining:
            self.enqueue_encoder_status(self.encoders[number], deleted=True)
            del self.encoders[number]

    def enqueue_reset(self):
        self.queue.append(OSCMessage('/reset'))

    def enqueue_pad_status(self, pad, deleted=False):
        address = '/pad/' + '/'.join(str(n) for n in pad.number) + '/state'
        if deleted:
            self.queue.append(OSCMessage(address))
        else:
            self.queue.append(OSCMessage(address, pad.name, *pad.color, pad.quantize, pad.touched, pad.toggled))

    def enqueue_encoder_status(self, encoder, deleted=False):
        address = f'/encoder/{encoder.number}/state'
        if deleted:
            self.queue.append(OSCMessage(address))
        else:
            self.queue.append(OSCMessage(address, encoder.name, *encoder.color, encoder.touched, encoder.value,
                                         encoder.lower, encoder.upper, encoder.origin, encoder.decimals, encoder.percent))

    def enqueue_tempo(self):
        self.queue.append(OSCMessage('/tempo', self.counter.tempo, self.counter.quantum, self.counter.start))

    def enqueue_page_status(self):
        self.queue.append(OSCMessage('/page_left', self.current_page > 0))
        self.queue.append(OSCMessage('/page_right', self.current_page < len(self.pages)-1))

    def process_message(self, message):
        if isinstance(message, OSCBundle):
            for element in message.elements:
                self.process_message(element)
            return
        Log.debug("Received OSC message: %r", message)
        parts = message.address.strip('/').split('/')
        if parts[0] == 'hello':
            self.enqueue_tempo()
            for pad in self.pads.values():
                self.enqueue_pad_status(pad)
            for encoder in self.encoders.values():
                self.enqueue_encoder_status(encoder)
            self.enqueue_page_status()
        elif parts[0] == 'tempo':
            tempo, quantum, start = message.args
            self.counter.update(tempo, int(quantum), start)
            self[('_counter',)] = tempo, int(quantum), start
            self.enqueue_tempo()
        elif parts[0] == 'pad':
            number = tuple(int(n) for n in parts[1:-1])
            if number in self.pads:
                pad = self.pads[number]
                timestamp, *args = message.args
                beat = self.counter.beat_at_time(timestamp)
                toggled = None
                if parts[-1] == 'touched':
                    pad.on_touched(beat)
                    toggled = pad.on_pressure(beat, *args)
                elif parts[-1] == 'held':
                    toggled = pad.on_pressure(beat, *args)
                elif parts[-1] == 'released':
                    pad.on_pressure(beat, 0.0)
                    pad.on_released(beat)
                if toggled and pad.group is not None:
                    for other in self.pads.values():
                        if other is not pad and other.group == pad.group and other.toggled:
                            other.toggled = False
                            other._toggled_beat = beat  # noqa
                            self.enqueue_pad_status(other)
        elif parts[0] == 'encoder':
            number = int(parts[1])
            if number in self.encoders:
                encoder = self.encoders[number]
                timestamp, *args = message.args
                beat = self.counter.beat_at_time(timestamp)
                if parts[-1] == 'touched':
                    encoder.on_touched(beat)
                elif parts[-1] == 'turned':
                    encoder.on_turned(beat, *args)
                elif parts[-1] == 'released':
                    encoder.on_released(beat)
                elif parts[-1] == 'reset':
                    encoder.on_reset(beat)
        elif parts == ['page_left']:
            if self.current_page > 0:
                self.next_page = self.current_page - 1
        elif parts == ['page_right']:
            if self.current_page < len(self.pages) - 1:
                self.next_page = self.current_page + 1

    async def receive_messages(self):
        Log.info("Listening for OSC control messages on port %d", self.RECEIVE_PORT)
        while True:
            message = await self.osc_receiver.receive()
            self.process_message(message)

    def handle_pragmas(self, pragmas):
        counter_state = self.get(('_counter',))
        if counter_state is None:
            tempo = pragmas.get('tempo')
            if tempo is not None and len(tempo) == 1 and isinstance(tempo[0], float) and tempo[0] > 0:
                tempo = tempo[0]
            else:
                tempo = 120
            quantum = pragmas.get('quantum')
            if quantum is not None and len(quantum) == 1 and isinstance(quantum[0], float) and quantum[0] >= 2:
                quantum = int(quantum[0])
            else:
                quantum = 4
            self.counter.update(tempo, quantum, self.counter.clock())
            self[('_counter',)] = self.counter.tempo, self.counter.quantum, self.counter.start
            self.enqueue_tempo()
            Log.info("Start counter, tempo %d, quantum %d", self.counter.tempo, self.counter.quantum)

    def reset_state(self):
        for key in [key for key in self.state.keys() if not (len(key) == 1 and key[0].startswith('_'))]:
            del self.state[key]
        for pad in self.pads.values():
            pad.reset()
        for encoder in self.encoders.values():
            encoder.reset()
        self.state_timestamp = None
        self.global_state_dirty = True

    def debug(self, value):
        Log.debug("%r", value)
        return value

    async def run(self):
        try:
            loop = asyncio.get_event_loop()
            loop.create_task(self.receive_messages())
            frames = []
            self.enqueue_reset()
            self.enqueue_page_status()
            reload_task = None
            frame_time = self.counter.clock()
            last = self.counter.beat_at_time(frame_time)
            dump_time = frame_time
            execution = render = housekeeping = 0
            performance = 1
            gc.disable()
            while True:
                frames.append(frame_time)

                execution -= self.counter.clock()
                beat = self.counter.beat_at_time(frame_time)
                delta = beat - last
                last = beat
                names = {'beat': beat, 'quantum': self.counter.quantum, 'tempo': self.counter.tempo,
                         'delta': delta, 'clock': frame_time, 'performance': performance}
                context = self.program_top.run(self.state, read=self.read, csv=self.csv, debug=self.debug, **names)
                execution += self.counter.clock()
                render -= self.counter.clock()
                self.handle_pragmas(context.pragmas)
                self.update_controls(context.graph)
                async with asyncio.TaskGroup() as group:
                    group.create_task(self.update_windows(context.graph, **names))
                    group.create_task(self.update_lasers(context.graph))
                    group.create_task(self.update_dmx(context.graph))
                render += self.counter.clock()

                housekeeping -= self.counter.clock()
                if self.autoreset and self.state_timestamp and self.counter.clock() > self.state_timestamp + self.autoreset:
                    self.reset_state()
                count = gc.collect(0)
                if count:
                    Log.debug("Collected %d objects", count)
                if self.queue:
                    await self.osc_sender.send_bundle_from_queue(self.queue)

                if self.global_state_dirty and self.state_file is not None and frame_time > dump_time + 1:
                    Log.debug("Saving state")
                    with open(self.state_file, 'wb') as file:
                        pickle.dump(self.global_state, file)
                    self.global_state_dirty = False
                    dump_time = frame_time

                if self.next_page is not None:
                    if reload_task is not None:
                        reload_task.cancel()
                        reload_task = None
                    if self.autoreset:
                        self.reset_state()
                    self.switch_to_page(self.next_page)
                    self.next_page = None
                    performance = 1
                    count = gc.collect(2)
                    Log.debug("Collected %d objects (full collection)", count)

                mtime = self.current_filename.stat().st_mtime
                if mtime > self.current_mtime:
                    if reload_task is None:
                        Log.debug("Begin reload of page %i: %s", self.current_page, self.current_filename)
                        reload_task = loop.run_in_executor(None, self.load_source, self.current_filename)
                    elif reload_task.done():
                        try:
                            program_top = await reload_task
                            self.program_top = program_top
                            self.pages[self.current_page] = self.current_filename, self.current_mtime, self.program_top, self.state
                            Log.info("Reloaded page %i: %s", self.current_page, self.current_filename)
                        except Exception:
                            Log.exception("Error reloading page")
                        finally:
                            reload_task = None
                        self.current_mtime = mtime

                frame_time += 1 / self.max_fps
                now = self.counter.clock()
                if frame_time > now:
                    delay = frame_time - now
                    await asyncio.sleep(delay)
                    housekeeping -= delay
                    if delay > 0.05/self.max_fps:
                        performance = min(1.5, performance * 1.001)
                else:
                    Log.debug("Slow frame - %.0fms", (now - frame_time + 1/self.max_fps) * 1000)
                    await asyncio.sleep(0)
                    frame_time = self.counter.clock()
                    performance = max(0.5, performance * 0.999)
                housekeeping += self.counter.clock()

                if len(frames) > 1 and frames[-1] - frames[0] > 5:
                    nframes = len(frames) - 1
                    fps = nframes / (frames[-1] - frames[0])
                    Log.info("%.1ffps; /frame execute %.1fms, render %.1fms, housekeep %.1fms; perf %.2f",
                             fps, 1000*execution/nframes, 1000*render/nframes, 1000*housekeeping/nframes, performance)
                    frames = frames[-1:]
                    execution = render = housekeeping = 0
        finally:
            while self.windows:
                self.windows.pop().destroy()
            count = gc.collect(2)
            Log.debug("Collected %d objects (full collection)", count)
            gc.enable()
