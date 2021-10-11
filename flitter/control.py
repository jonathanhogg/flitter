"""
The main Flitter engine
"""

import asyncio
import logging
from pathlib import Path

from .clock import BeatCounter
from .interface.controls import Pad, Encoder
from .interface.osc import OSCReceiver, OSCSender, OSCMessage, OSCBundle
from .language.interpreter import simplify, evaluate
from .language.parser import parse
from .language.ast import Literal
from .model import Context, Vector, Node, null
from .render.scene import Window


Log = logging.getLogger(__name__)


class Controller:
    SEND_PORT = 47177
    RECEIVE_PORT = 47178

    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.state = {}
        self.tree = None
        self.simplified = None
        self.windows = []
        self.counter = BeatCounter()
        self.pads = {}
        self.encoders = {}
        self.osc_sender = OSCSender('localhost', self.SEND_PORT)
        self.osc_receiver = OSCReceiver('localhost', self.RECEIVE_PORT)
        self.queue = []
        self.read_cache = {}
        self.pages = []
        self.current_page = None
        self.current_filename = None
        self.current_mtime = None

    def load_page(self, filename):
        page_number = len(self.pages)
        filename = Path(filename)
        with open(filename, encoding='utf8') as file, Context() as context:
            tree = simplify(parse(file.read()), context)
        Log.info("Loaded page %i: %s", page_number, filename)
        self.pages.append((filename, filename.stat().st_mtime, tree))

    def switch_to_page(self, page_number):
        if self.pages is not None and 0 <= page_number < len(self.pages):
            filename, mtime, tree = self.pages[page_number]
            self.tree = tree
            self.simplified = None
            self.current_page = page_number
            self.current_filename = filename
            self.current_mtime = mtime
            Log.info("Switched to page %i: %s", page_number, filename)

    def reload_current_page(self):
        with open(self.current_filename, encoding='utf8') as file, Context() as context:
            self.tree = simplify(parse(file.read()), context)
        self.simplified = None
        self.current_mtime = self.current_filename.stat().st_mtime
        self.pages[self.current_page] = self.current_filename, self.current_mtime, self.tree
        Log.info("Reloaded page %i: %s", self.current_page, self.current_filename)

    def get(self, key, default=None):
        return self.state.get(key, default)

    def __contains__(self, key):
        return key in self.state

    def __getitem__(self, key):
        return self.state[key]

    def __setitem__(self, key, value):
        if key not in self.state or value != self.state[key]:
            self.state[key] = value
            self.simplified = None
            Log.debug("State changed: %r = %r", key, value)

    def read(self, filename):
        if len(filename) == 1 and filename.isinstance(str):
            path = self.root_dir / filename[0]
            if path in self.read_cache:
                text, mtime = self.read_cache[path]
                if path.stat().st_mtime == mtime:
                    return text
            text = Vector((path.open(encoding='utf8').read(),))
            self.read_cache[path] = text, path.stat().st_mtime
            Log.info("Read: %s", filename)
            return text
        return null

    def execute(self):
        if self.simplified is None:
            with Context(state=self.state) as context:
                self.simplified = simplify(self.tree, context)
        beat = self.counter.beat
        variables = {'beat': Vector((beat,)),
                     'quantum': Vector((self.counter.quantum,)),
                     'clock': Vector((self.counter.time_at_beat(beat),)),
                     'read': Vector((self.read,))}
        with Context(variables=variables, state=self.state) as context:
            expressions = [self.simplified] if isinstance(self.simplified, Literal) else self.simplified.expressions
            for expr in expressions:
                result = evaluate(expr, context)
                for value in result:
                    if isinstance(value, Node) and value.parent is None:
                        context.graph.append(value)
        return context.graph

    def update_windows(self, graph):
        count = 0
        for i, node in enumerate(graph.select_below('window.')):
            if i == len(self.windows):
                self.windows.append(Window())
            self.windows[i].update(node)
            count += 1
        while len(self.windows) > count:
            self.windows.pop().destroy()

    def update_controls(self, graph):
        remaining = set(self.pads)
        for node in graph.select_below('pad.'):
            if 'number' in node and node['number']:
                number = node['number']
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
            if 'number' in node and node['number']:
                number = node['number']
                if number not in self.encoders:
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

    def enqueue_pad_status(self, pad, deleted=False):
        address = '/pad/' + '/'.join(str(int(n)) for n in pad.number) + '/state'
        if deleted:
            self.queue.append(OSCMessage(address))
        else:
            self.queue.append(OSCMessage(address, pad.name, *pad.color, pad.touched, pad.toggled))

    def enqueue_encoder_status(self, encoder, deleted=False):
        address = '/encoder/' + '/'.join(str(int(n)) for n in encoder.number) + '/state'
        if deleted:
            self.queue.append(OSCMessage(address))
        else:
            self.queue.append(OSCMessage(address, encoder.name, *encoder.color, encoder.touched, encoder.value, encoder.lower, encoder.upper))

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
            self.counter.update(tempo, quantum, start)
            self.enqueue_tempo()
        elif parts[0] == 'pad':
            number = Vector(float(n) for n in parts[1:-1])
            if number in self.pads:
                pad = self.pads[number]
                timestamp, *args = message.args
                beat = self.counter.beat_at_time(timestamp)
                if parts[-1] == 'touched':
                    pad.on_touched(beat)
                    pad.on_pressure(beat, *args)
                elif parts[-1] == 'held':
                    pad.on_pressure(beat, *args)
                elif parts[-1] == 'released':
                    pad.on_pressure(beat, 0.0)
                    pad.on_released(beat)
        elif parts[0] == 'encoder':
            number = Vector(float(n) for n in parts[1:-1])
            if number in self.encoders:
                pad = self.encoders[number]
                timestamp, *args = message.args
                beat = self.counter.beat_at_time(timestamp)
                if parts[-1] == 'touched':
                    pad.on_touched(beat)
                elif parts[-1] == 'turned':
                    pad.on_turned(beat, *args)
                elif parts[-1] == 'released':
                    pad.on_released(beat)
        elif parts == ['page_left']:
            if self.current_page > 0:
                self.switch_to_page(self.current_page - 1)
                self.enqueue_page_status()
        elif parts == ['page_right']:
            if self.current_page < len(self.pages) - 1:
                self.switch_to_page(self.current_page + 1)
                self.enqueue_page_status()

    async def receive_messages(self):
        while True:
            message = await self.osc_receiver.receive()
            self.process_message(message)

    async def run(self):
        asyncio.get_event_loop().create_task(self.receive_messages())
        frames = []
        self.enqueue_tempo()
        self.enqueue_page_status()
        while True:
            if self.current_filename.stat().st_mtime > self.current_mtime:
                try:
                    self.reload_current_page()
                except Exception:
                    Log.exception("Syntax error")
            graph = self.execute()
            self.update_windows(graph)
            self.update_controls(graph)
            if self.queue:
                await self.osc_sender.send_bundle_from_queue(self.queue)
            else:
                await asyncio.sleep(0)
            frames.append(self.counter.clock())
            if len(frames) > 1 and frames[-1] - frames[0] > 2:
                fps = (len(frames) - 1) / (frames[-1] - frames[0])
                Log.info("Frame rate = %.1ffps", fps)
                frames = []
