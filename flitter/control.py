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

    def load(self, filename):
        with open(filename, encoding='utf8') as file, Context() as context:
            self.tree = simplify(parse(file.read()), context)
            self.simplified = None

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
            with open(self.root_dir / filename[0], encoding='utf8') as file:
                return Vector((file.read(),))
        return null

    def execute(self):
        if self.simplified is None:
            with Context(state=self.state) as context:
                self.simplified = simplify(self.tree, context)
        variables = {'beat': Vector((self.counter.beat,)), 'read': Vector((self.read,))}
        with Context(variables=variables, state=self.state) as context:
            for expr in self.simplified.expressions:
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

    def update_controls(self, graph, queue):
        remaining = set(self.pads)
        for node in graph.select_below('pad.'):
            if 'number' in node and node['number']:
                number = node['number']
                address = '/pad/' + '/'.join(str(int(n)) for n in number) + '/state'
                if number not in self.pads:
                    Log.debug("New pad @ %r", number)
                    pad = self.pads[number] = Pad(number)
                elif number in remaining:
                    pad = self.pads[number]
                    remaining.remove(number)
                else:
                    continue
                if pad.update(node, self):
                    queue.append(OSCMessage(address, pad.name, *pad.color, pad.touched, pad.pressure, pad.toggled))
        for number in remaining:
            del self.pads[number]
            queue.append(OSCMessage(address))
        remaining = set(self.encoders)
        for node in graph.select_below('encoder.'):
            if 'number' in node and node['number']:
                number = node['number']
                address = '/encoder/' + '/'.join(str(int(n)) for n in number) + '/state'
                if number not in self.encoders:
                    encoder = self.encoders[number] = Encoder(number)
                elif number in remaining:
                    encoder = self.encoders[number]
                    remaining.remove(number)
                else:
                    continue
                if encoder.update(node, self):
                    queue.append(OSCMessage(address, encoder.name, *encoder.color, encoder.touched, encoder.value))
        for number in remaining:
            del self.encoders[number]

    def process_message(self, message):
        if isinstance(message, OSCBundle):
            for element in message.elements:
                self.process_message(element)
            return
        Log.debug("Received OSC message: %r", message)
        parts = message.address.strip('/').split('/')
        if parts == ['tempo']:
            tempo, quantum, start = message.args
            self.counter.update(tempo, quantum, start)
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

    async def receive_messages(self):
        while True:
            message = await self.osc_receiver.receive()
            self.process_message(message)

    async def run(self):
        queue = []
        asyncio.get_event_loop().create_task(self.receive_messages())
        frames = []
        while True:
            graph = self.execute()
            self.update_windows(graph)
            self.update_controls(graph, queue)
            if queue:
                await self.osc_sender.send_bundle_from_queue(queue)
            else:
                await asyncio.sleep(0)
            frames.append(self.counter.clock())
            if len(frames) > 1 and frames[-1] - frames[0] > 2:
                fps = (len(frames) - 1) / (frames[-1] - frames[0])
                Log.info("Frame rate = %.1ffps", fps)
                frames = []
