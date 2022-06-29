"""
Ableton Push OSC controller for Flitter
"""

# pylama:ignore=W0601,C0103,R0912,R0915,R0914,R0902,C901

import argparse
import asyncio
from dataclasses import dataclass
import logging
import math
import sys

import skia

from ..clock import TapTempo
from ..ableton.constants import Encoder, Control, BUTTONS
from ..ableton.events import (ButtonPressed, ButtonReleased, PadPressed, PadHeld, PadReleased,
                              EncoderTurned, EncoderTouched, EncoderReleased, MenuButtonReleased)
from ..ableton.push import Push2
from ..ableton.palette import PrimaryPalette
from .osc import OSCSender, OSCReceiver, OSCBundle


Log = logging.getLogger(__name__)


@dataclass
class PadState:
    name: str
    r: float
    g: float
    b: float
    touched: bool
    toggled: bool


@dataclass
class EncoderState:
    name: str
    r: float
    g: float
    b: float
    touched: bool
    value: float
    lower: float
    upper: float
    origin: float
    decimals: float
    percent: bool


class Controller:
    HELLO_RETRY_INTERVAL = 1
    RECEIVE_TIMEOUT = 5
    RESET_TIMEOUT = 10

    def __init__(self, tempo_control=True):
        self.tempo_control = tempo_control
        self.push = None
        self.osc_sender = OSCSender('localhost', 47178)
        self.osc_receiver = OSCReceiver('localhost', 47177)
        self.pads = {}
        self.encoders = {}
        self.buttons = {}
        self.last_received = None
        self.last_hello = None
        self.updated = asyncio.Event()
        self.touched_pads = set()
        self.touched_encoders = set()

    async def process_message(self, message):
        if isinstance(message, OSCBundle):
            for element in message.elements:
                await self.process_message(element)
            return
        Log.info("Received OSC message: %r", message)
        match message.address.strip('/').split('/'):
            case ['tempo']:
                tempo, quantum, start = message.args
                self.push.counter.update(tempo, quantum, start)
            case ['pad', column, row, 'state']:
                column, row = int(column), int(row)
                if 0 <= column < 8 and 0 <= row < 8 and message.args:
                    state = PadState(*message.args)
                    brightness = 1 if state.touched or state.toggled else 0.5
                    self.push.set_pad_rgb(row * 8 + column, state.r*brightness, state.g*brightness, state.b*brightness)
                    self.pads[column, row] = state
                    if state.touched and (column, row) not in self.touched_pads:
                        await self.osc_sender.send_message(f'/pad/{column}/{row}/released', self.push.counter.clock())
                elif (column, row) in self.pads:
                    self.push.set_pad_rgb(row * 8 + column, 0, 0, 0)
                    del self.pads[column, row]
            case ['encoder', number, 'state']:
                number = int(number)
                if 0 <= number < 8 and message.args:
                    state = EncoderState(*message.args)
                    brightness = 1 if state.touched else 0.5
                    self.push.set_menu_button_rgb(number + 8, state.r*brightness, state.g*brightness, state.b*brightness)
                    self.encoders[number] = state
                    if state.touched and number not in self.touched_encoders:
                        await self.osc_sender.send_message(f'/encoder/{number}/released', self.push.counter.clock())
                elif number in self.encoders:
                    self.push.set_menu_button_rgb(number + 8, 0, 0, 0)
                    del self.encoders[number]
            case ['page_left']:
                enabled, = message.args
                self.push.set_button_white(Control.PAGE_LEFT, 1 if enabled else 0)
                if enabled:
                    if Control.PAGE_LEFT not in self.buttons:
                        self.buttons[Control.PAGE_LEFT] = 1
                elif Control.PAGE_LEFT in self.buttons:
                    del self.buttons[Control.PAGE_LEFT]
            case ['page_right']:
                enabled, = message.args
                self.push.set_button_white(Control.PAGE_RIGHT, 1 if enabled else 0)
                if enabled:
                    if Control.PAGE_RIGHT not in self.buttons:
                        self.buttons[Control.PAGE_RIGHT] = 1
                elif Control.PAGE_RIGHT in self.buttons:
                    del self.buttons[Control.PAGE_RIGHT]
            case ['reset']:
                self.reset()

    def reset(self):
        for column, row in self.pads:
            self.push.set_pad_rgb(row * 8 + column, 0, 0, 0)
        self.pads.clear()
        for number in self.encoders:
            self.push.set_menu_button_rgb(number + 8, 0, 0, 0)
        self.encoders.clear()
        for control in self.buttons:
            self.push.set_button_white(control, 0)
        self.buttons.clear()
        self.push.counter.update(120, 4, self.push.counter.clock())
        self.last_received = None
        self.updated.set()

    async def receive_messages(self):
        while True:
            message = await self.osc_receiver.receive()
            self.last_received = self.push.counter.clock()
            await self.process_message(message)
            self.updated.set()

    async def run(self):
        self.push = Push2(palette=PrimaryPalette())
        self.push.start()
        for n in range(64):
            self.push.set_pad_rgb(n, 0, 0, 0)
        for n in range(16):
            self.push.set_menu_button_rgb(n, 0, 0, 0)
        for n in BUTTONS:
            self.push.set_button_white(n, 0)
        if self.tempo_control:
            self.push.set_button_white(Control.TAP_TEMPO, 1)
            self.push.set_button_white(Control.SHIFT, 1)
        brightness = 1
        self.push.set_led_brightness(brightness)
        self.push.set_display_brightness(brightness)
        self.push.set_touch_strip_position(0)
        shift_pressed = False
        tap_tempo_pressed = False
        tap_tempo = TapTempo(rounding=1)
        asyncio.get_event_loop().create_task(self.receive_messages())
        self.updated.set()
        try:
            wait_event = asyncio.create_task(self.push.get_event())
            wait_update = asyncio.create_task(self.updated.wait())
            while True:
                done, _ = await asyncio.wait({wait_event, wait_update}, timeout=1/10, return_when=asyncio.FIRST_COMPLETED)
                if wait_event in done:
                    event = wait_event.result()
                    wait_event = asyncio.create_task(self.push.get_event())
                    match event:
                        case ButtonPressed(number=Control.SHIFT):
                            shift_pressed = True
                        case ButtonReleased(number=Control.SHIFT):
                            shift_pressed = False
                        case PadPressed():
                            if not tap_tempo_pressed:
                                self.touched_pads.add((event.column, event.row))
                                address = f'/pad/{event.column}/{event.row}/touched'
                                await self.osc_sender.send_message(address, event.timestamp, event.pressure)
                            else:
                                tap_tempo.tap(event.timestamp)
                        case PadHeld():
                            if not tap_tempo_pressed:
                                address = f'/pad/{event.column}/{event.row}/held'
                                await self.osc_sender.send_message(address, event.timestamp, event.pressure)
                        case PadReleased():
                            if not tap_tempo_pressed:
                                self.touched_pads.discard((event.column, event.row))
                                address = f'/pad/{event.column}/{event.row}/released'
                                await self.osc_sender.send_message(address, event.timestamp)
                        case EncoderTouched() if event.number < 8:
                            self.touched_encoders.add(event.number)
                            address = f'/encoder/{event.number}/touched'
                            await self.osc_sender.send_message(address, event.timestamp)
                        case EncoderTurned() if event.number < 8:
                            address = f'/encoder/{event.number}/turned'
                            await self.osc_sender.send_message(address, event.timestamp, event.amount / 400)
                        case EncoderReleased() if event.number < 8:
                            self.touched_encoders.discard(event.number)
                            address = f'/encoder/{event.number}/released'
                            await self.osc_sender.send_message(address, event.timestamp)
                        case EncoderTurned(number=Encoder.TEMPO) if self.tempo_control:
                            if shift_pressed:
                                self.push.counter.quantum = max(2, self.push.counter.quantum + event.amount)
                            else:
                                tempo = max(0.5, (round(self.push.counter.tempo * 2) + event.amount) / 2)
                                self.push.counter.set_tempo(tempo, timestamp=event.timestamp)
                            await self.osc_sender.send_message('/tempo', self.push.counter.tempo, self.push.counter.quantum, self.push.counter.start)
                        case ButtonPressed(number=Control.TAP_TEMPO) if self.tempo_control:
                            tap_tempo_pressed = True
                        case ButtonReleased(number=Control.TAP_TEMPO) if self.tempo_control:
                            tap_tempo.apply(self.push.counter, event.timestamp, backslip_limit=1)
                            tap_tempo_pressed = False
                            await self.osc_sender.send_message('/tempo', self.push.counter.tempo, self.push.counter.quantum, self.push.counter.start)
                        case EncoderTurned(number=Encoder.MASTER):
                            brightness = min(max(0, brightness + event.amount / 200), 1)
                            self.push.set_led_brightness(brightness)
                            self.push.set_display_brightness(brightness)
                        case MenuButtonReleased() if event.row == 1:
                            await self.osc_sender.send_message(f'/encoder/{event.column}/reset', event.timestamp)
                        case ButtonReleased(number=Control.PAGE_LEFT):
                            await self.osc_sender.send_message('/page_left')
                        case ButtonReleased(number=Control.PAGE_RIGHT):
                            await self.osc_sender.send_message('/page_right')
                    self.updated.set()
                elif wait_update in done:
                    self.updated.clear()
                    wait_update = asyncio.create_task(self.updated.wait())
                    async with self.push.screen_canvas() as canvas:
                        canvas.clear(skia.ColorBLACK)
                        paint = skia.Paint(Color=skia.ColorWHITE, AntiAlias=True)
                        font = skia.Font(skia.Typeface("helvetica"), 20)
                        if self.tempo_control:
                            canvas.drawSimpleText(f"BPM: {self.push.counter.tempo:5.1f}", 10, 150, font, paint)
                            canvas.drawSimpleText(f"Quantum: {self.push.counter.quantum}", 130, 150, font, paint)
                        for number, state in self.encoders.items():
                            canvas.save()
                            canvas.translate(120 * number, 0)
                            paint.setStyle(skia.Paint.kStroke_Style)
                            if state.touched:
                                paint.setColor4f(skia.Color4f(state.r, state.g, state.b, 1))
                            else:
                                paint.setColor4f(skia.Color4f(state.r/2, state.g/2, state.b/2, 1))
                            path = skia.Path()
                            paint.setStrokeWidth(2)
                            path.addArc(skia.Rect.MakeXYWH(20, 40, 80, 80), -240, 300)
                            canvas.drawPath(path, paint)
                            start = 300 * (state.origin - state.lower) / (state.upper - state.lower)
                            end = 300 * (state.value - state.lower) / (state.upper - state.lower)
                            canvas.save()
                            canvas.translate(60, 80)
                            canvas.rotate(-240 + start)
                            path = skia.Path()
                            path.moveTo(26, 0)
                            path.lineTo(44, 0)
                            canvas.drawPath(path, paint)
                            canvas.restore()
                            path = skia.Path()
                            paint.setStrokeWidth(12)
                            if end > start:
                                path.addArc(skia.Rect.MakeXYWH(26, 46, 68, 68), -240 + start, end - start)
                            else:
                                path.addArc(skia.Rect.MakeXYWH(26, 46, 68, 68), -240 + end, start - end)
                            canvas.drawPath(path, paint)
                            path = skia.Path()
                            path.addRect(2, 2, 116, 26)
                            paint.setStyle(skia.Paint.kFill_Style)
                            canvas.drawPath(path, paint)
                            font.setSize(14)
                            exponent = 10**state.decimals
                            value = int((state.value * 100 if state.percent else state.value) * exponent) / exponent
                            text = f'{{:.{max(0, int(math.ceil(state.decimals)))}f}}'.format(value)
                            if state.percent:
                                text += '%'
                            width = font.measureText(text)
                            canvas.drawString(text, (120-width) / 2, 84, font, paint)
                            font.setSize(16)
                            text = state.name
                            width = font.measureText(text)
                            paint.setColor(skia.ColorBLACK)
                            canvas.drawString(text, (120-width) / 2, 20, font, paint)
                            canvas.restore()
                else:
                    now = self.push.counter.clock()
                    if (self.last_hello is None or now > self.last_hello + self.HELLO_RETRY_INTERVAL) \
                            and (self.last_received is None or now > self.last_received + self.RECEIVE_TIMEOUT):
                        await self.osc_sender.send_message('/hello')
                        self.last_hello = now
                    if self.last_received is not None and now > self.last_received + self.RESET_TIMEOUT:
                        self.reset()
        finally:
            for n in range(64):
                self.push.set_pad_rgb(n, 0, 0, 0)
            for n in range(16):
                self.push.set_menu_button_rgb(n, 0, 0, 0)
            for n in BUTTONS:
                self.push.set_button_white(n, 0)


parser = argparse.ArgumentParser(description="Flight Server")
parser.add_argument('--debug', action='store_true', default=False, help="Debug logging")
parser.add_argument('--verbose', action='store_true', default=False, help="Informational logging")
parser.add_argument('--notempo', action='store_true', default=False, help="Disable tempo control")
args = parser.parse_args()
logging.basicConfig(level=logging.DEBUG if args.debug else (logging.INFO if args.verbose else logging.WARNING), stream=sys.stderr)

controller = Controller(tempo_control=not args.notempo)
asyncio.run(controller.run())
