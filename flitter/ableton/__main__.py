"""
Ableton Push OSC controller for Flitter
"""

# pylama:ignore=W0601,C0103,R0912,R0915,R0914,R0902

import argparse
import asyncio
import logging
import sys

import skia

from ..clock import TapTempo
from .constants import Encoder, Control, BUTTONS
from .events import (ButtonPressed, ButtonReleased, PadPressed, PadHeld, PadReleased,
                     EncoderTurned, EncoderTouched, EncoderReleased, MenuButtonReleased)
from .push import Push
from ..interface.osc import OSCSender, OSCReceiver, OSCBundle


Log = logging.getLogger(__name__)


class Controller:
    HELLO_RETRY_INTERVAL = 10
    RECEIVE_TIMEOUT = 5
    RESET_TIMEOUT = 30

    def __init__(self):
        self.push = None
        self.osc_sender = OSCSender('localhost', 47178)
        self.osc_receiver = OSCReceiver('localhost', 47177)
        self.pads = {}
        self.encoders = {}
        self.buttons = {}
        self.last_received = None
        self.last_hello = None

    def process_message(self, message):
        if isinstance(message, OSCBundle):
            for element in message.elements:
                self.process_message(element)
            return
        Log.info("Received OSC message: %r", message)
        parts = message.address.strip('/').split('/')
        if parts == ['tempo']:
            tempo, quantum, start = message.args
            self.push.counter.update(tempo, quantum, start)
        elif parts[0] == 'pad' and parts[-1] == 'state':
            column, row = map(int, parts[1:-1])
            if message.args:
                name, r, g, b, touched, toggled = message.args
                brightness = 255 if touched or toggled else 63
                self.push.set_pad_color(row * 8 + column, int(r*brightness), int(g*brightness), int(b*brightness))
                self.pads[column, row] = name, (r, g, b), touched, toggled
            elif (column, row) in self.pads:
                self.push.set_pad_color(row * 8 + column, 0, 0, 0)
                del self.pads[column, row]
        elif parts[0] == 'encoder' and parts[2] == 'state':
            number = int(parts[1])
            if message.args:
                name, r, g, b, touched, value, lower, upper = message.args
                brightness = 255 if touched else 63
                self.push.set_menu_button_color(number + 8, int(r*brightness), int(g*brightness), int(b*brightness))
                self.encoders[number] = name, (r, g, b), touched, value, lower, upper
            elif number in self.encoders:
                self.push.set_menu_button_color(number + 8, 0, 0, 0)
                del self.encoders[number]
        elif parts[0] == 'page_left':
            enabled, = message.args
            self.push.set_button_white(Control.PAGE_LEFT, 255 if enabled else 0)
            if enabled:
                if Control.PAGE_LEFT not in self.buttons:
                    self.buttons[Control.PAGE_LEFT] = 255
            elif Control.PAGE_LEFT in self.buttons:
                del self.buttons[Control.PAGE_LEFT]
        elif parts[0] == 'page_right':
            enabled, = message.args
            self.push.set_button_white(Control.PAGE_RIGHT, 255 if enabled else 0)
            if enabled:
                if Control.PAGE_LEFT not in self.buttons:
                    self.buttons[Control.PAGE_RIGHT] = 255
            elif Control.PAGE_LEFT in self.buttons:
                del self.buttons[Control.PAGE_RIGHT]

    def reset(self):
        for column, row in self.pads:
            self.push.set_pad_color(row * 8 + column, 0, 0, 0)
        self.pads.clear()
        for number in self.encoders:
            self.push.set_menu_button_color(number + 8, 0, 0, 0)
        self.encoders.clear()
        for control in self.buttons:
            self.push.set_button_white(control, 0)
        self.buttons.clear()
        self.last_received = None

    async def receive_messages(self):
        while True:
            message = await self.osc_receiver.receive()
            self.last_received = self.push.counter.clock()
            self.process_message(message)

    async def run(self):
        self.push = Push()
        self.push.start()
        for n in range(64):
            self.push.set_pad_color(n, 0, 0, 0)
        for n in range(16):
            self.push.set_menu_button_color(n, 0, 0, 0)
        for n in BUTTONS:
            self.push.set_button_white(n, 0)
        self.push.set_button_white(Control.TAP_TEMPO, 255)
        self.push.set_button_white(Control.SHIFT, 255)
        brightness = 1
        self.push.set_led_brightness(brightness)
        self.push.set_display_brightness(brightness)
        self.push.set_touch_strip_position(0)
        shift_pressed = False
        tap_tempo_pressed = False
        tap_tempo = TapTempo(rounding=1)
        asyncio.get_event_loop().create_task(self.receive_messages())
        try:
            while True:
                event = await self.push.get_event(1/60)
                if event:
                    if isinstance(event, ButtonPressed) and event.number == Control.SHIFT:
                        shift_pressed = True
                    elif isinstance(event, ButtonReleased) and event.number == Control.SHIFT:
                        shift_pressed = False
                    elif isinstance(event, PadPressed):
                        if not tap_tempo_pressed:
                            address = f'/pad/{event.column}/{event.row}/touched'
                            await self.osc_sender.send_message(address, event.timestamp, event.pressure)
                        else:
                            tap_tempo.tap(event.timestamp)
                    elif isinstance(event, PadHeld):
                        if not tap_tempo_pressed:
                            address = f'/pad/{event.column}/{event.row}/held'
                            await self.osc_sender.send_message(address, event.timestamp, event.pressure)
                    elif isinstance(event, PadReleased):
                        if not tap_tempo_pressed:
                            address = f'/pad/{event.column}/{event.row}/released'
                            await self.osc_sender.send_message(address, event.timestamp)
                    elif isinstance(event, EncoderTouched) and event.number < 8:
                        address = f'/encoder/{event.number}/touched'
                        await self.osc_sender.send_message(address, event.timestamp)
                    elif isinstance(event, EncoderTurned) and event.number < 8:
                        address = f'/encoder/{event.number}/turned'
                        await self.osc_sender.send_message(address, event.timestamp, event.amount / 400)
                    elif isinstance(event, EncoderReleased) and event.number < 8:
                        address = f'/encoder/{event.number}/released'
                        await self.osc_sender.send_message(address, event.timestamp)
                    elif isinstance(event, EncoderTurned) and event.number == Encoder.TEMPO:
                        if shift_pressed:
                            self.push.counter.quantum = max(2, self.push.counter.quantum + event.amount)
                        else:
                            self.push.counter.set_tempo((round(self.push.counter.tempo * 2) + event.amount) / 2, timestamp=event.timestamp)
                        await self.osc_sender.send_message('/tempo', self.push.counter.tempo, self.push.counter.quantum, self.push.counter.start)
                    elif isinstance(event, ButtonPressed) and event.number == Control.TAP_TEMPO:
                        tap_tempo_pressed = True
                    elif isinstance(event, ButtonReleased) and event.number == Control.TAP_TEMPO:
                        tap_tempo.apply(self.push.counter, event.timestamp, backslip_limit=1)
                        tap_tempo_pressed = False
                        await self.osc_sender.send_message('/tempo', self.push.counter.tempo, self.push.counter.quantum, self.push.counter.start)
                    elif isinstance(event, EncoderTurned) and event.number == Encoder.MASTER:
                        brightness = min(max(0, brightness + event.amount / 200), 1)
                        self.push.set_led_brightness(brightness)
                        self.push.set_display_brightness(brightness)
                    elif isinstance(event, MenuButtonReleased) and event.row == 1:
                        await self.osc_sender.send_message(f'/encoder/{event.column}/reset', event.timestamp)
                    elif isinstance(event, ButtonReleased) and event.number == Control.PAGE_LEFT:
                        await self.osc_sender.send_message('/page_left')
                    elif isinstance(event, ButtonReleased) and event.number == Control.PAGE_RIGHT:
                        await self.osc_sender.send_message('/page_right')
                else:
                    now = self.push.counter.clock()
                    if (self.last_hello is None or now > self.last_hello + self.HELLO_RETRY_INTERVAL) \
                            and (self.last_received is None or now > self.last_received + self.RECEIVE_TIMEOUT):
                        await self.osc_sender.send_message('/hello')
                        self.last_hello = now
                    if self.last_received is not None and now > self.last_received + self.RESET_TIMEOUT:
                        self.reset()
                    async with self.push.screen_canvas() as canvas:
                        canvas.clear(skia.ColorBLACK)
                        paint = skia.Paint(Color=skia.ColorWHITE, AntiAlias=True)
                        font = skia.Font(skia.Typeface("helvetica"), 20)
                        canvas.drawSimpleText(f"BPM: {self.push.counter.tempo:5.1f}", 10, 150, font, paint)
                        canvas.drawSimpleText(f"Quantum: {self.push.counter.quantum}", 130, 150, font, paint)
                        font.setSize(16)
                        for number, (name, (r, g, b), touched, value, lower, upper) in self.encoders.items():
                            canvas.save()
                            canvas.translate(120 * number, 0)
                            paint.setStyle(skia.Paint.kStroke_Style)
                            if touched:
                                paint.setColor4f(skia.Color4f(r, g, b, 1))
                            else:
                                paint.setColor4f(skia.Color4f(r/2, g/2, b/2, 1))
                            path = skia.Path()
                            paint.setStrokeWidth(2)
                            path.addArc(skia.Rect.MakeXYWH(30, 50, 60, 60), -240, 300)
                            canvas.drawPath(path, paint)
                            path = skia.Path()
                            paint.setStrokeWidth(10)
                            sweep = 300 * (value - lower) / (upper - lower)
                            path.addArc(skia.Rect.MakeXYWH(30, 50, 60, 60), -240, sweep)
                            canvas.drawPath(path, paint)
                            path = skia.Path()
                            path.addRect(2, 2, 116, 26)
                            paint.setStyle(skia.Paint.kFill_Style)
                            canvas.drawPath(path, paint)
                            text = name.upper()
                            width = font.measureText(text)
                            paint.setColor(skia.ColorBLACK)
                            canvas.drawString(text, (120-width) / 2, 20, font, paint)
                            canvas.restore()
        finally:
            for n in range(64):
                self.push.set_pad_color(n, 0, 0, 0)
            for n in range(16):
                self.push.set_menu_button_color(n, 0, 0, 0)
            for n in BUTTONS:
                self.push.set_button_white(n, 0)


parser = argparse.ArgumentParser(description="Flight Server")
parser.add_argument('--debug', action='store_true', default=False, help="Debug logging")
parser.add_argument('--verbose', action='store_true', default=False, help="Informational logging")
args = parser.parse_args()
logging.basicConfig(level=logging.DEBUG if args.debug else (logging.INFO if args.verbose else logging.WARNING), stream=sys.stderr)

controller = Controller()
asyncio.run(controller.run())
