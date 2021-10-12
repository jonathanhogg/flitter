"""
Ableton Push API test
"""

# pylama:ignore=W0601,C0103,R0912,R0915

import argparse
import asyncio
import logging
import math
import sys

from ..clock import TapTempo
from .constants import Animation, Encoder, Control, BUTTONS
from .events import (ButtonPressed, ButtonReleased, TouchStripDragged, PadPressed, PadHeld, PadReleased,
                     EncoderTurned, EncoderTouched, EncoderReleased, MenuButtonReleased)
from .push import Push
from ..interface.osc import OSCSender, OSCReceiver, OSCBundle


Log = logging.getLogger(__name__)


class Controller:
    def __init__(self):
        self.push = None
        self.osc_sender = OSCSender('localhost', 47178)
        self.osc_receiver = OSCReceiver('localhost', 47177)
        self.encoders = {}

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
            else:
                self.push.set_pad_color(row * 8 + column, 0, 0, 0)
        elif parts[0] == 'encoder' and parts[2] == 'state':
            number = int(parts[1])
            if message.args:
                name, r, g, b, touched, value, lower, upper = message.args
                brightness = 255 if touched else 63
                self.push.set_menu_button_color(number + 8, int(r*brightness), int(g*brightness), int(b*brightness))
                self.encoders[number] = name, (r,g,b), touched, value, lower, upper
            elif number in self.encoders:
                self.push.set_menu_button_color(number + 8, 0, 0, 0)
                del self.encoders[number]
        elif parts[0] == 'page_left':
            enabled, = message.args
            self.push.set_button_white(Control.PAGE_LEFT, 255 if enabled else 0)
        elif parts[0] == 'page_right':
            enabled, = message.args
            self.push.set_button_white(Control.PAGE_RIGHT, 255 if enabled else 0)

    async def receive_messages(self):
        while True:
            message = await self.osc_receiver.receive()
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
            await self.osc_sender.send_message('/hello')
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
                    async with self.push.screen_context() as ctx:
                        ctx.set_source_rgb(0, 0, 0)
                        ctx.paint()
                        ctx.set_source_rgb(1, 1, 1)
                        ctx.set_font_size(20)
                        ctx.move_to(10, 150)
                        ctx.show_text(f"BPM: {self.push.counter.tempo:5.1f}")
                        ctx.move_to(130, 150)
                        ctx.show_text(f"Quantum: {self.push.counter.quantum}")
                        ctx.set_font_size(16)
                        for number in self.encoders:
                            ctx.save()
                            ctx.translate(120 * number, 0)
                            name, (r, g, b), touched, value, lower, upper = self.encoders[number]
                            if touched:
                                ctx.set_source_rgb(r, g, b)
                            else:
                                ctx.set_source_rgb(r/2, g/2, b/2)
                            ctx.new_path()
                            ctx.set_line_width(2)
                            ctx.arc(60, 80, 30, 2/3*math.pi, 7/3*math.pi)
                            ctx.stroke()
                            ctx.new_path()
                            ctx.set_line_width(10)
                            p = 2/3 + 5/3 * (value - lower) / (upper - lower)
                            ctx.arc(60, 80, 40, 2/3*math.pi, p*math.pi)
                            ctx.stroke()
                            ctx.new_path()
                            ctx.rectangle(2, 2, 116, 26)
                            ctx.fill()
                            name = name.upper()
                            extents = ctx.text_extents(name)
                            ctx.move_to((120 - extents.width) / 2, 20)
                            ctx.set_source_rgb(0, 0, 0)
                            ctx.show_text(name)
                            ctx.restore()
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
