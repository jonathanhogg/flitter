"""
Ableton Push OSC controller for Flitter
"""

# pylama:ignore=W0601,C0103,R0912,R0915,R0914,R0902,C901

import argparse
import asyncio
from dataclasses import dataclass
import math
import sys

from loguru import logger
import skia

import flitter
from ..clock import TapTempo
from ..ableton.constants import Encoder, Control, BUTTONS
from ..ableton.events import (ButtonPressed, ButtonReleased, PadEvent, PadPressed, PadHeld, PadReleased,
                              EncoderTurned, EncoderTouched, EncoderReleased, MenuButtonReleased)
from ..ableton.push import Push2
from ..ableton.palette import HuePalette
from .osc import OSCSender, OSCReceiver, OSCBundle


@dataclass
class PadState:
    name: str
    r: float
    g: float
    b: float
    quantize: float
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

    def __init__(self, tempo_control=True, fader_control=True):
        self.tempo_control = tempo_control
        self.fader_control = fader_control
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
        self.recording = False
        self.record_buffer = []
        self.playing = False
        self.playback_position = None

    async def process_message(self, message):
        if isinstance(message, OSCBundle):
            for element in message.elements:
                await self.process_message(element)
            return
        logger.debug("Received OSC message: {!r}", message)
        match message.address.strip('/').split('/'):
            case ['tempo']:
                tempo, quantum, start = message.args
                self.push.counter.update(tempo, quantum, start)
            case ['pad', column, row, 'state']:
                column, row = int(column), int(row)
                if 0 <= column < 8 and 0 <= row < 8 and message.args:
                    state = PadState(*message.args)
                    brightness = 1 if state.touched or state.toggled else 0.5
                    self.push.set_pad_rgb((7 - row) * 8 + column, state.r*brightness, state.g*brightness, state.b*brightness)
                    self.pads[column, row] = state
                    if state.touched and (column, row) not in self.touched_pads:
                        await self.osc_sender.send_message(f'/pad/{column}/{row}/released', self.push.counter.clock())
                elif (column, row) in self.pads:
                    self.push.set_pad_rgb((7 - row) * 8 + column, 0, 0, 0)
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
            self.push.set_pad_rgb((7-row) * 8 + column, 0, 0, 0)
        self.pads.clear()
        for number in self.encoders:
            self.push.set_menu_button_rgb(number + 8, 0, 0, 0)
        self.encoders.clear()
        for control in self.buttons:
            self.push.set_button_white(control, 0)
        self.buttons.clear()
        self.push.counter.update(120, 4, self.push.counter.clock())
        self.last_received = None
        self.recording = False
        self.playing = False
        self.record_buffer = []
        self.updated.set()

    async def receive_messages(self):
        while True:
            message = await self.osc_receiver.receive()
            self.last_received = self.push.counter.clock()
            await self.process_message(message)
            self.updated.set()

    def record_event(self, event):
        beat = self.push.counter.beat_at_time(event.timestamp)
        quantum = self.push.counter.quantum
        phase = beat % quantum
        event.timestamp = self.push.counter.time_at_beat(beat + quantum)
        self.record_buffer.append((phase, event))

    async def run(self):
        logger.info("Starting Ableton Push 2 interface")
        self.push = Push2(palette=HuePalette())
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
        record_pressed_at = None
        receive_task = asyncio.create_task(self.receive_messages())
        next_playback_event = None
        playback_release_pads = set()
        pad_shifts = {}
        self.updated.set()
        try:
            wait_event = asyncio.create_task(self.push.get_event())
            wait_update = asyncio.create_task(self.updated.wait())
            wait_beat = asyncio.create_task(self.push.counter.wait_for_beat(self.push.counter.beat*2//1/2 + 0.5))
            while True:
                if self.playing and next_playback_event is None:
                    _, event = self.record_buffer[0]
                    beat = self.push.counter.beat_at_time(event.timestamp)
                    next_playback_event = asyncio.create_task(self.push.counter.wait_for_beat(beat))
                    logger.debug("Next playback event: {!r}", event)
                events = {wait_event, wait_update, wait_beat}
                if next_playback_event is not None:
                    events.add(next_playback_event)
                done, _ = await asyncio.wait(events, timeout=1/10, return_when=asyncio.FIRST_COMPLETED)
                if wait_event in done or next_playback_event in done:
                    synthetic = False
                    if wait_event in done:
                        event = wait_event.result()
                        wait_event = asyncio.create_task(self.push.get_event())
                    elif next_playback_event in done:
                        next_playback_event = None
                        if not self.playing or not self.record_buffer:
                            continue
                        phase, event = self.record_buffer.pop(0)
                        self.record_buffer.append((phase, event))
                        synthetic = True
                    match event:
                        case ButtonPressed(number=Control.SHIFT):
                            shift_pressed = True
                        case ButtonReleased(number=Control.SHIFT):
                            shift_pressed = False
                        case PadPressed():
                            if synthetic or not tap_tempo_pressed:
                                row = 7 - event.row
                                quantize = None
                                if not synthetic:
                                    pad_state = self.pads.get((row, event.column))
                                    if pad_state and pad_state.quantize:
                                        beat = self.push.counter.beat_at_time(event.timestamp)
                                        beat = round(beat * pad_state.quantize) / pad_state.quantize
                                        timestamp = self.push.counter.time_at_beat(beat)
                                        pad_shifts[(row, event.column)] = timestamp - event.timestamp
                                        event.timestamp = timestamp
                                self.touched_pads.add((event.column, row))
                                address = f'/pad/{event.column}/{row}/touched'
                                await self.osc_sender.send_message(address, event.timestamp, event.pressure)
                                if synthetic:
                                    playback_release_pads.add((event.column, row))
                                elif self.recording:
                                    self.record_event(event)
                            else:
                                tap_tempo.tap(event.timestamp)
                        case PadHeld():
                            if synthetic or not tap_tempo_pressed:
                                row = 7 - event.row
                                event.timestamp += pad_shifts.get((row, event.column), 0)
                                address = f'/pad/{event.column}/{row}/held'
                                await self.osc_sender.send_message(address, event.timestamp, event.pressure)
                                if not synthetic and self.recording:
                                    self.record_event(event)
                        case PadReleased():
                            if synthetic or not tap_tempo_pressed:
                                row = 7 - event.row
                                self.touched_pads.discard((event.column, row))
                                if (row, event.column) in pad_shifts:
                                    event.timestamp += pad_shifts.pop((row, event.column))
                                address = f'/pad/{event.column}/{row}/released'
                                await self.osc_sender.send_message(address, event.timestamp)
                                if synthetic:
                                    playback_release_pads.discard((event.column, row))
                                elif self.recording:
                                    self.record_event(event)
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
                        case EncoderTurned(number=Encoder.MASTER) if self.fader_control:
                            brightness = min(max(0, brightness + event.amount / 200), 1)
                            self.push.set_led_brightness(brightness)
                            self.push.set_display_brightness(brightness)
                        case MenuButtonReleased() if event.row == 1:
                            await self.osc_sender.send_message(f'/encoder/{event.column}/reset', event.timestamp)
                        case ButtonReleased(number=Control.PAGE_LEFT):
                            await self.osc_sender.send_message('/page_left')
                        case ButtonReleased(number=Control.PAGE_RIGHT):
                            await self.osc_sender.send_message('/page_right')
                        case ButtonPressed(number=Control.RECORD):
                            record_pressed_at = event.timestamp
                        case ButtonReleased(number=Control.RECORD):
                            if self.recording:
                                logger.info("Stop recording")
                                self.recording = False
                            else:
                                logger.info("Start recording")
                                if event.timestamp - record_pressed_at > 0.5:
                                    self.push.counter.set_phase(0, event.timestamp, backslip_limit=1)
                                    self.record_buffer = []
                                    await self.osc_sender.send_message('/tempo', self.push.counter.tempo, self.push.counter.quantum, self.push.counter.start)
                                    if self.playing:
                                        logger.info("Stop playback")
                                        self.playing = False
                                        next_playback_event = None
                                        while playback_release_pads:
                                            column, row = playback_release_pads.pop()
                                            address = f'/pad/{column}/{row}/released'
                                            await self.osc_sender.send_message(address, event.timestamp)
                                self.recording = True
                            record_pressed_at = None
                        case ButtonReleased(number=Control.PLAY):
                            if self.playing:
                                logger.info("Stop playback")
                                self.playing = False
                                next_playback_event = None
                                while playback_release_pads:
                                    column, row = playback_release_pads.pop()
                                    address = f'/pad/{column}/{row}/released'
                                    await self.osc_sender.send_message(address, event.timestamp)
                            elif self.record_buffer:
                                logger.info("Start playback")
                                beat = self.push.counter.beat_at_time(event.timestamp)
                                quantum = self.push.counter.quantum
                                phase = beat % quantum
                                for event_phase, event in self.record_buffer:
                                    if event_phase < phase:
                                        event_phase += quantum
                                    event.timestamp = self.push.counter.time_at_beat(beat - phase + event_phase)
                                self.record_buffer.sort(key=lambda pe: pe[1].timestamp)
                                self.playing = True
                    if synthetic:
                        beat = self.push.counter.beat_at_time(event.timestamp)
                        event.timestamp = self.push.counter.time_at_beat(beat + self.push.counter.quantum)
                    self.updated.set()
                elif wait_update in done or wait_beat in done:
                    self.updated.clear()
                    if wait_update in done:
                        wait_update = asyncio.create_task(self.updated.wait())
                    if wait_beat in done:
                        wait_beat = asyncio.create_task(self.push.counter.wait_for_beat(self.push.counter.beat*2//1/2 + 0.5))
                    async with self.push.screen_context() as ctx:
                        ctx.clear(skia.ColorBLACK)
                        paint = skia.Paint(Color=skia.ColorWHITE, AntiAlias=True)
                        red_paint = skia.Paint(Color=skia.ColorRED, AntiAlias=True)
                        font = skia.Font(skia.Typeface("helvetica"), 18)
                        if self.tempo_control:
                            ctx.drawSimpleText(f"BPM: {self.push.counter.tempo:5.1f}", 10, 150, font, paint)
                            ctx.drawSimpleText(f"Quantum: {self.push.counter.quantum}", 130, 150, font, paint)
                            if record_pressed_at is not None and self.push.counter.clock() > record_pressed_at + 0.5:
                                ctx.drawSimpleText(f"Phase: {int(self.push.counter.phase):2d}", 250, 150, font, red_paint)
                            else:
                                ctx.drawSimpleText(f"Phase: {int(self.push.counter.phase):2d}", 250, 150, font, paint)
                        for number, state in self.encoders.items():
                            ctx.save()
                            ctx.translate(120 * number, 0)
                            paint.setStyle(skia.Paint.kStroke_Style)
                            if state.touched:
                                paint.setColor4f(skia.Color4f(state.r, state.g, state.b, 1))
                            else:
                                paint.setColor4f(skia.Color4f(state.r/2, state.g/2, state.b/2, 1))
                            path = skia.Path()
                            paint.setStrokeWidth(2)
                            path.addArc(skia.Rect.MakeXYWH(20, 40, 80, 80), -240, 300)
                            ctx.drawPath(path, paint)
                            start = 300 * (state.origin - state.lower) / (state.upper - state.lower)
                            end = 300 * (state.value - state.lower) / (state.upper - state.lower)
                            ctx.save()
                            ctx.translate(60, 80)
                            ctx.rotate(-240 + start)
                            path = skia.Path()
                            path.moveTo(26, 0)
                            path.lineTo(44, 0)
                            ctx.drawPath(path, paint)
                            ctx.restore()
                            path = skia.Path()
                            paint.setStrokeWidth(12)
                            if end > start:
                                path.addArc(skia.Rect.MakeXYWH(26, 46, 68, 68), -240 + start, end - start)
                            else:
                                path.addArc(skia.Rect.MakeXYWH(26, 46, 68, 68), -240 + end, start - end)
                            ctx.drawPath(path, paint)
                            path = skia.Path()
                            path.addRect(2, 2, 116, 26)
                            paint.setStyle(skia.Paint.kFill_Style)
                            ctx.drawPath(path, paint)
                            font.setSize(14)
                            exponent = 10**state.decimals
                            value = int((state.value * 100 if state.percent else state.value) * exponent) / exponent
                            text = f'{{:.{max(0, int(math.ceil(state.decimals)))}f}}'.format(value)
                            if state.percent:
                                text += '%'
                            width = font.measureText(text)
                            ctx.drawString(text, (120-width) / 2, 84, font, paint)
                            font.setSize(16)
                            text = state.name
                            width = font.measureText(text)
                            paint.setColor(skia.ColorBLACK)
                            ctx.drawString(text, (120-width) / 2, 20, font, paint)
                            ctx.restore()
                else:
                    now = self.push.counter.clock()
                    if (self.last_hello is None or now > self.last_hello + self.HELLO_RETRY_INTERVAL) \
                            and (self.last_received is None or now > self.last_received + self.RECEIVE_TIMEOUT):
                        await self.osc_sender.send_message('/hello')
                        self.last_hello = now
                    if self.last_received is not None and now > self.last_received + self.RESET_TIMEOUT:
                        self.reset()
                flash = 1 if self.push.counter.beat % 1 < 0.5 else 0.5
                if self.tempo_control:
                    self.push.set_button_white(Control.TAP_TEMPO, flash)
                if self.recording:
                    self.push.set_button_rgb(Control.RECORD, flash, 0, 0)
                else:
                    self.push.set_button_rgb(Control.RECORD, 0.5, 0, 0)
                if self.playing:
                    self.push.set_button_rgb(Control.PLAY, 0, flash, 0)
                elif self.record_buffer:
                    self.push.set_button_rgb(Control.PLAY, 0, 0.5, 0)
                else:
                    self.push.set_button_rgb(Control.PLAY, 0, 0, 0)
        finally:
            for n in range(64):
                self.push.set_pad_rgb(n, 0, 0, 0)
            for n in range(16):
                self.push.set_menu_button_rgb(n, 0, 0, 0)
            for n in BUTTONS:
                self.push.set_button_white(n, 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Flitter Ableton Push 2 Interface")
    parser.set_defaults(level=None)
    levels = parser.add_mutually_exclusive_group()
    levels.add_argument('--trace', action='store_const', const='TRACE', dest='level', help="Trace logging")
    levels.add_argument('--debug', action='store_const', const='DEBUG', dest='level', help="Debug logging")
    levels.add_argument('--verbose', action='store_const', const='INFO', dest='level', help="Informational logging")
    parser.add_argument('--notempo', action='store_true', default=False, help="Disable tempo control")
    parser.add_argument('--nofader', action='store_true', default=False, help="Disable fader control")
    args = parser.parse_args()
    flitter.configure_logger(args.level)

    try:
        controller = Controller(tempo_control=not args.notempo, fader_control=not args.nofader)
        asyncio.run(controller.run())
    except KeyboardInterrupt:
        logger.info("Exiting Push 2 interface on keyboard interrupt")
    except Exception:
        logger.exception("Unhandled exception in Push 2 interface")
