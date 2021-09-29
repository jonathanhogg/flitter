"""
Ableton Push API
"""

# pylama:ignore=C0103,R0911,R0912,R0903,R0902,W0401,R0913,W0703

import asyncio
from contextlib import contextmanager
import logging

import cairo
import numpy as np
import rtmidi
import usb.core

from ..clock import BeatCounter
from .constants import (MIDI, Command, Animation, Control, Note, Encoder, TouchStripFlags,
                        BUTTONS, COLOR_BUTTONS, ENCODER_CONTROLS, MENU_CONTROLS, MENU_NUMBERS)
from .events import (PadPressed, PadHeld, PadReleased, ButtonPressed, ButtonReleased,
                     EncoderTouched, EncoderTurned, EncoderReleased, TouchStripTouched,
                     TouchStripDragged, TouchStripReleased, MenuButtonPressed, MenuButtonReleased)


Log = logging.getLogger(__name__)


def color_to_index(r, g, b):
    r, g, b = (int(round(min(max(0, c), 255) / 63.75)) for c in (r, g, b))
    return ((r * 5) + g) * 5 + b


def index_to_color(i):
    i = min(max(0, int(i)), 124)
    return int((i // 25) * 63.75), int(((i // 5) % 5) * 63.75), int((i % 5) * 63.75)


def white_to_index(w):
    return int(round(min(max(0, w), 255) / 2.008))


def index_to_white(i):
    i = min(max(0, int(i)), 127)
    return (i << 1) + (i >> 6)


class Push:
    PORT_NAME = 'Ableton Push 2 User Port'
    SCREEN_HEIGHT = 160
    SCREEN_WIDTH = 960
    SCREEN_STRIDE = 1024
    SYSEX_ID = [0x00, 0x21, 0x1D, 0x01, 0x01]
    FRAME_HEADER = bytes([0xFF, 0xCC, 0xAA, 0x88, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])

    def __init__(self):
        self._screen_data = bytearray(self.SCREEN_HEIGHT * self.SCREEN_STRIDE * 2)
        self._screen_array = np.ndarray(buffer=self._screen_data, shape=(self.SCREEN_HEIGHT, self.SCREEN_STRIDE), dtype='uint16')
        self._screen_image = cairo.ImageSurface(cairo.Format.ARGB32, self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
        self._image_array = np.ndarray(buffer=self._screen_image.get_data(), shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 4), dtype='uint8')
        self._screen_ready = asyncio.Event()
        self._usb_device = usb.core.find(idVendor=0x2982, idProduct=0x1967)
        if self._usb_device is None:
            raise RuntimeError("Cannot locate USB device")
        self._usb_device.set_configuration()
        self._screen_endpoint = self._usb_device.get_active_configuration()[0, 0][0]
        self._midi_out = rtmidi.MidiOut()
        for i, name in enumerate(self._midi_out.get_ports()):
            if name == self.PORT_NAME:
                self._midi_out.open_port(i)
                break
        else:
            raise RuntimeError("Cannot open MIDI output port: " + self.PORT_NAME)
        self._midi_in = rtmidi.MidiIn()
        for i, name in enumerate(self._midi_in.get_ports()):
            if name == self.PORT_NAME:
                self._midi_in.open_port(i)
                break
        else:
            raise RuntimeError("Cannot open MIDI input port: " + self.PORT_NAME)
        self._loop = None
        self._receive_queue = asyncio.Queue()
        self._last_receive_timestamp = None
        self._counter = BeatCounter()
        self._clock_task = None
        self._screen_task = None

    @property
    def counter(self):
        return self._counter

    def start(self):
        self._loop = asyncio.get_event_loop()
        self._midi_in.set_callback(self._receive_callback)
        self._send_midi([MIDI.STOP])
        self._send_sysex(Command.SET_MIDI_MODE, 1)
        self._send_sysex(Command.SET_AFTERTOUCH_MODE, 1)
        self._send_sysex(Command.SET_TOUCHSTRIP_CONFIG, TouchStripFlags.HOST_SENDS | TouchStripFlags.LED_POINT)
        for i in range(128):
            r, g, b = index_to_color(i)
            w = index_to_white(i)
            self._send_sysex(Command.SET_COLOR_PALETTE_ENTRY, i, r & 0x7f, r >> 7, g & 0x7f, g >> 7, b & 0x7f, b >> 1, w & 0x7f, w >> 7)
        self._send_sysex(Command.REAPPLY_COLOR_PALETTE)
        self._clock_task = asyncio.create_task(self._run_clock())
        self._screen_task = asyncio.create_task(self._run_screen())

    def _send_midi(self, message):
        Log.debug("Send MIDI - %s", " ".join(f"{b:02x}" for b in message))
        self._midi_out.send_message(message)

    def _receive_callback(self, message_delta, _):
        now = self.counter.clock()
        message, delta = message_delta
        timestamp = now if self._last_receive_timestamp is None else min(now, self._last_receive_timestamp + delta)
        self._last_receive_timestamp = timestamp
        Log.debug("Received @ %.2f MIDI - %s", timestamp, " ".join(f"{b:02x}" for b in message))
        self._loop.call_soon_threadsafe(self._receive_queue.put_nowait, (message, timestamp))

    def _send_sysex(self, cmd: Command, *args):
        self._send_midi([MIDI.START_OF_SYSEX] + self.SYSEX_ID + [cmd] + list(args) + [MIDI.END_OF_SYSEX])

    def set_pad_color(self, number, r, g, b, animation=Animation.NONE):
        assert number < 64
        self._send_midi([MIDI.NOTE_ON + animation, number + Note.PAD_0_0, color_to_index(r, g, b)])

    def set_button_white(self, number, w, animation=Animation.NONE):
        assert number in BUTTONS
        if number in COLOR_BUTTONS:
            self._send_midi([MIDI.CONTROL_CHANGE + animation, number, color_to_index(w, w, w)])
        else:
            self._send_midi([MIDI.CONTROL_CHANGE + animation, number, white_to_index(w)])

    def set_button_color(self, number, r, g, b, animation=Animation.NONE):
        assert number in COLOR_BUTTONS
        self._send_midi([MIDI.CONTROL_CHANGE + animation, number, color_to_index(r, g, b)])

    def set_menu_button_color(self, number, r, g, b, animation=Animation.NONE):
        assert number in range(16)
        self._send_midi([MIDI.CONTROL_CHANGE + animation, MENU_NUMBERS[number], color_to_index(r, g, b)])

    def set_touch_strip_position(self, position):
        value = int(min(max(0, position), 1) * 255) << 6
        self._send_midi([MIDI.PITCH_BEND_CHANGE, value & 0x7f, value >> 7])

    def set_led_brightness(self, brightness):
        value = int(round(min(max(0, brightness), 1) * 127))
        self._send_sysex(Command.SET_LED_BRIGHTNESS, value)

    def set_display_brightness(self, brightness):
        value = int(round(min(max(0, brightness), 1) * 255))
        self._send_sysex(Command.SET_DISPLAY_BRIGHTNESS, value & 0x7f, value >> 7)

    async def get_event(self, timeout=None):
        try:
            message, timestamp = await asyncio.wait_for(self._receive_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
        if message[0] == MIDI.NOTE_ON:
            if Note.PAD_0_0 <= message[1] <= Note.PAD_7_7:
                number = message[1] - Note.PAD_0_0
                column, row = number % 8, number // 8
                return PadPressed(timestamp=timestamp, number=number, column=column, row=row, pressure=message[2]/127)
            if message[1] in set(Encoder):
                if message[2] == 0x7f:
                    return EncoderTouched(timestamp=timestamp, number=Encoder(message[1]))
                if message[2] == 0x00:
                    return EncoderReleased(timestamp=timestamp, number=Encoder(message[1]))
            if message[1] == Note.TOUCH_STRIP:
                if message[2] == 0x7f:
                    return TouchStripTouched(timestamp=timestamp)
                if message[2] == 0x00:
                    return TouchStripReleased(timestamp=timestamp)
        elif message[0] == MIDI.POLYPHONIC_PRESSURE:
            if Note.PAD_0_0 <= message[1] <= Note.PAD_7_7:
                number = message[1] - Note.PAD_0_0
                column, row = number % 8, number // 8
                return PadHeld(timestamp=timestamp, number=number, column=column, row=row, pressure=message[2]/127)
        elif message[0] == MIDI.NOTE_OFF:
            if Note.PAD_0_0 <= message[1] <= Note.PAD_7_7:
                number = message[1] - Note.PAD_0_0
                column, row = number % 8, number // 8
                return PadReleased(timestamp=timestamp, number=number, column=column, row=row)
        elif message[0] == MIDI.CONTROL_CHANGE:
            if message[1] in BUTTONS:
                if message[2] == 0x7f:
                    return ButtonPressed(timestamp=timestamp, number=Control(message[1]))
                if message[2] == 0x00:
                    return ButtonReleased(timestamp=timestamp, number=Control(message[1]))
            if message[1] in ENCODER_CONTROLS:
                amount = message[2] if message[2] < 64 else message[2] - 128
                return EncoderTurned(timestamp=timestamp, number=ENCODER_CONTROLS[message[1]], amount=amount)
            if message[1] in MENU_CONTROLS:
                number = MENU_CONTROLS[message[1]]
                column, row = number % 8, number // 8
                if message[2] == 0x7f:
                    return MenuButtonPressed(timestamp=timestamp, number=number, column=column, row=row)
                if message[2] == 0x00:
                    return MenuButtonReleased(timestamp=timestamp, number=number, column=column, row=row)
        elif message[0] == MIDI.PITCH_BEND_CHANGE:
            position = ((message[2] << 7) + message[1]) / (1 << 14)
            return TouchStripDragged(timestamp=timestamp, position=position)
        Log.warning("Unrecognised MIDI - %s", " ".join(f"{b:02x}" for b in message))

    @contextmanager
    def screen_context(self):
        yield cairo.Context(self._screen_image)
        self._screen_image.flush()
        self._screen_ready.set()

    async def _run_clock(self):
        self._send_midi([MIDI.START])
        tick = int(self._counter.beat * 24)
        try:
            while True:
                await self._counter.wait_for_beat((tick + 1) / 24)
                tick = int(self._counter.beat * 24)
                if int(self._counter.phase * 24) == 0:
                    self._send_midi([MIDI.START])
                else:
                    self._send_midi([MIDI.CLOCK])
        except Exception:
            Log.exception("Unexpected exception")

    async def _run_screen(self):
        try:
            while True:
                self._screen_array[:, :] = 0
                pixels = self._screen_array[:, :self._image_array.shape[1]]
                pixels += self._image_array[:, :, 0] >> 3
                pixels <<= 6
                pixels += self._image_array[:, :, 1] >> 2
                pixels <<= 5
                pixels += self._image_array[:, :, 2] >> 3
                self._screen_array[:, 0::2] ^= 0xF3E7
                self._screen_array[:, 1::2] ^= 0xFFE7
                self._screen_endpoint.write(self.FRAME_HEADER)
                self._screen_endpoint.write(self._screen_data)
                self._screen_ready.clear()
                await asyncio.sleep(1/60)
                try:
                    await asyncio.wait_for(self._screen_ready.wait(), timeout=59/60)
                except asyncio.TimeoutError:
                    pass
        except Exception:
            Log.exception("Unexpected exception")
