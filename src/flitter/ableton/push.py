"""
Ableton Push 2 API

As per official hardware documentation at: https://github.com/Ableton/push-interface
"""

import asyncio
from contextlib import asynccontextmanager

from loguru import logger
import numpy as np
import rtmidi2
import skia
import usb.core

from ..clock import BeatCounter, system_clock
from .constants import (MIDI, Command, Animation, Control, Note, Encoder, TouchStripFlags,
                        BUTTONS, COLOR_BUTTONS, ENCODER_CONTROLS)
from .events import (PadPressed, PadHeld, PadReleased,
                     ButtonPressed, ButtonReleased,
                     EncoderTouched, EncoderTurned, EncoderReleased,
                     TouchStripTouched, TouchStripDragged, TouchStripReleased)
from .palette import SimplePalette


class Push2CommunicationError(Exception):
    pass


class Push2:
    USB_VENDOR = 0x2982
    USB_PRODUCT = 0x1967
    MIDI_PORT_NAME = 'Ableton Push 2 User Port'
    SCREEN_HEIGHT = 160
    SCREEN_WIDTH = 960
    SCREEN_STRIDE = 1024
    SYSEX_ID = [0x00, 0x21, 0x1D, 0x01, 0x01]
    FRAME_HEADER = bytes([0xFF, 0xCC, 0xAA, 0x88, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])

    @classmethod
    def test_presence(cls):
        return usb.core.find(idVendor=cls.USB_VENDOR, idProduct=cls.USB_PRODUCT) is not None

    def __init__(self, palette=None):
        self._palette = palette if palette is not None else SimplePalette()
        self._screen_data = bytearray(self.SCREEN_HEIGHT * self.SCREEN_STRIDE * 2)
        self._screen_array = np.ndarray(buffer=self._screen_data, shape=(self.SCREEN_HEIGHT, self.SCREEN_STRIDE), dtype='uint16')
        self._surface_array = np.zeros((self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 4), dtype='uint8')
        self._screen_surface = skia.Surface(self._surface_array)
        self._screen_update = asyncio.Condition()
        self._loop = None
        self._usb_device = None
        self._screen_endpoint = None
        self._midi_out = None
        self._midi_in = None
        self._receive_queue = asyncio.Queue()
        self._last_receive_timestamp = None
        self._counter = None
        self._clock_task = None
        self._screen_task = None

    @property
    def counter(self):
        return self._counter

    async def start(self, counter=None):
        self._counter = BeatCounter() if counter is None else counter
        self._loop = asyncio.get_running_loop()
        self._usb_device = usb.core.find(idVendor=self.USB_VENDOR, idProduct=self.USB_PRODUCT)
        if self._usb_device is None:
            raise RuntimeError("Cannot locate USB device")
        self._usb_device.reset()
        self._usb_device.set_configuration()
        self._screen_endpoint = self._usb_device.get_active_configuration()[0, 0][0]
        self._midi_out = rtmidi2.MidiOut()
        for i, name in enumerate(rtmidi2.get_out_ports()):
            if name == self.MIDI_PORT_NAME:
                self._midi_out.open_port(i)
                break
        else:
            raise RuntimeError("Cannot open MIDI output port: " + self.MIDI_PORT_NAME)
        self._midi_in = rtmidi2.MidiIn()
        for i, name in enumerate(rtmidi2.get_in_ports()):
            if name == self.MIDI_PORT_NAME:
                self._midi_in.open_port(i)
                break
        else:
            raise RuntimeError("Cannot open MIDI input port: " + self.MIDI_PORT_NAME)
        self._midi_in.callback = self._receive_callback
        self._send_midi([MIDI.STOP])
        self._send_sysex(Command.SET_MIDI_MODE, 1)
        self._send_sysex(Command.SET_AFTERTOUCH_MODE, 1)
        self._send_sysex(Command.SET_TOUCHSTRIP_CONFIG, TouchStripFlags.HOST_CONTROL | TouchStripFlags.LEDS_POINT)
        self.apply_palette()
        self._clock_task = asyncio.create_task(self._run_clock())
        self._screen_task = asyncio.create_task(self._run_screen())

    @property
    def palette(self):
        return self._palette

    @palette.setter
    def palette(self, palette):
        self._palette = palette
        self.apply_palette()

    def apply_palette(self):
        for i in range(128):
            r, g, b = self._palette.index_to_rgb_led(i)
            w = self._palette.index_to_white_led(i)
            self._send_sysex(Command.SET_COLOR_PALETTE_ENTRY, i, r & 0x7f, r >> 7, g & 0x7f, g >> 7, b & 0x7f, b >> 7, w & 0x7f, w >> 7)
        self._send_sysex(Command.REAPPLY_COLOR_PALETTE)

    async def stop(self):
        if self._clock_task is not None:
            self._clock_task.cancel()
            await self._clock_task
            self._clock_task = None
        if self._midi_out is not None:
            for number in BUTTONS:
                self.set_button_white(number, 0)
            for number in range(64):
                self.set_pad_rgb(number, 0, 0, 0)
            self.set_touch_strip_position(0)
            self._midi_out.close_port()
            self._midi_out = None
        if self._midi_in is not None:
            self._midi_in.close_port()
            self._midi_in = None
        if self._screen_task is not None:
            self._screen_task.cancel()
            await self._screen_task
            self._screen_task = None
        if self._usb_device is not None:
            self._screen_endpoint = None
            self._usb_device = None
        self._receive_queue = asyncio.Queue()
        self._last_receive_timestamp = None
        self._counter = None
        self._loop = None

    def _send_midi(self, message):
        logger.trace("Send MIDI - {}", " ".join(f"{b:02x}" for b in message))
        self._midi_out.send_raw(*message)

    def _receive_callback(self, message, delta):
        now = system_clock()
        timestamp = now if self._last_receive_timestamp is None else min(now, self._last_receive_timestamp + delta)
        self._last_receive_timestamp = timestamp
        logger.trace("Received @ {:.2f} MIDI - {}", timestamp, " ".join(f"{b:02x}" for b in message))
        self._loop.call_soon_threadsafe(self._receive_queue.put_nowait, (message, timestamp))

    def _send_sysex(self, cmd: Command, *args):
        self._send_midi([MIDI.START_OF_SYSEX] + self.SYSEX_ID + [cmd] + list(args) + [MIDI.END_OF_SYSEX])

    def set_pad_rgb(self, number, r, g, b, animation=Animation.NONE):
        self.set_pad_index(number, self._palette.rgb_to_index(r, g, b), animation)

    def set_pad_index(self, number, index, animation=Animation.NONE):
        assert number < 64
        assert index in range(0, 128)
        self._send_midi([MIDI.NOTE_ON + animation, number + Note.PAD_0_0, index])

    def set_button_rgb(self, number, r, g, b, animation=Animation.NONE):
        if number in COLOR_BUTTONS:
            self.set_button_index(number, self._palette.rgb_to_index(r, g, b), animation)
        else:
            w = 0.299*r + 0.587*g + 0.114*b
            self.set_button_index(number, self._palette.white_to_index(w), animation)

    def set_button_white(self, number, w, animation=Animation.NONE):
        if number in COLOR_BUTTONS:
            self.set_button_index(number, self._palette.rgb_to_index(w, w, w), animation)
        else:
            self.set_button_index(number, self._palette.white_to_index(w), animation)

    def set_button_index(self, number, index, animation=Animation.NONE):
        assert number in BUTTONS
        assert index in range(0, 128)
        self._send_midi([MIDI.CONTROL_CHANGE + animation, number, index])

    def set_touch_strip_position(self, position):
        value = int(min(max(0, position), 1) * 255) << 6
        self._send_midi([MIDI.PITCH_BEND_CHANGE, value & 0x7f, value >> 7])

    def set_led_brightness(self, brightness):
        value = int(round(min(max(0, brightness), 1) * 127))
        self._send_sysex(Command.SET_LED_BRIGHTNESS, value)

    def set_display_brightness(self, brightness):
        value = int(round(min(max(0, brightness), 1) * 255))
        self._send_sysex(Command.SET_DISPLAY_BRIGHTNESS, value & 0x7f, value >> 7)

    def get_event(self):
        try:
            message, timestamp = self._receive_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None
        return self.process_message(message, timestamp)

    async def wait_event(self):
        if self._clock_task.done():
            await self._clock_task
        if self._screen_task.done():
            await self._screen_task
        message, timestamp = await self._receive_queue.get()
        return self.process_message(message, timestamp)

    def process_message(self, message, timestamp):
        if message[0] == MIDI.NOTE_ON:
            if Note.PAD_0_0 <= message[1] <= Note.PAD_7_7:
                number = message[1] - Note.PAD_0_0
                column, row = number % 8, number // 8
                return PadPressed(timestamp=timestamp, number=number, column=column, row=row, pressure=message[2])
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
                return PadHeld(timestamp=timestamp, number=number, column=column, row=row, pressure=message[2])
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
        elif message[0] == MIDI.PITCH_BEND_CHANGE:
            position = (message[2] << 7) + message[1]
            return TouchStripDragged(timestamp=timestamp, position=position)
        logger.warning("Unrecognised MIDI - {}", " ".join(f"{b:02x}" for b in message))

    @asynccontextmanager
    async def screen_context(self):
        if self._screen_task is None:
            raise TypeError("Screen task not running")
        async with self._screen_update:
            with self._screen_surface as ctx:
                ctx.save()
                yield ctx
                ctx.restore()
            self._screen_update.notify()

    async def _run_clock(self):
        logger.trace("Started Push2 clock task")
        try:
            tick = int(self._counter.beat * 24)
            while True:
                if tick % (self._counter.quantum * 24) == 0:
                    self._send_midi([MIDI.START])
                else:
                    self._send_midi([MIDI.CLOCK])
                tick += 1
                await self._counter.wait_for_beat(tick / 24)
        except asyncio.CancelledError:
            logger.trace("Stopped Push2 clock task")
            self._send_midi([MIDI.STOP])
        except Exception:
            logger.exception("Unexpected exception in Push2 clock task")

    async def _run_screen(self):
        logger.trace("Started Push2 screen task")
        try:
            async with self._screen_update:
                while True:
                    await asyncio.to_thread(self._update_screen)
                    try:
                        await asyncio.wait_for(self._screen_update.wait(), timeout=1)
                    except asyncio.TimeoutError:
                        pass
        except asyncio.CancelledError:
            logger.trace("Stopped Push2 screen task")
            self._surface_array[:, :, :] = 0
            self._update_screen()
        except usb.core.USBError as exc:
            raise Push2CommunicationError("Error sending screen data") from exc
        except Exception:
            logger.exception("Unexpected exception in Push2 screen task")
            raise

    def _update_screen(self):
        self._screen_array[:, :] = 0
        pixels = self._screen_array[:, :self._surface_array.shape[1]]
        pixels += self._surface_array[:, :, 2] >> 3
        pixels <<= 6
        pixels += self._surface_array[:, :, 1] >> 2
        pixels <<= 5
        pixels += self._surface_array[:, :, 0] >> 3
        self._screen_array[:, 0::2] ^= 0xF3E7
        self._screen_array[:, 1::2] ^= 0xFFE7
        self._screen_endpoint.write(self.FRAME_HEADER)
        self._screen_endpoint.write(self._screen_data)
