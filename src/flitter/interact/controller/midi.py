"""
Flitter generic MIDI interface
"""

import asyncio
from dataclasses import dataclass
import enum

from loguru import logger
import rtmidi2

from ...clock import system_clock


class MIDI(enum.IntEnum):
    NOTE_OFF = 0x80
    NOTE_ON = 0x90
    POLYPHONIC_PRESSURE = 0xA0
    CONTROL_CHANGE = 0xB0
    PROGRAM_CHANGE = 0xC0
    CHANNEL_PRESSURE = 0xD0
    PITCH_BEND_CHANGE = 0xE0
    SYSEX_START = 0xF0
    SYSEX_END = 0xF7
    CLOCK = 0xF8
    START = 0xFA
    CONTINUE = 0xFB
    STOP = 0xFC
    ACTIVE = 0xFE
    RESET = 0xFF


@dataclass
class MidiEvent:
    timestamp: float
    channel: int


@dataclass
class NoteEvent(MidiEvent):
    note: int


@dataclass
class NoteOnEvent(NoteEvent):
    velocity: int


@dataclass
class NotePressureEvent(NoteEvent):
    pressure: int


@dataclass
class NoteOffEvent(NoteEvent):
    pass


@dataclass
class ControlChangeEvent(MidiEvent):
    control: int
    value: int


@dataclass
class ChannelPressureEvent(MidiEvent):
    pressure: int


@dataclass
class PitchBendEvent(MidiEvent):
    value: int


class MidiPort:
    def __init__(self, name, input=True, output=True, sysex_id=None, virtual=False):
        self.name = name
        self.sysex_id = bytes(sysex_id) if sysex_id is not None else None
        self._midi_in = None
        self._midi_out = None
        try:
            if input:
                self._midi_in = rtmidi2.MidiIn()
                if virtual:
                    self._midi_in.open_virtual_port(name)
                else:
                    for port in rtmidi2.get_in_ports():
                        if port.split(':')[0] == name:
                            name = port
                    self._midi_in.open_port(name)
                self._last_receive_timestamp = None
                self._loop = asyncio.get_event_loop()
                self._receive_queue = asyncio.Queue()
                self._midi_in.callback = self._message_received
            if output:
                self._midi_out = rtmidi2.MidiOut()
                if virtual:
                    self._midi_out.open_virtual_port(self.name)
                else:
                    for port in rtmidi2.get_in_ports():
                        if port.split(':')[0] == name:
                            name = port
                    self._midi_out.open_port(name)
        except RuntimeError as exc:
            if self._midi_in is not None:
                self._midi_in.close()
            raise ValueError(f"Unable to open port {name}") from exc
        logger.debug("Opened MIDI port '{}'", self.name)

    def close(self):
        if self._midi_in is not None:
            self._midi_in.close_port()
            self._midi_in = None
        if self._midi_out is not None:
            self._midi_out.close_port()
            self._midi_out = None

    def _message_received(self, message, delta):
        now = system_clock()
        if self._last_receive_timestamp is None:
            timestamp = now
        else:
            timestamp = min(now, self._last_receive_timestamp + delta)
        self._last_receive_timestamp = timestamp
        logger.trace("Received MIDI @ {:.2f} - {}", timestamp, " ".join(f"{b:02x}" for b in message))
        self._loop.call_soon_threadsafe(self._receive_queue.put_nowait, (message, timestamp))

    async def wait_event(self, timeout=None):
        if self._midi_in is None:
            raise TypeError("MIDI port not open for input")
        try:
            message, timestamp = await asyncio.wait_for(self._receive_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
        event_type = message[0] & 0xf0
        channel = message[0] & 0x0f
        if event_type == MIDI.NOTE_ON:
            return NoteOnEvent(note=message[1], velocity=message[2], channel=channel, timestamp=timestamp)
        elif event_type == MIDI.POLYPHONIC_PRESSURE:
            return NotePressureEvent(note=message[1], pressure=message[2], channel=channel, timestamp=timestamp)
        elif event_type == MIDI.NOTE_OFF:
            return NoteOffEvent(note=message[1], channel=channel, timestamp=timestamp)
        elif event_type == MIDI.CONTROL_CHANGE:
            return ControlChangeEvent(control=message[1], value=message[2], channel=channel, timestamp=timestamp)
        elif event_type == MIDI.CHANNEL_PRESSURE:
            return ChannelPressureEvent(pressure=message[1], channel=channel, timestamp=timestamp)
        elif event_type == MIDI.PITCH_BEND_CHANGE:
            return PitchBendEvent(value=((message[2] << 7) + message[1]), channel=channel, timestamp=timestamp)
        logger.debug("Unhandled MIDI message: {}", " ".join(f"{b:02x}" for b in message))
        return None

    def send_raw(self, *data):
        if self._midi_out is None:
            raise TypeError("MIDI port not open for output")
        logger.trace("Send MIDI - {}", " ".join(f"{b:02x}" for b in data))
        self._midi_out.send_raw(*data)

    def send_note_off(self, note, channel=0):
        self.send_raw(MIDI.NOTE_OFF + channel, note)

    def send_note_on(self, note, velocity=127, channel=0):
        self.send_raw(MIDI.NOTE_ON + channel, note, velocity)

    def send_note_pressure(self, note, pressure, channel=0):
        self.send_raw(MIDI.POLYPHONIC_PRESSURE + channel, note, pressure)

    def send_channel_pressure(self, pressure, channel=0):
        self.send_raw(MIDI.CHANNEL_PRESSURE + channel, pressure)

    def send_control_change(self, control, value, channel=0):
        self.send_raw(MIDI.CONTROL_CHANGE + channel, control, value)

    def send_program_change(self, program, channel=0):
        self.send_raw(MIDI.PROGRAM_CHANGE + channel, program)

    def send_pitch_bend_change(self, value, channel=0):
        self.send_raw(MIDI.PITCH_BEND_CHANGE + channel, value & 0x7f, value >> 7)

    def send_sysex(self, *data, sysex_id=None):
        if sysex_id is None:
            sysex_id = self.sysex_id
            if sysex_id is None:
                raise TypeError("MIDI port sysex_id not set")
        self.send_raw(MIDI.SYSEX_START, *sysex_id, *data, MIDI.SYSEX_END)

    def send_clock(self):
        self.send_raw(MIDI.CLOCK)

    def send_start(self):
        self.send_raw(MIDI.START)

    def send_continue(self):
        self.send_raw(MIDI.CONTINUE)

    def send_stop(self):
        self.send_raw(MIDI.STOP)

    def send_active(self):
        self.send_raw(MIDI.ACTIVE)

    def send_reset(self):
        self.send_raw(MIDI.RESET)
