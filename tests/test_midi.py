"""
Tests of the flitter controller generic MIDI module
"""

import asyncio
import time
import unittest
from unittest.mock import patch

from loguru import logger

from flitter.clock import system_clock
from flitter.render.controller import midi


logger.disable('flitter.render.controller.midi')


@patch('flitter.render.controller.midi.rtmidi2')
class TestMidiPortCreation(unittest.IsolatedAsyncioTestCase):
    async def test_create_input(self, rtmidi2_mock):
        port = midi.MidiPort('TEST_MIDI_PORT', output=False)
        port._midi_in.open_port.assert_called_with('TEST_MIDI_PORT')

    def test_create_output(self, rtmidi2_mock):
        port = midi.MidiPort('TEST_MIDI_PORT', input=False)
        port._midi_out.open_port.assert_called_with('TEST_MIDI_PORT')

    async def test_create_input_output(self, rtmidi2_mock):
        port = midi.MidiPort('TEST_MIDI_PORT')
        port._midi_in.open_port.assert_called_with('TEST_MIDI_PORT')
        port._midi_out.open_port.assert_called_with('TEST_MIDI_PORT')

    def test_attempt_create_missing(self, rtmidi2_mock):
        rtmidi2_mock.MidiIn.side_effect = RuntimeError()
        rtmidi2_mock.MidiOut.side_effect = RuntimeError()
        with self.assertRaises(ValueError):
            midi.MidiPort('MISSING_PORT')

    def test_attempt_create_missing_input(self, rtmidi2_mock):
        rtmidi2_mock.MidiIn.side_effect = RuntimeError()
        rtmidi2_mock.MidiOut.side_effect = RuntimeError()
        with self.assertRaises(ValueError):
            midi.MidiPort('MISSING_PORT', output=False)

    def test_attempt_create_missing_output(self, rtmidi2_mock):
        rtmidi2_mock.MidiIn.side_effect = RuntimeError()
        rtmidi2_mock.MidiOut.side_effect = RuntimeError()
        with self.assertRaises(ValueError):
            midi.MidiPort('MISSING_PORT', input=False)

    async def test_create_multiple_ports(self, rtmidi2_mock):
        port1 = midi.MidiPort('TEST_MIDI_PORT1')
        port1._midi_in.open_port.assert_called_with('TEST_MIDI_PORT1')
        port1._midi_out.open_port.assert_called_with('TEST_MIDI_PORT1')
        port2 = midi.MidiPort('TEST_MIDI_PORT2')
        port2._midi_in.open_port.assert_called_with('TEST_MIDI_PORT2')
        port2._midi_out.open_port.assert_called_with('TEST_MIDI_PORT2')

    async def test_virtual_port(self, rtmidi2_mock):
        port = midi.MidiPort('TEST_VIRTUAL_PORT', virtual=True)
        port._midi_in.open_virtual_port.assert_called_with('TEST_VIRTUAL_PORT')
        port._midi_out.open_virtual_port.assert_called_with('TEST_VIRTUAL_PORT')


@patch('flitter.render.controller.midi.rtmidi2')
class TestMidiPortClose(unittest.IsolatedAsyncioTestCase):
    def test_close_input_output(self, rtmidi2_mock):
        port = midi.MidiPort('TEST_MIDI_PORT')
        midi_in_mock = port._midi_in
        midi_out_mock = port._midi_out
        port.close()
        midi_in_mock.close_port.assert_called_with()
        midi_out_mock.close_port.assert_called_with()
        self.assertIsNone(port._midi_in)
        self.assertIsNone(port._midi_out)

    def test_close_send_fail(self, rtmidi2_mock):
        port = midi.MidiPort('TEST_MIDI_PORT', input=False)
        port.send_note_off(10)
        port.close()
        with self.assertRaises(TypeError):
            port.send_note_off(10)

    async def test_close_wait_fail(self, rtmidi2_mock):
        port = midi.MidiPort('TEST_MIDI_PORT', output=False)
        await port.wait_event(0.1)
        port.close()
        with self.assertRaises(TypeError):
            await port.wait_event(0.1)


class TestReceiveMidi(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.rtmidi2_patcher = patch('flitter.render.controller.midi.rtmidi2')
        self.rtmidi2_mock = self.rtmidi2_patcher.start()
        self.port = midi.MidiPort('TEST_MIDI_PORT', output=False)

    def tearDown(self):
        self.rtmidi2_patcher.stop()

    def test_send_attempt(self):
        with self.assertRaises(TypeError):
            self.port.send_note_off(10)

    async def test_receive_timeout(self):
        with self.assertRaises(asyncio.TimeoutError):
            await asyncio.wait_for(self.port.wait_event(), 0.1)

    async def test_receive_with_timeout(self):
        event = await self.port.wait_event(0.1)
        self.assertIsNone(event)

    async def test_receive_note_off(self):
        start = system_clock()
        self.port._midi_in.callback(bytes([0x80, 0x0a]), 0.0)
        end = system_clock()
        event = await self.port.wait_event()
        self.assertIsInstance(event, midi.NoteOffEvent)
        self.assertEqual(event.channel, 0)
        self.assertTrue(start < event.timestamp < end)
        self.assertEqual(event.note, 10)

    async def test_receive_note_on(self):
        start = system_clock()
        self.port._midi_in.callback(bytes([0x91, 0x0a, 0x7f]), 0.0)
        end = system_clock()
        event = await self.port.wait_event()
        self.assertIsInstance(event, midi.NoteOnEvent)
        self.assertEqual(event.channel, 1)
        self.assertTrue(start < event.timestamp < end)
        self.assertEqual(event.note, 10)
        self.assertEqual(event.velocity, 127)

    async def test_receive_note_pressure(self):
        start = system_clock()
        self.port._midi_in.callback(bytes([0xa2, 0x0a, 0x37]), 0.0)
        end = system_clock()
        event = await self.port.wait_event()
        self.assertIsInstance(event, midi.NotePressureEvent)
        self.assertEqual(event.channel, 2)
        self.assertTrue(start < event.timestamp < end)
        self.assertEqual(event.note, 10)
        self.assertEqual(event.pressure, 55)

    async def test_receive_control_change(self):
        start = system_clock()
        self.port._midi_in.callback(bytes([0xb3, 0x10, 0x21]), 0.0)
        end = system_clock()
        event = await self.port.wait_event()
        self.assertIsInstance(event, midi.ControlChangeEvent)
        self.assertEqual(event.channel, 3)
        self.assertTrue(start < event.timestamp < end)
        self.assertEqual(event.control, 16)
        self.assertEqual(event.value, 33)

    async def test_receive_channel_pressure(self):
        start = system_clock()
        self.port._midi_in.callback(bytes([0xd4, 0x37]), 0.0)
        end = system_clock()
        event = await self.port.wait_event()
        self.assertIsInstance(event, midi.ChannelPressureEvent)
        self.assertEqual(event.channel, 4)
        self.assertTrue(start < event.timestamp < end)
        self.assertEqual(event.pressure, 55)

    async def test_receive_pitch_bend(self):
        start = system_clock()
        self.port._midi_in.callback(bytes([0xe5, 0x72, 0x7f]), 0.0)
        end = system_clock()
        event = await self.port.wait_event()
        self.assertIsInstance(event, midi.PitchBendEvent)
        self.assertEqual(event.channel, 5)
        self.assertTrue(start < event.timestamp < end)
        self.assertEqual(event.value, 16370)

    async def test_receive_timing(self):
        start = system_clock()
        self.port._midi_in.callback(bytes([0x80, 0x0a]), 0.0)
        time.sleep(1/20)
        self.port._midi_in.callback(bytes([0x80, 0x0a]), 1/32)
        event1 = await self.port.wait_event()
        event2 = await self.port.wait_event()
        end = system_clock()
        self.assertTrue(start < event1.timestamp < end)
        self.assertEqual(event2.timestamp, event1.timestamp + 1/32)


class TestSendMidi(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.rtmidi2_patcher = patch('flitter.render.controller.midi.rtmidi2')
        self.rtmidi2_mock = self.rtmidi2_patcher.start()
        self.port = midi.MidiPort('TEST_MIDI_PORT', input=False, sysex_id=(0x01, 0x02, 0x03))

    def tearDown(self):
        self.rtmidi2_patcher.stop()

    async def test_receive_attempt(self):
        with self.assertRaises(TypeError):
            await self.port.wait_event()

    def test_send_note_off(self):
        self.port.send_note_off(10)
        self.port._midi_out.send_raw.assert_called_with(0x80, 0x0a)

    def test_send_note_on(self):
        self.port.send_note_on(10, channel=1)
        self.port._midi_out.send_raw.assert_called_with(0x91, 0x0a, 0x7f)
        self.port.send_note_on(10, 55, channel=2)
        self.port._midi_out.send_raw.assert_called_with(0x92, 0x0a, 0x37)

    def test_send_note_pressure(self):
        self.port.send_note_pressure(10, 55, channel=2)
        self.port._midi_out.send_raw.assert_called_with(0xa2, 0x0a, 0x37)

    def test_send_control_change(self):
        self.port.send_control_change(16, 33, channel=3)
        self.port._midi_out.send_raw.assert_called_with(0xb3, 0x10, 0x21)

    def test_send_program_change(self):
        self.port.send_program_change(19, channel=9)
        self.port._midi_out.send_raw.assert_called_with(0xc9, 0x13)

    def test_send_channel_pressure(self):
        self.port.send_channel_pressure(55, channel=4)
        self.port._midi_out.send_raw.assert_called_with(0xd4, 0x37)

    def test_send_pitch_bend_change(self):
        self.port.send_pitch_bend_change(16370, channel=5)
        self.port._midi_out.send_raw.assert_called_with(0xe5, 0x72, 0x7f)

    def test_send_sysex(self):
        self.port.send_sysex(0x04, 0x05, 0x06)
        self.port._midi_out.send_raw.assert_called_with(0xf0, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0xf7)

    def test_send_sysex_explicit_id(self):
        self.port.send_sysex(0x04, 0x05, 0x06, sysex_id=(0x71, 0x72, 0x73))
        self.port._midi_out.send_raw.assert_called_with(0xf0, 0x71, 0x72, 0x73, 0x04, 0x05, 0x06, 0xf7)

    def test_send_clock(self):
        self.port.send_clock()
        self.port._midi_out.send_raw.assert_called_with(0xf8)

    def test_send_start(self):
        self.port.send_start()
        self.port._midi_out.send_raw.assert_called_with(0xfa)

    def test_send_continue(self):
        self.port.send_continue()
        self.port._midi_out.send_raw.assert_called_with(0xfb)

    def test_send_stop(self):
        self.port.send_stop()
        self.port._midi_out.send_raw.assert_called_with(0xfc)

    def test_send_active(self):
        self.port.send_active()
        self.port._midi_out.send_raw.assert_called_with(0xfe)

    def test_send_reset(self):
        self.port.send_reset()
        self.port._midi_out.send_raw.assert_called_with(0xff)
