"""
Behringer X-Touch Mini controller driver
"""

import asyncio

from loguru import logger

from . import driver, midi
from ...model import Vector, Node


def get_driver_class():
    return XTouchMiniDriver


DEFAULT_CONFIG = [
    Node('button', attributes={'id': Vector('a'), 'action': Vector('next')}),
    Node('button', attributes={'id': Vector('b'), 'action': Vector('previous')}),
    Node('button', attributes={'id': Vector(('rotary', 1)), 'action': Vector(('reset', 'rotary', 1))}),
    Node('button', attributes={'id': Vector(('rotary', 2)), 'action': Vector(('reset', 'rotary', 2))}),
    Node('button', attributes={'id': Vector(('rotary', 3)), 'action': Vector(('reset', 'rotary', 3))}),
    Node('button', attributes={'id': Vector(('rotary', 4)), 'action': Vector(('reset', 'rotary', 4))}),
    Node('button', attributes={'id': Vector(('rotary', 5)), 'action': Vector(('reset', 'rotary', 5))}),
    Node('button', attributes={'id': Vector(('rotary', 6)), 'action': Vector(('reset', 'rotary', 6))}),
    Node('button', attributes={'id': Vector(('rotary', 7)), 'action': Vector(('reset', 'rotary', 7))}),
    Node('button', attributes={'id': Vector(('rotary', 8)), 'action': Vector(('reset', 'rotary', 8))}),
]

NOTE_BUTTON_MAPPING = {
    32: Vector(('rotary', 1)), 33: Vector(('rotary', 2)), 34: Vector(('rotary', 3)), 35: Vector(('rotary', 4)),
    36: Vector(('rotary', 5)), 37: Vector(('rotary', 6)), 38: Vector(('rotary', 7)), 39: Vector(('rotary', 8)),
    89: Vector(1), 90: Vector(2), 40: Vector(3), 41: Vector(4), 42: Vector(5), 43: Vector(6), 44: Vector(7), 45: Vector(8),
    87: Vector(9), 88: Vector(10), 91: Vector(11), 92: Vector(12), 86: Vector(13), 93: Vector(14), 94: Vector(15), 95: Vector(16),
    84: Vector('a'), 85: Vector('b'),
}

BUTTON_LIGHT_MAPPING = {
    Vector(1): 89, Vector(2): 90, Vector(3): 40, Vector(4): 41, Vector(5): 42, Vector(6): 43, Vector(7): 44, Vector(8): 45,
    Vector(9): 87, Vector(10): 88, Vector(11): 91, Vector(12): 92, Vector(13): 86, Vector(14): 93, Vector(15): 94, Vector(16): 95,
    Vector("a"): 84, Vector("b"): 85,
}

ROTARY_CONTROLS_MAPPING = {
    Vector(1): (16, 48), Vector(2): (17, 49), Vector(3): (18, 50), Vector(4): (19, 51),
    Vector(5): (20, 52), Vector(6): (21, 53), Vector(7): (22, 54), Vector(8): (23, 55),
}

TURN_CONTROL_ROTARY_MAPPING = {turn_control: rotary_id for rotary_id, (turn_control, light_control) in ROTARY_CONTROLS_MAPPING.items()}

ALIASES = {
    ('button', Vector('mc')): Vector(9),
    ('button', Vector('reverse')): Vector(11),
    ('button', Vector('forward')): Vector(12),
    ('button', Vector('loop')): Vector(13),
    ('button', Vector('stop')): Vector(14),
    ('button', Vector('play')): Vector(15),
    ('button', Vector('record')): Vector(16),
}

SPECIAL_ACTIONS = {
    Vector("a"): 'next', Vector("b"): 'previous',
}


class XTouchMiniRotary(driver.EncoderControl):
    DEFAULT_LAG = 1/4

    def __init__(self, driver, control_id, light_control):
        super().__init__(driver, control_id)
        self._light_control = light_control

    @property
    def raw_divisor(self):
        return int(round(24 * self._turns))

    def update_representation(self):
        if self.driver._midi_port is None:
            return
        if self._initialised:
            divisor = self.raw_divisor
            if self._upper == self._lower:
                value = 0x3f
            elif self._style == 'volume':
                value = int(round(self._raw_position / divisor * 10)) + 0x21
            elif self._style == 'pan':
                value = int(round(self._raw_position / divisor * 10)) + 0x11
            else:
                value = int(self._raw_position / divisor % 1 * 11) + 0x01
            self.driver._midi_port.send_control_change(self._light_control, value)
        else:
            self.driver._midi_port.send_control_change(self._light_control, 0)


class XTouchMiniButton(driver.ButtonControl):
    def __init__(self, driver, control_id, light_note):
        super().__init__(driver, control_id)
        self._light_note = light_note

    def update_representation(self):
        if self._light_note is None or self.driver._midi_port is None:
            return
        if self._initialised:
            if self._action:
                velocity = 127 if self._action_can_trigger else 0
            elif self._color is not None:
                velocity = 127 if self._color != [0, 0, 0] else 0
            elif self._toggle:
                velocity = 127 if self._toggled else 0
            else:
                velocity = 127 if self._pushed else 0
            self.driver._midi_port.send_note_on(self._light_note, velocity)
        else:
            self.driver._midi_port.send_note_on(self._light_note, 0)


class XTouchMiniFader(driver.PositionControl):
    def reset(self):
        raw_position = self._raw_position if hasattr(self, '_raw_position') else None
        super().reset()
        self._raw_position = raw_position

    def update_representation(self):
        pass

    @property
    def raw_divisor(self):
        return 16256


class XTouchMiniDriver(driver.ControllerDriver):
    VENDOR_ID = 0x1397
    PRODUCT_ID = 0x00b3
    PORT_NAME = 'X-TOUCH MINI'

    def __init__(self, engine):
        super().__init__(engine)
        self._rotaries = {}
        self._buttons = {}
        self._sliders = {}
        self._toggle_groups = {}
        self._run_task = None
        self._midi_port = None
        for rotary_id, (_, light_control) in ROTARY_CONTROLS_MAPPING.items():
            rotary = XTouchMiniRotary(self, rotary_id, light_control)
            self._rotaries[rotary_id] = rotary
        for note, button_id in NOTE_BUTTON_MAPPING.items():
            light_note = BUTTON_LIGHT_MAPPING.get(button_id)
            button = XTouchMiniButton(self, button_id, light_note)
            self._buttons[button_id] = button
        self._sliders[Vector('main')] = XTouchMiniFader(self, Vector('main'))

    async def start(self):
        self._run_task = asyncio.create_task(self.run())

    async def stop(self):
        self._run_task.cancel()
        await self._run_task
        self._run_task = None

    def get_default_config(self):
        return DEFAULT_CONFIG

    async def run(self):
        try:
            while True:
                while not self._try_connect():
                    await asyncio.sleep(1)
                self._midi_port.send_control_change(127, 1, channel=11)  # Switch to Mackie Control mode
                self.refresh()
                logger.debug("X-Touch Mini controller driver ready")
                while True:
                    event = await self._midi_port.wait_event(1)
                    if event is None:
                        self.refresh()
                        continue
                    match event:
                        case midi.NoteOnEvent(note=note, channel=0) if note in NOTE_BUTTON_MAPPING:
                            button = self._buttons[NOTE_BUTTON_MAPPING[note]]
                            if button.handle_push(event.velocity == 127, event.timestamp):
                                button.update_representation()
                        case midi.ControlChangeEvent(control=control, value=value, channel=0) if control in TURN_CONTROL_ROTARY_MAPPING:
                            rotary = self._rotaries[TURN_CONTROL_ROTARY_MAPPING[control]]
                            if rotary.handle_turn(64 - value if value > 64 else value, event.timestamp):
                                rotary.update_representation()
                        case midi.PitchBendEvent(channel=8):
                            self._sliders[Vector('main')].handle_raw_position_change(event.value, event.timestamp)
                        case _:
                            logger.warning("Unhandled MIDI event: {}", event)
        except asyncio.CancelledError:
            if self._midi_port is not None:
                self._midi_port.send_control_change(127, 0, channel=11)
                self._midi_port.close()
                self._midi_port = None
        except Exception as exc:
            logger.opt(exception=exc).error("Unhandled exception in X-Touch mini controller main loop")

    def refresh(self):
        for rotary in self._rotaries.values():
            rotary.update_representation()
        for button in self._buttons.values():
            button.update_representation()
        for slider in self._sliders.values():
            slider.update_representation()

    def _try_connect(self):
        try:
            self._midi_port = midi.MidiPort(self.PORT_NAME)
        except ValueError:
            return False
        else:
            return True

    def get_control(self, kind, control_id):
        control_id = ALIASES.get((kind, control_id), control_id)
        if kind == 'rotary':
            return self._rotaries.get(control_id)
        if kind == 'button':
            return self._buttons.get(control_id)
        if kind == 'slider':
            return self._sliders.get(control_id)
