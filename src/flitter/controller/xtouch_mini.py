"""
Behringer X-Touch Mini controller driver
"""

import asyncio

from loguru import logger

from . import driver, midi
from ..model import Vector


def get_driver_class():
    return XTouchMiniDriver


BUTTON_NOTE_MAPPING = {
    Vector(("rotary", 1)): 32, Vector(("rotary", 2)): 33, Vector(("rotary", 3)): 34, Vector(("rotary", 4)): 35,
    Vector(("rotary", 5)): 36, Vector(("rotary", 6)): 37, Vector(("rotary", 7)): 38, Vector(("rotary", 8)): 39,
    Vector(1): 89, Vector(2): 90, Vector(3): 40, Vector(4): 41,
    Vector(5): 42, Vector(6): 43, Vector(7): 44, Vector(8): 45,
    Vector(9): 87, Vector(10): 88, Vector(11): 91, Vector(12): 92,
    Vector(13): 86, Vector(14): 93, Vector(15): 94, Vector(16): 95,
    Vector("a"): 84, Vector("b"): 85,
}

NOTE_BUTTON_MAPPING = {note: button_id for button_id, note in BUTTON_NOTE_MAPPING.items()}

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


class XTouchMiniRotary(driver.EncoderControl):
    def __init__(self, control_id, driver, light_control):
        super().__init__(control_id)
        self._driver = driver
        self._light_control = light_control

    def get_raw_divisor(self):
        return int(round(24 * self._turns))

    def update_representation(self):
        if self._driver._midi_port is None:
            return
        if self._initialised:
            divisor = self.get_raw_divisor()
            if self._upper == self._lower:
                value = 0x3f
            elif self._style == 'volume':
                value = int(round(self._raw_position / divisor * 10)) + 0x21
            elif self._style == 'pan':
                value = int(round(self._raw_position / divisor * 10)) + 0x11
            else:
                value = int(self._raw_position / divisor % 1 * 11) + 0x01
            self._driver._midi_port.send_control_change(self._light_control, value)
        else:
            self._driver._midi_port.send_control_change(self._light_control, 0)

    def _handle_event(self, event):
        if not self._initialised:
            return
        raw_position = self._raw_position
        delta = 64 - event.value if event.value > 64 else event.value
        raw_position += delta
        if self._style != 'continuous':
            raw_position = min(max(0, raw_position), self.get_raw_divisor())
        if raw_position != self._raw_position:
            self._raw_position = raw_position
            self._position_time = event.timestamp
            self.update_representation()


class XTouchMiniButton(driver.ButtonControl):
    def __init__(self, control_id, driver, light_note):
        super().__init__(control_id)
        self._driver = driver
        self._light_note = light_note

    def update_representation(self):
        if self._driver._midi_port is None:
            return
        if self._initialised:
            if self._toggle:
                velocity = 127 if self._toggled else 0
            else:
                velocity = 127 if self._pushed else 0
            self._driver._midi_port.send_note_on(self._light_note, velocity)
        else:
            self._driver._midi_port.send_note_on(self._light_note, 0)

    def _handle_event(self, event):
        if not self._initialised:
            return
        pushed = event.velocity == 127
        if pushed != self._pushed:
            self._pushed = pushed
            if self._pushed:
                self._push_time = event.timestamp
                if self._toggle:
                    self._toggled = not self._toggled
                    self._toggle_time = event.timestamp
            else:
                self._release_time = event.timestamp
            self.update_representation()


class XTouchMiniFader(driver.PositionControl):
    def update_representation(self):
        pass

    def get_raw_divisor(self):
        return 16256

    def _handle_event(self, event):
        self._raw_position = event.value


class XTouchMiniDriver(driver.ControllerDriver):
    def __init__(self, node):
        self._port_name = node.get('port', 1, str, 'X-TOUCH MINI')
        self._rotaries = {}
        self._buttons = {}
        self._sliders = {}
        self._run_task = None
        self._midi_port = None
        self._ready = asyncio.Event()
        for rotary_id, (_, light_control) in ROTARY_CONTROLS_MAPPING.items():
            rotary = XTouchMiniRotary(rotary_id, self, light_control)
            self._rotaries[rotary_id] = rotary
        for button_id, light_note in BUTTON_NOTE_MAPPING.items():
            button = XTouchMiniButton(button_id, self, light_note)
            self._buttons[button_id] = button
        self._sliders[Vector('main')] = XTouchMiniFader(Vector('main'))

    @property
    def is_ready(self):
        return self._ready.is_set()

    async def start(self):
        self._run_task = asyncio.create_task(self.run())

    def stop(self):
        self._run_task.cancel()
        self._run_task = None

    async def run(self):
        try:
            while True:
                while not self._try_connect():
                    await asyncio.sleep(1)
                self.refresh()
                self._ready.set()
                logger.debug("X-Touch Mini controller driver ready")
                while True:
                    event = await self._midi_port.wait_event(1)
                    if event is None:
                        self.refresh()
                        continue
                    match event:
                        case midi.NoteOnEvent(note=note, channel=0) if note in NOTE_BUTTON_MAPPING:
                            button = self._buttons[NOTE_BUTTON_MAPPING[note]]
                            button._handle_event(event)
                        case midi.ControlChangeEvent(control=control, channel=0) if control in TURN_CONTROL_ROTARY_MAPPING:
                            rotary = self._rotaries[TURN_CONTROL_ROTARY_MAPPING[control]]
                            rotary._handle_event(event)
                        case midi.PitchBendEvent(channel=8):
                            self._sliders[Vector('main')]._handle_event(event)
                        case _:
                            logger.warning("Unhandled MIDI event: {}", event)
        except asyncio.CancelledError:
            self._ready.clear()
            self._midi_port.close()
            self._midi_port = None
        except Exception as exc:
            logger.opt(exception=exc).error("Unhandled exception in X-Touch mini controller main loop")

    def refresh(self):
        self._midi_port.send_control_change(1, 127, channel=11)  # Switch to Mackie Control mode
        for rotary in self._rotaries.values():
            rotary.update_representation()
        for button in self._buttons.values():
            button.update_representation()
        for slider in self._sliders.values():
            slider.update_representation()

    def _try_connect(self):
        try:
            self._midi_port = midi.MidiPort(self._port_name)
        except ValueError:
            return False
        return True

    def get_control(self, kind, control_id):
        control_id = ALIASES.get((kind, control_id), control_id)
        if kind == 'rotary':
            return self._rotaries.get(control_id)
        if kind == 'button':
            return self._buttons.get(control_id)
        if kind == 'slider':
            return self._sliders.get(control_id)


async def main():
    import flitter
    from flitter.clock import BeatCounter, system_clock
    from flitter.model import Node, StateDict
    flitter.configure_logger('TRACE')
    driver = XTouchMiniDriver(Node('controller'))
    await driver.start()
    await driver._ready.wait()
    counter = BeatCounter()
    state = StateDict()
    while True:
        now = system_clock()
        state.set_time(now)
        rotary = driver.get_rotary(Vector(1))
        if rotary.update(Node('rotary'), counter, now, state):
            rotary.update_representation()
        rotary = driver.get_rotary(Vector(2))
        if rotary.update(Node('rotary', set(), {'state': 2, 'style': 'pan', 'turns': 2}), counter, now, state):
            rotary.update_representation()
        rotary = driver.get_rotary(Vector(3))
        if rotary.update(Node('rotary', set(), {'state': 3, 'style': 'pan', 'lower': -10, 'upper': 10}), counter, now, state):
            rotary.update_representation()
        rotary = driver.get_rotary(Vector(4))
        if rotary.update(Node('rotary', set(), {'state': 4, 'style': 'continuous'}), counter, now, state):
            rotary.update_representation()
        rotary = driver.get_rotary(Vector(8))
        if rotary.update(Node('rotary', set(), {'state': 8, 'upper': 0}), counter, now, state):
            rotary.update_representation()
        button = driver.get_button(Vector(1))
        if button.update(Node('button', set(), {'state': 'button1'}), counter, now, state):
            button.update_representation()
        button = driver.get_button(Vector('a'))
        if button.update(Node('button', set(), {'state': 'a', 'toggle': True}), counter, now, state):
            button.update_representation()
        button = driver.get_button(Vector('b'))
        if button.update(Node('button', set(), {'state': 'b', 'toggle': True, 'initial': True}), counter, now, state):
            button.update_representation()
        slider = driver.get_slider(Vector(None))
        if slider.update(Node('slider', set(), {'state': 'fader'}), counter, now, state):
            slider.update_representation()
        print(state)
        await asyncio.sleep(1/10)


if __name__ == '__main__':
    asyncio.run(main())
