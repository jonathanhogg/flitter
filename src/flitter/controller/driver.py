"""
Flitter controller driver API
"""

import math

from ..clock import BeatCounter
from ..model import Vector, Node, StateDict


class Control:
    def __init__(self, control_id):
        self.control_id = control_id
        self.reset()

    def reset(self):
        self._initialised = False
        self._state_prefix = None
        self._name = None
        self._color = None

    def update(self, node: Node, counter: BeatCounter, now: float, state: StateDict):
        self._initialised = True
        changed = False
        if (state_prefix := node.get('state')) != self._state_prefix:
            self._state_prefix = state_prefix
            changed = True
        if (name := node.get('name', 1, str)) != self._name:
            self._name = name
            changed = True
        if (color := node.get('color', 3, float)) != self._color:
            self._color = color
            changed = True
        return changed

    def update_representation(self):
        raise NotImplementedError()


class PositionControl(Control):
    DEFAULT_LAG = 1/4

    def reset(self):
        super().reset()
        self._lag = None
        self._lower = None
        self._upper = None
        self._raw_position = None
        self._position = None
        self._position_time = None

    def get_raw_divisor(self):
        raise NotImplementedError()

    def update(self, node, counter, now, state):
        changed = super().update(node, counter, now, state)
        if (lag := node.get('lag', 1, float, self.DEFAULT_LAG)) != self._lag:
            self._lag = lag
            changed = True
        if (lower := node.get('lower', 1, float, 0)) != self._lower:
            self._lower = lower
            changed = True
        if (upper := max(self._lower, node.get('upper', 1, float, 1))) != self._upper:
            self._upper = upper
            changed = True
        position_range = self._upper - self._lower
        if self._raw_position is not None:
            raw_divisor = self.get_raw_divisor()
            position = self._raw_position / raw_divisor * position_range + self._lower
            if self._position is not None and abs(position - self._position) > position_range / 1000:
                delta = counter.beat_at_time(now) - counter.beat_at_time(self._position_time)
                alpha = math.exp(-delta / self._lag)
                self._position = self._position * alpha + position * (1 - alpha)
                self._position_time = now
            elif self._position != position:
                self._position = position
                self._position_time = now
        if self._state_prefix and self._position is not None:
            state[self._state_prefix] = self._position
        return changed


class EncoderControl(PositionControl):
    STYLES = {'volume', 'pan', 'continuous'}

    def reset(self):
        super().reset()
        self._style = None
        self._turns = None

    def update(self, node, counter, now, state):
        changed = super().update(node, counter, now, state)
        style = node.get('style', 1, str, 'volume').lower()
        if style not in self.STYLES:
            style = 'volume'
        if style != self._style:
            self._style = style
            changed = True
        turns = node.get('turns', 1, float, 1)
        if turns != self._turns:
            self._turns = turns
            changed = True
        if self._raw_position is None:
            position_range = self._upper - self._lower
            if self._state_prefix and self._state_prefix in state:
                initial = float(state[self._state_prefix])
            else:
                initial = node.get('initial', 1, float, self._lower + position_range / 2 if self._style == 'pan' else self._lower)
            self._position = min(max(self._lower, initial), self._upper)
            self._position_time = now
            self._raw_position = (self._position - self._lower) / position_range * self.get_raw_divisor() if position_range else 0
            changed = True
        return changed


class ButtonControl(Control):
    def reset(self):
        super().reset()
        self._pushed = None
        self._push_time = None
        self._release_time = None
        self._toggle = None
        self._toggled = None
        self._toggle_time = None

    def update(self, node, counter, now, state):
        changed = super().update(node, counter, now, state)
        toggle = node.get('toggle', 1, bool, False)
        if toggle != self._toggle:
            self._toggle = toggle
            self._toggled = None
            self._toggle_time = None
            changed = True
        if self._toggle and self._toggled is None:
            self._toggled = node.get('initial', 1, bool, False)
            self._toggle_time = now
            changed = True
        if self._pushed is None:
            self._pushed = False
        if self._state_prefix:
            state[self._state_prefix] = self._pushed if not self._toggle else self._toggled
            state[self._state_prefix + ['pushed']] = self._pushed
            if self._push_time is not None:
                state[self._state_prefix + ['pushed', 'beat']] = counter.beat_at_time(self._push_time)
            if self._release_time is not None:
                state[self._state_prefix + ['released', 'beat']] = counter.beat_at_time(self._release_time)
            state[self._state_prefix + ['toggled']] = self._toggled
            if self._toggle_time is not None:
                state[self._state_prefix + ['toggled', 'beat']] = counter.beat_at_time(self._toggle_time)
        return changed


class ControllerDriver:
    def __init__(self, node: Node):
        pass

    @property
    def is_ready(self):
        raise NotImplementedError()

    async def start(self):
        raise NotImplementedError()

    def stop(self):
        raise NotImplementedError()

    def get_control(self, kind: str, control_id: Vector) -> Control:
        pass
