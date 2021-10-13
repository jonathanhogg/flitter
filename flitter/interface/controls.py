"""
Flitter user interface controls
"""

# pylama:ignore=R0902,R0903

import math

from ..model import Vector, true, false


class Control:
    DEFAULT_NAME = "?"
    DEFAULT_COLOR = (1.0, 1.0, 1.0)

    def __init__(self, number):
        self.number = number
        self.name = self.DEFAULT_NAME
        self.color = self.DEFAULT_COLOR
        self.state = None
        self._changed = False

    def update(self, node, _):
        changed = self._changed
        self._changed = False
        if 'state' in node:
            state = node['state']
            if state != self.state:
                self.state = state
                self.reset()
                changed = True
            name = node.get('name', 1, str, self.DEFAULT_NAME)
            if name != self.name:
                self.name = name
                changed = True
            color = tuple(node.get('color', 3, float, self.DEFAULT_COLOR))
            if color != self.color:
                self.color = color
                changed = True
            return changed
        if self.state is not None:
            self.reset()
            return True
        return False

    def reset(self):
        self.name = self.DEFAULT_NAME
        self.color = self.DEFAULT_COLOR


class TouchControl(Control):
    def __init__(self, number):
        super().__init__(number)
        self.touched = None
        self._touched_beat = None

    def update(self, node, controller):
        changed = super().update(node, controller)
        if self.touched is not None:
            controller[Vector((*self.state, "touched"))] = true if self.touched else false
            controller[Vector((*self.state, "touched", "beat"))] = Vector((self._touched_beat,))
        return changed

    def on_touched(self, beat):
        self.touched = True
        self._touched_beat = beat
        self._changed = True

    def on_released(self, beat):
        self.touched = False
        self._touched_beat = beat
        self._changed = True

    def reset(self):
        super().reset()
        self.touched = None
        self._touched_beat = None


class Pad(TouchControl):
    DEFAULT_THRESHOLD = 0.4

    def __init__(self, number):
        super().__init__(number)
        self.toggle = None
        self.toggled = None
        self._toggled_beat = None
        self.pressure = None
        self._pressure_beat = None
        self._toggle_threshold = self.DEFAULT_THRESHOLD
        self._can_toggle = False
        self._clock = None

    def update(self, node, controller):
        changed = super().update(node, controller)
        if self.state is not None:
            now = controller.counter.clock()
            delta = 0.0 if self._clock is None else now - self._clock
            self._clock = now
            toggle = node.get('toggle', 1, bool, False)
            toggled_key = Vector((*self.state, "toggled"))
            toggled_beat_key = Vector((*self.state, "toggled", "beat"))
            if toggle != self.toggle:
                self.toggle = toggle
                if not self.toggle and self.toggled:
                    self.toggled = False
                    self._toggled_beat = controller.counter.beat
                self._can_toggle = False
                changed = True
            if self.toggled is None and toggled_key in controller:
                self.toggled = controller[toggled_key].istrue()
                self._toggled_beat = controller[toggled_beat_key][0]
            self._toggle_threshold = node.get('threshold', 1, float, self.DEFAULT_THRESHOLD)
            if self.pressure is not None:
                pressure_key = Vector((*self.state, "pressure"))
                pressure_beat_key = Vector((*self.state, "pressure", "beat"))
                pressure = self.pressure
                if pressure_key in controller:
                    alpha = math.exp(20 * -delta)
                    current_pressure = controller[pressure_key][0]
                    pressure = pressure * (1-alpha) + current_pressure * alpha
                    if math.isclose(pressure, self.pressure, rel_tol=1e-3, abs_tol=1e-3):
                        pressure = self.pressure
                controller[pressure_key] = Vector((pressure,))
                controller[pressure_beat_key] = Vector((self._pressure_beat,))
            if self.toggled is not None:
                controller[toggled_key] = true if self.toggled else false
                controller[toggled_beat_key] = Vector((self._toggled_beat,))
        return changed

    def on_touched(self, beat):
        super().on_touched(beat)
        self._can_toggle = self.toggle

    def on_released(self, beat):
        super().on_released(beat)
        self._can_toggle = False

    def on_pressure(self, beat, pressure):
        self.pressure = pressure
        self._pressure_beat = beat
        if self._can_toggle and self.pressure > self._toggle_threshold:
            self.toggled = not self.toggled
            self._toggled_beat = beat
            self._can_toggle = False
            self._changed = True

    def reset(self):
        super().reset()
        self.toggle = None
        self.toggled = None
        self._toggled_beat = None
        self.pressure = None
        self._pressure_beat = None
        self._toggle_threshold = self.DEFAULT_THRESHOLD
        self._can_toggle = False


class Encoder(TouchControl):
    DEFAULT_LOWER = 0.0
    DEFAULT_UPPER = 1.0

    def __init__(self, number):
        super().__init__(number)
        self.initial = None
        self.lower = None
        self.upper = None
        self.value = None
        self._value_beat = None
        self._clock = None

    def update(self, node, controller):
        changed = super().update(node, controller)
        if self.state is not None:
            now = controller.counter.clock()
            delta = 0.0 if self._clock is None else now - self._clock
            self._clock = now
            lower = node.get('lower', 1, float, self.DEFAULT_LOWER)
            if lower != self.lower:
                self.lower = lower
                if self.value is not None:
                    self.value = max(self.lower, self.value)
                changed = True
            upper = node.get('upper', 1, float, self.DEFAULT_UPPER)
            if upper != self.upper:
                self.upper = upper
                if self.value is not None:
                    self.value = min(self.value, self.upper)
                changed = True
            initial = node.get('initial', 1, float, self.lower)
            if initial != self.initial:
                self.initial = initial
                changed = True
            value_key = Vector((*self.state, "value"))
            value_beat_key = Vector((*self.state, "value", "beat"))
            if self.value is None:
                if value_key in controller:
                    self.value = controller[value_key][0]
                    self._value_beat = controller[value_beat_key][0]
                else:
                    self.value = self.initial
                    self._value_beat = controller.counter.beat
            value = self.value
            if value_key in controller:
                alpha = math.exp(10 * -delta)
                current_value = controller[value_key][0]
                value = value * (1-alpha) + current_value * alpha
                if math.isclose(value, self.value, rel_tol=1e-3, abs_tol=1e-3):
                    value = self.value
            controller[value_key] = Vector((value,))
            controller[value_beat_key] = Vector((self._value_beat,))
        return changed

    def on_turned(self, beat, amount):
        self.value = min(max(self.lower, self.value + amount * (self.upper - self.lower)), self.upper)
        self._value_beat = beat
        self._changed = True

    def on_reset(self, beat):
        self.value = self.initial
        self._value_beat = beat
        self._changed = True

    def reset(self):
        self.initial = None
        self.lower = None
        self.upper = None
        self.value = None
        self._value_beat = None
        self._clock = None
