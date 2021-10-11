"""
Flitter user interface controls
"""

# pylama:ignore=R0902,R0903

from ..model import Vector, true, false


class Control:
    DEFAULT_NAME = "?"
    DEFAULT_COLOR = (1.0, 1.0, 1.0)

    def __init__(self, number):
        self.number = number
        self.name = self.DEFAULT_NAME
        self.color = self.DEFAULT_COLOR
        self._changed = False

    def update(self, node, _):
        changed = self._changed
        self._changed = False
        name = node.get('name', 1, str, self.DEFAULT_NAME)
        if name != self.name:
            self.name = name
            changed = True
        color = tuple(node.get('color', 3, float, self.DEFAULT_COLOR))
        if color != self.color:
            self.color = color
            changed = True
        return changed


class TouchControl(Control):
    def __init__(self, number):
        super().__init__(number)
        self.touched = None
        self._touched_beat = None

    def update(self, node, controller):
        changed = super().update(node, controller)
        if self.touched is not None:
            controller[Vector((*self._prefix, "touched"))] = true if self.touched else false
            controller[Vector((*self._prefix, "touched", "beat"))] = Vector((self._touched_beat,))
        return changed

    def on_touched(self, beat):
        self.touched = True
        self._touched_beat = beat
        self._changed = True

    def on_released(self, beat):
        self.touched = False
        self._touched_beat = beat
        self._changed = True


class Pad(TouchControl):
    DEFAULT_THRESHOLD = 0.4

    def __init__(self, number):
        super().__init__(number)
        self._prefix = None
        self.toggled = None
        self._toggled_beat = None
        self.pressure = None
        self._pressure_beat = None
        self._toggle_threshold = self.DEFAULT_THRESHOLD
        self._can_toggle = False

    def update(self, node, controller):
        self._prefix = node['state'] if 'state' in node else Vector(("pad", *self.number))
        changed = super().update(node, controller)
        self._toggle_threshold = node.get('threshold', 1, float, self.DEFAULT_THRESHOLD)
        if self.pressure is not None:
            controller[Vector((*self._prefix, "pressure"))] = Vector((self.pressure,))
            controller[Vector((*self._prefix, "pressure", "beat"))] = Vector((self._pressure_beat,))
        if self.toggled is not None:
            controller[Vector((*self._prefix, "toggled"))] = true if self.toggled else false
            controller[Vector((*self._prefix, "toggled", "beat"))] = Vector((self._toggled_beat,))
        return changed

    def on_touched(self, beat):
        super().on_touched(beat)
        self._can_toggle = True

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


class Encoder(TouchControl):
    DEFAULT_LOWER = 0.0
    DEFAULT_UPPER = 1.0

    def __init__(self, number):
        super().__init__(number)
        self._prefix = None
        self.initial = None
        self.lower = None
        self.upper = None
        self.value = None
        self._value_beat = None

    def update(self, node, controller):
        self._prefix = node['state'] if 'state' in node else Vector(("encoder", *self.number))
        changed = super().update(node, controller)
        initial = node.get('initial', 1, float, self.DEFAULT_LOWER)
        if initial != self.initial:
            self.initial = initial
            changed = True
        lower = node.get('lower', 1, float, self.DEFAULT_LOWER)
        if lower != self.lower:
            self.lower = lower
            changed = True
        upper = node.get('upper', 1, float, self.DEFAULT_UPPER)
        if upper != self.upper:
            self.upper = upper
            changed = True
        value_key = Vector((*self._prefix, "value"))
        value_beat_key = Vector((*self._prefix, "value", "beat"))
        if self.value is None:
            if value_key in controller:
                self.value = controller[value_key][0]
                self._value_beat = controller[value_beat_key][0]
            else:
                self.value = self.initial
                self._value_beat = controller.counter.beat
        controller[value_key] = Vector((self.value,))
        controller[value_beat_key] = Vector((self._value_beat,))
        return changed

    def on_turned(self, beat, amount):
        self.value = min(max(self.lower, self.value + amount * (self.upper - self.lower)), self.upper)
        self._value_beat = beat
        self._changed = True
