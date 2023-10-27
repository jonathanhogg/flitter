"""
Flitter controller driver API
"""

import math

from loguru import logger

from ...clock import TapTempo
from ...model import Vector, Node


class Control:
    def __init__(self, driver, control_id):
        self._initialised = False
        self.driver = driver
        self.control_id = control_id
        self.reset()

    def reset(self):
        """Set the control state back to uninitialised/default values."""
        if self._initialised:
            logger.trace("De-initialised {}({!r})", self.__class__.__name__, self.control_id)
        self._initialised = False
        self._state_prefix = None
        self._label = None
        self._color = None
        self._action = None

    def primary_state_value(self):
        raise NotImplementedError()

    def update(self, node: Node, now: float):
        """Update the configuration of the control to match the tree node."""
        changed = False
        action = node.get('action')
        if action is not None:
            action = action[0] if len(action) == 1 else tuple(action)
        if action != self._action:
            self._state_prefix = None
            self._action = action
            changed = True
        elif (state_prefix := node.get('state')) != self._state_prefix:
            self._state_prefix = state_prefix
            changed = True
        if (label := node.get('label', 1, str)) != self._label:
            self._label = label
            changed = True
        if (color := node.get('color', 3, float)) != self._color:
            self._color = color
            changed = True
        if not self._initialised:
            logger.trace("Initialised {}({!r})", self.__class__.__name__, self.control_id)
        self._initialised = True
        return changed

    def update_representation(self):
        """Update the device such that any display/LEDs/whatever match the
           current configuration and state of the control."""
        raise NotImplementedError()

    def update_state(self):
        """Update the program state dictionary and any other engine state
           (such as the beat counter tempo) to match the control state."""
        if self._state_prefix:
            self.driver.engine.state[self._state_prefix] = self.primary_state_value()


class PositionControl(Control):
    DEFAULT_LAG = 1/16

    def reset(self):
        super().reset()
        self._lag = None
        self._lower = None
        self._upper = None
        self._origin = None
        self._raw_position = None
        self._position = None
        self._position_time = None
        self._wrap = None

    @property
    def raw_divisor(self):
        raise NotImplementedError()

    @property
    def position(self):
        if self._position is None:
            return None
        if self._wrap:
            return self._lower + (self._position - self._lower) % (self._upper - self._lower)
        return min(max(self._lower, self._position), self._upper)

    def primary_state_value(self):
        return self.position

    def update(self, node, now):
        changed = super().update(node, now)
        if self._action is not None:
            match self._action:
                case 'tempo':
                    raw_position = self.driver.engine.counter.tempo * 2
                    if raw_position != self._raw_position:
                        self._raw_position = raw_position
                        changed = True
                    self._lag = None
                    self._lower = None
                    self._upper = None
                    self._origin = None
                    self._wrap = None
                    return changed
        if (lag := node.get('lag', 1, float, self.DEFAULT_LAG)) != self._lag:
            self._lag = lag
            changed = True
        if (lower := node.get('lower', 1, float, 0)) != self._lower:
            self._lower = lower
            changed = True
        if (upper := max(self._lower, node.get('upper', 1, float, 1))) != self._upper:
            self._upper = upper
            changed = True
        default_origin = 0 if self._lower < 0 < self._upper else self._lower
        if (origin := min(max(self._lower, node.get('origin', 1, float, default_origin)), self._upper)) != self._origin:
            self._origin = origin
            changed = True
        if (wrap := node.get('wrap', 1, bool, False)) != self._wrap:
            self._wrap = wrap
            changed = True
        if self._raw_position is not None:
            position_range = self._upper - self._lower
            position = self._raw_position / self.raw_divisor * position_range + self._lower
            if self._position is not None and abs(position - self._position) > position_range / 1000:
                delta = self.driver.engine.counter.beat_at_time(now) - self.driver.engine.counter.beat_at_time(self._position_time)
                alpha = math.exp(-delta / self._lag) if self._lag > 0 else 0
                self._position = self._position * alpha + position * (1 - alpha)
                self._position_time = now
                changed = True
            elif self._position != position:
                if self._wrap:
                    self._raw_position = self._raw_position % self.raw_divisor
                    self._position = self._lower + (position - self._lower) % (self._upper - self._lower)
                else:
                    self._position = position
                self._position_time = None
                changed = True
        return changed

    def handle_raw_position_change(self, raw_position, timestamp):
        if not self._initialised:
            return False
        if raw_position != self._raw_position:
            self._raw_position = raw_position
            self._position_time = timestamp
            if self._action == 'tempo' and self._raw_position != self.driver.engine.counter.tempo * 2:
                self.driver.engine.counter.tempo = self._raw_position / 2
            return True
        return False


class SettablePositionControl(PositionControl):
    def reset(self):
        super().reset()
        self._initial = None

    @property
    def initial_raw_position(self):
        if self._initialised:
            position_range = self._upper - self._lower
            position = min(max(self._lower, self._initial), self._upper)
            return (position - self._lower) / position_range * self.raw_divisor if position_range else 0
        return None

    def update(self, node, now):
        changed = super().update(node, now)
        if (initial := node.get('initial', 1, float, self._origin)) != self._initial:
            self._initial = initial
            changed = True
        if self._raw_position is None:
            if self._state_prefix and self._state_prefix in self.driver.engine.state:
                initial = float(self.driver.engine.state[self._state_prefix])
            else:
                initial = self._initial
            self._position = min(max(self._lower, initial), self._upper)
            self._position_time = now
            position_range = self._upper - self._lower
            self._raw_position = (self._position - self._lower) / position_range * self.raw_divisor if position_range else 0
            changed = True
        return changed

    def handle_raw_position_reset(self, timestamp):
        if not self._initialised or self._action is not None:
            return False
        raw_position = self.initial_raw_position
        if self._wrap:
            raw_divisor = self.raw_divisor
            delta = raw_position - self._raw_position
            while delta > raw_divisor//2:
                raw_position -= raw_divisor
                delta -= raw_divisor
            while delta < -raw_divisor//2:
                raw_position += raw_divisor
                delta += raw_divisor
        return self.handle_raw_position_change(raw_position, timestamp)


class EncoderControl(SettablePositionControl):
    STYLES = {'volume', 'pan', 'continuous'}

    def reset(self):
        super().reset()
        self._style = None
        self._turns = None

    @property
    def position(self):
        if self._wrap:
            return self._lower + (self._position - self._lower) % (self._upper - self._lower)
        elif self._style != 'continuous':
            return min(max(self._lower, self._position), self._upper)
        else:
            return self._position

    def update(self, node, now):
        changed = False
        if self._action is None and (turns := node.get('turns', 1, float, 1)) != self._turns:
            self._turns = turns
            changed = True
        if super().update(node, now):
            changed = True
        default_style = 'volume' if self._origin == self._lower else 'pan'
        style = node.get('style', 1, str, default_style).lower()
        if style not in self.STYLES:
            style = default_style
        if style != self._style:
            if self._style == 'continuous':
                raw_position = min(max(0, self._raw_position), self.raw_divisor)
                if raw_position != self._raw_position:
                    self._raw_position = raw_position
                    position_range = self._upper - self._lower
                    self._position = self._raw_position / self.raw_divisor * position_range + self._lower
                    self._position_time = now
            self._style = style
            changed = True
        return changed

    def handle_turn(self, delta, timestamp):
        if not self._initialised or delta == 0:
            return False
        raw_position = self._raw_position + delta
        if self._style != 'continuous' and not self._wrap:
            raw_position = min(max(0, raw_position), self.raw_divisor)
        return self.handle_raw_position_change(raw_position, timestamp)


class ButtonControl(Control):
    ToggleGroups = {}

    def reset(self):
        super().reset()
        self._pushed = None
        self._push_time = None
        self._release_time = None
        self._toggle = None
        self._toggled = None
        self._toggled_changed = None
        self._toggle_time = None
        self._toggle_group = None
        self._action_control = None
        self._action_control_color = None
        self._action_can_trigger = None
        self._action_triggered = None
        self._tap_tempo = None
        self._tap_time = None

    def primary_state_value(self):
        return self._pushed if not self._toggle else self._toggled

    def update(self, node, now):
        changed = super().update(node, now)
        if self._action is not None:
            triggered = self._action_can_trigger and self._action_triggered
            action_control = None
            action_can_trigger = False
            action_control_color = None
            match self._action:
                case 'next':
                    if triggered:
                        self.driver.engine.next_page()
                    return self.driver.engine.has_next_page()
                case 'previous':
                    if triggered:
                        self.driver.engine.previous_page()
                    return self.driver.engine.has_previous_page()
                case 'reset', kind, *control_id:
                    control = self.driver.get_control(kind, Vector(control_id))
                    if control is not None and control._initialised and isinstance(control, SettablePositionControl):
                        action_control = control
                        if triggered:
                            if control.handle_raw_position_reset(now):
                                control.update_representation()
                        action_control_color = control._color
                        action_can_trigger = control._raw_position != control.initial_raw_position
                case 'tap_tempo', kind, *control_id:
                    control = self.driver.get_control(kind, Vector(control_id))
                    if control is not None and control._initialised and isinstance(control, TouchControl):
                        action_control = control
                        if triggered:
                            if self._tap_tempo is None:
                                self._tap_tempo = TapTempo()
                            else:
                                self._tap_tempo.apply(self.driver.engine.counter, now)
                                self._tap_tempo = None
                                self.driver._screen_update_requested = True
                        else:
                            if control._touched_time != self._tap_time:
                                self._tap_time = control._touched_time
                                if self._tap_tempo is not None:
                                    self._tap_tempo.tap(self._tap_time)
                        action_control_color = control._color
                        action_can_trigger = True
                    else:
                        self._tap_tempo = None
                        self._tap_time = None
                case _:
                    self._tap_tempo = None
                    self._tap_time = None
            if action_control != self._action_control:
                self._action_control = action_control
                changed = True
            if action_can_trigger != self._action_can_trigger:
                self._action_can_trigger = action_can_trigger
                changed = True
            if action_control_color != self._action_control_color:
                self._action_control_color = action_control_color
                changed = True
            self._action_triggered = False
            self._toggled = None
            self._toggle_time = None
            self._toggled_changed = None
            return changed
        toggle = node.get('toggle', 1, bool, False)
        group = node.get('group')
        if group is not None:
            group = tuple(group)
        if toggle != self._toggle:
            self._toggle = toggle
            self._toggled = None
            self._toggle_time = None
            self._toggled_changed = None
            changed = True
        if self._toggle and self._toggled is None:
            key = self._state_prefix + ['toggled'] if self._state_prefix else None
            if key and key in self.driver.engine.state:
                self._toggled = bool(self.driver.engine.state[key])
                toggled_beat = float(self.driver.engine.state[key + ['beat']])
                self._toggle_time = self.driver.engine.counter.time_at_beat(toggled_beat)
            else:
                self._toggled = node.get('initial', 1, bool, False)
                self._toggle_time = now
            changed = True
        if group != self._toggle_group:
            if self._toggle_group is not None:
                buttons = self.ToggleGroups[self._toggle_group]
                buttons.remove(self)
                if not buttons:
                    del self.ToggleGroups[self._toggle_group]
            self._toggle_group = group
            self.ToggleGroups.setdefault(self._toggle_group, set()).add(self)
            changed = True
        if self._toggled_changed:
            self._toggled_changed = False
            changed = True
        return changed

    def update_state(self):
        super().update_state()
        if self._state_prefix:
            engine = self.driver.engine
            engine.state[self._state_prefix + ['pushed']] = self._pushed
            engine.state[self._state_prefix + ['released']] = not self._pushed if self._pushed is not None else None
            engine.state[self._state_prefix + ['pushed', 'beat']] = \
                engine.counter.beat_at_time(self._push_time) if self._push_time is not None else None
            engine.state[self._state_prefix + ['released', 'beat']] = \
                engine.counter.beat_at_time(self._release_time) if self._release_time is not None else None
            engine.state[self._state_prefix + ['toggled']] = self._toggled if self._toggle else None
            engine.state[self._state_prefix + ['toggled', 'beat']] = \
                engine.counter.beat_at_time(self._toggle_time) if self._toggle_time is not None else None

    def handle_push(self, pushed, timestamp):
        if not self._initialised or pushed == self._pushed:
            return False
        self._pushed = pushed
        if self._pushed:
            self._push_time = timestamp
            if self._action is not None:
                self._action_triggered = True
            elif self._toggle:
                self._toggled = not self._toggled
                self._toggle_time = timestamp
                if self._toggle_group is not None and self._toggled:
                    for button in self.ToggleGroups[self._toggle_group]:
                        if button is not self and button._toggle and button._toggled:
                            button._toggled = False
                            button._toggle_time = timestamp
                            button._toggled_changed = True
        else:
            self._release_time = timestamp
        return True


class TouchControl(Control):
    def reset(self):
        super().reset()
        self._touched = None
        self._touched_time = None
        self._untouched_time = None

    def update_state(self):
        super().update_state()
        if self._state_prefix:
            engine = self.driver.engine
            engine.state[self._state_prefix + ['touched']] = self._touched
            engine.state[self._state_prefix + ['untouched']] = not self._touched
            engine.state[self._state_prefix + ['touched', 'beat']] = \
                engine.counter.beat_at_time(self._touched_time) if self._touched_time is not None else None
            engine.state[self._state_prefix + ['untouched', 'beat']] = \
                engine.counter.beat_at_time(self._untouched_time) if self._untouched_time is not None else None

    def handle_touch(self, touched, timestamp):
        if not self._initialised or touched == self._touched:
            return False
        self._touched = touched
        if self._touched:
            self._touched_time = timestamp
        else:
            self._untouched_time = timestamp
        return True


class PressureControl(PositionControl, TouchControl, ButtonControl):
    def reset(self):
        super().reset()
        self._raw_threshold = None

    @property
    def default_raw_threshold(self):
        return self.raw_divisor // 2

    def update(self, node, now):
        if self._raw_position is None:
            self._raw_position = 0
        changed = super().update(node, now)
        if (threshold := node.get('threshold', 1, float)) is not None:
            position_range = self._upper - self._lower
            raw_threshold = (threshold - self._lower) / position_range * self.raw_divisor
        else:
            raw_threshold = self.default_raw_threshold
        if raw_threshold != self._raw_threshold:
            self._raw_threshold = raw_threshold
            changed = True
        return changed

    def handle_pressure(self, raw_position, timestamp):
        if not self._initialised or raw_position == self._raw_position:
            return False
        self._raw_position = raw_position
        changed = self.handle_push(self._raw_position >= self._raw_threshold, timestamp)
        changed = self.handle_touch(self._raw_position > 0, timestamp) or changed
        return changed


class ControllerDriver:
    def __init__(self, engine):
        self.engine = engine

    def get_default_config(self) -> list[Node]:
        """Return a list of nodes that specify a default configuration."""
        raise NotImplementedError()

    async def start(self):
        """Start the driver."""
        raise NotImplementedError()

    async def stop(self):
        """Stop the driver (or schedule it to stop)."""
        raise NotImplementedError()

    async def start_update(self, node: Node):
        """Start an update pass."""
        pass

    def get_control(self, kind: str, control_id: Vector) -> Control:
        """Return a control matching the node `kind` and `id` attribute."""
        raise NotImplementedError()

    def handle_node(self, node: Node) -> bool:
        """A second chance to handle nodes that don't appear to be controls.
           Return `False` if not handled, `True` otherwise."""
        return False

    async def finish_update(self):
        """Finish an update pass."""
        pass
