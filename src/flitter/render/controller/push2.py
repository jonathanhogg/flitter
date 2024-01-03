"""
Ableton Push 2 controller driver
"""

import asyncio

from loguru import logger
import math
import skia

from . import driver
from ...ableton.constants import Encoder, BUTTONS
from ...ableton.events import (ButtonPressed, ButtonReleased,
                               PadPressed, PadHeld, PadReleased,
                               EncoderTurned, EncoderTouched, EncoderReleased,
                               TouchStripTouched, TouchStripDragged, TouchStripReleased)
from ...ableton.push import Push2, Push2CommunicationError
from ...ableton.palette import SimplePalette, PrimaryPalette, HuePalette
from ...clock import system_clock
from ...model import Vector, Node
from ..window.canvas import draw


DEFAULT_CONFIG = [
    Node('rotary', attributes={'id': Vector('tempo'), 'style': Vector('continuous'), 'action': Vector('tempo')}),
    Node('button', attributes={'id': Vector('page_left'), 'action': Vector('previous')}),
    Node('button', attributes={'id': Vector('page_right'), 'action': Vector('next')}),
    Node('button', attributes={'id': Vector('tap_tempo'), 'action': Vector(['tap_tempo', 'rotary', 'tempo'])}),
    Node('button', attributes={'id': Vector('metronome'), 'action': Vector(['reset', 'rotary', 'metronome'])}),
    Node('button', attributes={'id': Vector('menu_1_0'), 'action': Vector(['reset', 'rotary', 1])}),
    Node('button', attributes={'id': Vector('menu_1_1'), 'action': Vector(['reset', 'rotary', 2])}),
    Node('button', attributes={'id': Vector('menu_1_2'), 'action': Vector(['reset', 'rotary', 3])}),
    Node('button', attributes={'id': Vector('menu_1_3'), 'action': Vector(['reset', 'rotary', 4])}),
    Node('button', attributes={'id': Vector('menu_1_4'), 'action': Vector(['reset', 'rotary', 5])}),
    Node('button', attributes={'id': Vector('menu_1_5'), 'action': Vector(['reset', 'rotary', 6])}),
    Node('button', attributes={'id': Vector('menu_1_6'), 'action': Vector(['reset', 'rotary', 7])}),
    Node('button', attributes={'id': Vector('menu_1_7'), 'action': Vector(['reset', 'rotary', 8])}),
]

PALETTES = {
    'simple': SimplePalette,
    'primary': PrimaryPalette,
    'hue': HuePalette,
}

PAD_NUMBER_ID_MAPPING = {number: Vector([8 - number // 8, number % 8 + 1]) for number in range(64)}

ROTARY_ID_MAPPING = {
    Encoder.TEMPO: Vector('tempo'),
    Encoder.METRONOME: Vector('metronome'),
    Encoder.ZERO: Vector(1),
    Encoder.ONE: Vector(2),
    Encoder.TWO: Vector(3),
    Encoder.THREE: Vector(4),
    Encoder.FOUR: Vector(5),
    Encoder.FIVE: Vector(6),
    Encoder.SIX: Vector(7),
    Encoder.SEVEN: Vector(8),
    Encoder.SETUP: Vector('setup'),
}

SCREEN_ROTARIES = {
    Encoder.ZERO: 0,
    Encoder.ONE: 1,
    Encoder.TWO: 2,
    Encoder.THREE: 3,
    Encoder.FOUR: 4,
    Encoder.FIVE: 5,
    Encoder.SIX: 6,
    Encoder.SEVEN: 7,
}

BUTTON_ID_MAPPING = {button: Vector(button.name.lower()) for button in BUTTONS}


def get_driver_class():
    return Push2Driver


class Push2RotaryControl(driver.TouchControl, driver.EncoderControl):
    DEFAULT_DECIMALS = 1

    def __init__(self, driver, control_id, number):
        super().__init__(driver, control_id)
        self._number = number

    def reset(self):
        super().reset()
        self._decimals = None
        self._percent = None
        self._degrees = None
        self._units = None

    @property
    def raw_divisor(self):
        steps = 18 if self.control_id == Vector('tempo') else 210
        if self._turns is not None:
            steps = int(steps * self._turns)
        return steps

    @property
    def color(self):
        if not self._initialised:
            return 0, 0, 0
        if self._color is None:
            red = green = blue = 1
        else:
            red, green, blue = self._color
        brightness = 1 if self._touched else 0.5
        return red*brightness, green*brightness, blue*brightness

    def update(self, node, now):
        changed = super().update(node, now)
        if (decimals := node.get('decimals', 1, int, self.DEFAULT_DECIMALS)) != self._decimals:
            self._decimals = decimals
            changed = True
        if (percent := node.get('percent', 1, bool, False)) != self._percent:
            self._percent = percent
            changed = True
        if (degrees := node.get('degrees', 1, bool, False)) != self._degrees:
            self._degrees = degrees
            changed = True
        if (units := node.get('units', 1, str)) != self._units:
            self._units = units
            changed = True
        return changed

    def update_representation(self):
        if self.driver._push2 is None:
            return
        number = SCREEN_ROTARIES.get(self._number)
        if number is not None:
            self.driver._screen_update_requested = True
        else:
            self.driver._screen_update_requested = self._action == 'tempo'


class Push2ButtonControl(driver.ButtonControl):
    def __init__(self, driver, control_id, number):
        super().__init__(driver, control_id)
        self._number = number

    @property
    def color(self):
        if not self._initialised or (self._action is not None and self._action_control is None):
            return 0, 0, 0
        color = self._color or self._action_control_color
        if color is None:
            red = green = blue = 1
        else:
            red, green, blue = color
        brightness = 0.5
        if self._pushed:
            brightness = 1
        elif self._action and isinstance(self._action, tuple) and self._action[0] == 'tap_tempo':
            if self._tap_tempo is not None:
                brightness = 1
        elif self._action_can_trigger:
            brightness = 1
        return red*brightness, green*brightness, blue*brightness

    def update_representation(self):
        if self.driver._push2 is None:
            return
        self.driver._push2.set_button_rgb(self._number, *self.color)


class Push2PadControl(driver.PressureControl):
    def __init__(self, driver, control_id, number):
        super().__init__(driver, control_id)
        self._number = number

    @property
    def default_raw_threshold(self):
        return 32

    @property
    def raw_divisor(self):
        return 127

    @property
    def color(self):
        if not self._initialised or (self._action is not None and self._action_control is None):
            return 0, 0, 0
        color = self._color or self._action_control_color
        if color is None:
            red = green = blue = 1
        else:
            red, green, blue = color
        brightness = 0.5
        if self._pushed or self._touched or self._toggled:
            brightness = 1
        elif self._action and isinstance(self._action, tuple) and self._action[0] == 'tap_tempo':
            if self._tap_tempo is not None:
                brightness = 1
        elif self._action_can_trigger:
            brightness = 1
        return red*brightness, green*brightness, blue*brightness

    def update_representation(self):
        if self.driver._push2 is None:
            return
        self.driver._push2.set_pad_rgb(self._number, *self.color)


class Push2SliderControl(driver.TouchControl, driver.SettablePositionControl):
    def reset(self):
        super().reset()
        self._return = None

    @property
    def raw_divisor(self):
        return 1 << 14

    def update(self, node, now):
        changed = super().update(node, now)
        if (return_ := node.get('return', 1, bool, False)) != self._return:
            self._return = return_
            changed = True
        return changed

    def update_representation(self):
        if self.driver._push2 is None:
            return
        if self._position is not None:
            position = (self._position - self._lower) / (self._upper - self._lower)
            self.driver._push2.set_touch_strip_position(position)

    def handle_touch(self, touched, timestamp):
        if self._return and self._touched is True and touched is False:
            changed = self.handle_raw_position_reset(timestamp)
        else:
            changed = False
        return super().handle_touch(touched, timestamp) or changed


class Push2Driver(driver.ControllerDriver):
    MAX_SCREEN_REFRESH_PERIOD = 1/30

    def __init__(self, engine):
        super().__init__(engine)
        self._push2 = None
        self._rotaries = {}
        for number, control_id in ROTARY_ID_MAPPING.items():
            self._rotaries[control_id] = Push2RotaryControl(self, control_id, number)
        self._buttons = {}
        for number, control_id in BUTTON_ID_MAPPING.items():
            self._buttons[control_id] = Push2ButtonControl(self, control_id, number)
        self._pads = {}
        for number, control_id in PAD_NUMBER_ID_MAPPING.items():
            self._pads[control_id] = Push2PadControl(self, control_id, number)
        self._slider = Push2SliderControl(self, Vector('main'))
        self._run_task = None
        self._screen_update_requested = True
        self._screen_canvas_node = None
        self._last_screen_update = None
        self._last_update_was_canvas = None

    async def start(self):
        self._run_task = asyncio.create_task(self.run())

    async def stop(self):
        self._run_task.cancel()
        await self._run_task
        self._run_task = None

    def get_default_config(self):
        return DEFAULT_CONFIG

    async def start_update(self, node):
        self._screen_canvas_node = None
        if self._push2 is not None:
            palette = node.get('palette', 1, str, 'hue').lower()
            palette_class = PALETTES.get(palette, HuePalette)
            if not isinstance(self._push2.palette, palette_class):
                self._push2.palette = palette_class()
        else:
            self._screen_update_requested = True
            self._last_screen_update = None
            self._last_update_was_canvas = None

    def get_control(self, kind, control_id):
        if kind == 'rotary':
            return self._rotaries.get(control_id)
        if kind == 'button':
            return self._buttons.get(control_id)
        if kind == 'slider' and control_id == Vector('main'):
            return self._slider
        if kind == 'pad':
            return self._pads.get(control_id)

    def handle_node(self, node):
        if node.kind == 'screen':
            self._screen_canvas_node = node
            return True
        return False

    async def finish_update(self):
        if self._push2 is not None:
            try:
                now = system_clock()
                if self._last_screen_update is None or now >= self._last_screen_update + self.MAX_SCREEN_REFRESH_PERIOD:
                    self._last_screen_update = now
                    if self._screen_canvas_node is not None:
                        await self.draw_screen_canvas()
                        self._last_update_was_canvas = True
                    elif self._screen_update_requested or self._last_update_was_canvas:
                        await self.update_screen()
                        self._last_update_was_canvas = False
                    self._screen_update_requested = False
            except Exception as exc:
                logger.warning("Unable to update Push2 screen: {}", str(exc))

    def request_screen_update(self):
        self._screen_update_requested = True

    async def refresh(self):
        for rotary in self._rotaries.values():
            rotary.update_representation()
        for button in self._buttons.values():
            button.update_representation()
        self._slider.update_representation()
        for pad in self._pads.values():
            pad.update_representation()
        self._screen_update_requested = True

    async def run(self):
        while True:
            try:
                while not Push2.test_presence():
                    await asyncio.sleep(1)
                self._push2 = Push2(palette=HuePalette())
                await self._push2.start(counter=self.engine.counter)
                await self.refresh()
                logger.debug("Ableton Push2 controller driver ready")
                while True:
                    try:
                        event = await asyncio.wait_for(self._push2.wait_event(), timeout=1)
                    except asyncio.TimeoutError:
                        await self.refresh()
                        continue
                    updates = set()
                    while event is not None:
                        match event:
                            case PadPressed(timestamp=timestamp, number=number, pressure=pressure):
                                pad = self._pads[PAD_NUMBER_ID_MAPPING[number]]
                                if pad.handle_pressure(pressure, timestamp):
                                    updates.add(pad)
                            case PadHeld(timestamp=timestamp, number=number, pressure=pressure):
                                pad = self._pads[PAD_NUMBER_ID_MAPPING[number]]
                                if pad.handle_pressure(pressure, timestamp):
                                    updates.add(pad)
                            case PadReleased(timestamp=timestamp, number=number):
                                pad = self._pads[PAD_NUMBER_ID_MAPPING[number]]
                                if pad.handle_pressure(0, timestamp):
                                    updates.add(pad)
                            case EncoderTouched(timestamp=timestamp, number=number):
                                rotary = self._rotaries[ROTARY_ID_MAPPING[number]]
                                if rotary.handle_touch(True, timestamp):
                                    updates.add(rotary)
                            case EncoderTurned(timestamp=timestamp, number=number, amount=amount):
                                rotary = self._rotaries[ROTARY_ID_MAPPING[number]]
                                if rotary.handle_turn(amount, timestamp):
                                    updates.add(rotary)
                            case EncoderReleased(timestamp=timestamp, number=number):
                                rotary = self._rotaries[ROTARY_ID_MAPPING[number]]
                                if rotary.handle_touch(False, timestamp):
                                    updates.add(rotary)
                            case ButtonPressed(timestamp=timestamp, number=number):
                                button = self._buttons[BUTTON_ID_MAPPING[number]]
                                if button.handle_push(True, timestamp):
                                    updates.add(button)
                            case ButtonReleased(timestamp=timestamp, number=number):
                                button = self._buttons[BUTTON_ID_MAPPING[number]]
                                if button.handle_push(False, timestamp):
                                    updates.add(button)
                            case TouchStripTouched(timestamp=timestamp):
                                if self._slider.handle_touch(True, timestamp):
                                    updates.add(self._slider)
                            case TouchStripDragged(timestamp=timestamp, position=position):
                                if self._slider.handle_raw_position_change(position, timestamp):
                                    updates.add(self._slider)
                            case TouchStripReleased(timestamp=timestamp):
                                if self._slider.handle_touch(False, timestamp):
                                    updates.add(self._slider)
                            case _:
                                logger.warning("Unhandled Push2 event: {}", event)
                        event = self._push2.get_event()
                    for control in updates:
                        control.update_representation()
            except asyncio.CancelledError:
                if self._push2 is not None:
                    await self._push2.stop()
                    self._push2 = None
                    logger.debug("Ableton Push2 controller driver stopped")
                return
            except Push2CommunicationError:
                logger.warning("Lost contact with Ableton Push2 device")
                try:
                    self._push2.stop()
                except Exception:
                    pass
                self._push2 = None
            except Exception:
                logger.exception("Unexpected error in Ableton Push2 driver")
                if self._push2 is not None:
                    try:
                        self._push2.stop()
                    except Exception:
                        pass
                    self._push2 = None

    async def draw_screen_canvas(self):
        async with self._push2.screen_context() as ctx:
            ctx.clear(skia.ColorBLACK)
            draw(self._screen_canvas_node, ctx)

    async def update_screen(self):
        async with self._push2.screen_context() as ctx:
            ctx.clear(skia.ColorBLACK)
            paint = skia.Paint(AntiAlias=True)
            white_paint = skia.Paint(Color=skia.ColorWHITE, AntiAlias=True)
            black_paint = skia.Paint(Color=skia.ColorBLACK, AntiAlias=True)
            font = skia.Font(skia.Typeface("helvetica"), 16)
            ctx.drawSimpleText(f"Tempo: {self.engine.counter.tempo:5.1f}", 10, 150, font, white_paint)
            ctx.drawSimpleText(f"Quantum: {self.engine.counter.quantum}", 130, 150, font, white_paint)
            for control_id, rotary in self._rotaries.items():
                number = SCREEN_ROTARIES.get(rotary._number)
                if number is None or not rotary._initialised:
                    continue
                ctx.save()
                ctx.translate(120 * number, 0)
                paint.setColor4f(skia.Color4f(*rotary.color, 1))
                raw = rotary._raw_position / rotary.raw_divisor
                origin = (rotary._origin - rotary._lower) / (rotary._upper - rotary._lower)
                if rotary._style == 'continuous':
                    position = (rotary._position - rotary._lower) / (rotary._upper - rotary._lower)
                    offset = -90
                    scale = 360
                else:
                    raw = raw % 1 if rotary._wrap else min(max(0, raw), 1)
                    position = (rotary.position - rotary._lower) / (rotary._upper - rotary._lower)
                    offset = -240
                    scale = 300
                path = skia.Path()
                paint.setStyle(skia.Paint.kStroke_Style)
                paint.setStrokeWidth(2)
                path.addArc(skia.Rect.MakeXYWH(20, 40, 80, 80), offset, scale)
                ctx.drawPath(path, paint)
                if rotary._style == 'continuous':
                    nicks = [origin]
                elif rotary._style == 'pan':
                    nicks = [0, 1]
                else:
                    nicks = [1]
                paint.setStrokeWidth(2)
                paint.setStyle(skia.Paint.kStroke_Style)
                path = skia.Path()
                path.moveTo(30, 0)
                path.lineTo(40, 0)
                for nick in nicks:
                    ctx.save()
                    ctx.translate(60, 80)
                    ctx.rotate(offset + nick*scale)
                    ctx.drawPath(path, paint)
                    ctx.restore()
                path = skia.Path()
                paint.setStrokeWidth(10)
                if rotary._style == 'continuous':
                    arc = position - raw
                    if arc < -1 or arc > 1:
                        path.addOval(skia.Rect.MakeXYWH(25, 45, 70, 70))
                    else:
                        path.addArc(skia.Rect.MakeXYWH(25, 45, 70, 70), offset + raw*scale, arc*scale)
                elif rotary._style == 'pan':
                    path.addArc(skia.Rect.MakeXYWH(25, 45, 70, 70), offset + origin*scale, (position - origin)*scale)
                else:
                    path.addArc(skia.Rect.MakeXYWH(25, 45, 70, 70), offset, position*scale)
                ctx.drawPath(path, paint)
                ctx.save()
                ctx.translate(60, 80)
                ctx.rotate(offset + raw*scale)
                path = skia.Path()
                path.moveTo(48, 0)
                path.lineTo(26, 6)
                path.lineTo(26, -6)
                path.close()
                ctx.drawPath(path, white_paint)
                black_paint.setStrokeWidth(2)
                black_paint.setStyle(skia.Paint.kStroke_Style)
                ctx.drawPath(path, black_paint)
                ctx.restore()
                paint.setStyle(skia.Paint.kFill_Style)
                font.setSize(14)
                value = rotary.position
                if rotary._percent:
                    value *= 100
                elif rotary._degrees:
                    value *= 360
                exponent = 10**rotary._decimals
                value = int(value * exponent) / exponent
                text = f'{{:.{max(0, int(math.ceil(rotary._decimals)))}f}}'.format(value)
                if rotary._percent:
                    text += '%'
                elif rotary._degrees:
                    text += '°'
                width = font.measureText(text)
                ctx.drawString(text, (120 - width) / 2, 84, font, white_paint)
                text = rotary._units
                if text:
                    font.setSize(10)
                    width = font.measureText(text)
                    ctx.drawString(text, (120 - width) / 2, 98, font, white_paint)
                text = rotary._label
                if text:
                    path = skia.Path()
                    path.addRect(2, 2, 116, 28)
                    paint.setStyle(skia.Paint.kFill_Style)
                    ctx.drawPath(path, paint)
                    font.setSize(16)
                    width = font.measureText(text)
                    black_paint.setStyle(skia.Paint.kStroke_Style)
                    black_paint.setStrokeWidth(5)
                    ctx.drawString(text, (120 - width) / 2, 21, font, black_paint)
                    ctx.drawString(text, (120 - width) / 2, 21, font, white_paint)
                ctx.restore()
