"""
Ableton Push event classes
"""

from dataclasses import dataclass

from .constants import Control, Encoder


__all__ = ['PadPressed', 'PadHeld', 'PadReleased',
           'ButtonPressed', 'ButtonReleased',
           'EncoderTouched', 'EncoderTurned', 'EncoderReleased', 'TouchStripTouched',
           'TouchStripDragged', 'TouchStripReleased']


@dataclass
class PushEvent:
    timestamp: float


@dataclass
class PadEvent(PushEvent):
    number: int
    column: int
    row: int


@dataclass
class PadHeld(PadEvent):
    pressure: float


@dataclass
class PadPressed(PadHeld):
    pass


@dataclass
class PadReleased(PadEvent):
    pass


@dataclass
class ButtonEvent(PushEvent):
    number: Control


@dataclass
class ButtonPressed(ButtonEvent):
    pass


@dataclass
class ButtonReleased(ButtonEvent):
    pass


@dataclass
class EncoderEvent(PushEvent):
    number: Encoder


@dataclass
class EncoderTouched(EncoderEvent):
    pass


@dataclass
class EncoderTurned(EncoderEvent):
    amount: int


@dataclass
class EncoderReleased(EncoderEvent):
    pass


@dataclass
class TouchStripEvent(PushEvent):
    pass


@dataclass
class TouchStripTouched(TouchStripEvent):
    pass


@dataclass
class TouchStripDragged(TouchStripEvent):
    position: float


@dataclass
class TouchStripReleased(TouchStripEvent):
    pass
