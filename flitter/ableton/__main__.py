"""
Ableton Push API test
"""

# pylama:ignore=W0601,C0103

import argparse
import asyncio
import logging
import sys

from .constants import Animation, Encoder, Control, BUTTONS, COLOR_BUTTONS
from .events import ButtonPressed, ButtonReleased, TouchStripDragged, PadPressed, PadReleased, EncoderTurned
from .push import Push


async def test():
    push = Push()
    push.start()
    for n in range(64):
        push.set_pad_color(n, 0, 0, 255)
        push.set_pad_color(n, 255, 0, 0, Animation.BLINKING_HALF)
    for n in range(16):
        push.set_menu_button_color(n, 0, 0, 255)
    for n in BUTTONS:
        if n in COLOR_BUTTONS:
            push.set_button_color(n, 255, 0, 0)
        else:
            push.set_button_white(n, 255)
    brightness = 1
    push.set_led_brightness(brightness)
    push.set_display_brightness(brightness)
    push.set_touch_strip_position(0)
    shift_pressed = False
    tap_tempo = False
    taps = []
    while True:
        with push.screen_context() as ctx:
            ctx.set_source_rgb(0, 0, 0.25)
            ctx.paint()
            ctx.set_source_rgb(1, 0, 0)
            ctx.set_font_size(60)
            ctx.move_to(100, 100)
            ctx.show_text("Hello world!")
            ctx.set_source_rgb(1, 1, 1)
            ctx.set_font_size(30)
            ctx.move_to(700, 50)
            ctx.show_text(f"BPM: {push.counter.tempo:5.1f}")
            ctx.move_to(700, 90)
            ctx.show_text(f"Beat: {int(push.counter.beat*10)/10:7.1f}")
            ctx.move_to(700, 130)
            ctx.show_text(f"Phase: {int(push.counter.phase*10)/10:4.1f}")
        event = await push.get_event(1/25)
        if event:
            print(event)
            if isinstance(event, ButtonPressed) and event.number == Control.SHIFT:
                shift_pressed = True
            elif isinstance(event, ButtonReleased) and event.number == Control.SHIFT:
                shift_pressed = False
            elif isinstance(event, TouchStripDragged):
                push.set_touch_strip_position(event.position)
            elif isinstance(event, PadPressed):
                if not tap_tempo:
                    push.set_pad_color(event.number, 0, 255, 0, Animation.ONE_SHOT_24TH)
                else:
                    taps.append(event.timestamp)
            elif isinstance(event, PadReleased):
                if not tap_tempo:
                    push.set_pad_color(event.number, 0, 0, 255)
                    push.set_pad_color(event.number, 255, 0, 0, Animation.BLINKING_HALF)
            elif isinstance(event, EncoderTurned) and event.number == Encoder.TEMPO:
                push.counter.set_tempo((round(push.counter.tempo * 2) + event.amount) / 2, timestamp=event.timestamp)
            elif isinstance(event, ButtonPressed) and event.number == Control.TAP_TEMPO:
                tap_tempo = True
            elif isinstance(event, ButtonReleased) and event.number == Control.TAP_TEMPO:
                if taps:
                    if len(taps) > 1:
                        period = (taps[-1] - taps[0]) / (len(taps) - 1)
                        push.counter.set_tempo(round(120 / period) / 2)
                    push.counter.set_phase(0, timestamp=taps[0], backslip_limit=1)
                taps = []
                tap_tempo = False
            elif isinstance(event, ButtonReleased) and event.number == Control.DOUBLE_LOOP:
                if shift_pressed:
                    if push.counter.quantum % 2 == 0:
                        push.counter.quantum //= 2
                else:
                    push.counter.quantum *= 2
            elif isinstance(event, EncoderTurned) and event.number == Encoder.MASTER:
                brightness = min(max(0, brightness + event.amount / 200), 1)
                push.set_led_brightness(brightness)
                push.set_display_brightness(brightness)


parser = argparse.ArgumentParser(description="Flight Server")
parser.add_argument('--debug', action='store_true', default=False, help="Debug logging")
parser.add_argument('--verbose', action='store_true', default=False, help="Informational logging")
args = parser.parse_args()
logging.basicConfig(level=logging.DEBUG if args.debug else (logging.INFO if args.verbose else logging.WARNING), stream=sys.stderr)
asyncio.run(test())
