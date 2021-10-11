"""
Ableton Push API test
"""

# pylama:ignore=W0601,C0103,R0912,R0915

import argparse
import asyncio
import logging
import sys

from ..clock import TapTempo
from .constants import Animation, Encoder, Control, BUTTONS
from .events import ButtonPressed, ButtonReleased, TouchStripDragged, PadPressed, PadHeld, PadReleased, EncoderTurned, EncoderTouched, EncoderReleased
from .push import Push
from ..interface.osc import OSCSender


osc_sender = OSCSender('localhost', 47178)


async def run():
    push = Push()
    push.start()
    for n in range(64):
        push.set_pad_color(n, 0, 0, 0)
    for n in range(16):
        push.set_menu_button_color(n, 0, 0, 0)
    for n in BUTTONS:
        push.set_button_white(n, 0)
    brightness = 1
    push.set_led_brightness(brightness)
    push.set_display_brightness(brightness)
    push.set_touch_strip_position(0)
    shift_pressed = False
    tap_tempo_pressed = False
    tap_tempo = TapTempo(rounding=1)
    try:
        while True:
            event = await push.get_event(1/60)
            if event:
                print(event)
                if isinstance(event, ButtonPressed) and event.number == Control.SHIFT:
                    shift_pressed = True
                elif isinstance(event, ButtonReleased) and event.number == Control.SHIFT:
                    shift_pressed = False
                elif isinstance(event, TouchStripDragged):
                    push.set_touch_strip_position(event.position)
                elif isinstance(event, PadPressed):
                    if not tap_tempo_pressed:
                        push.set_pad_color(event.number, 0, 255, 0, Animation.ONE_SHOT_SIXTH)
                        address = f'/pad/{event.column}/{event.row}/touched'
                        await osc_sender.send_message(address, push.counter.clock(), event.pressure)
                    else:
                        tap_tempo.tap(event.timestamp)
                elif isinstance(event, PadHeld):
                    if not tap_tempo_pressed:
                        address = f'/pad/{event.column}/{event.row}/held'
                        await osc_sender.send_message(address, push.counter.clock(), event.pressure)
                elif isinstance(event, PadReleased):
                    if not tap_tempo_pressed:
                        push.set_pad_color(event.number, 0, 0, 0, Animation.ONE_SHOT_SIXTH)
                        address = f'/pad/{event.column}/{event.row}/released'
                        await osc_sender.send_message(address, push.counter.clock())
                elif isinstance(event, EncoderTouched) and event.number < 8:
                    address = f'/encoder/{event.number}/touched'
                    await osc_sender.send_message(address, push.counter.clock())
                elif isinstance(event, EncoderTurned) and event.number < 8:
                    address = f'/encoder/{event.number}/turned'
                    await osc_sender.send_message(address, push.counter.clock(), event.amount / 200)
                elif isinstance(event, EncoderReleased) and event.number < 8:
                    address = f'/encoder/{event.number}/released'
                    await osc_sender.send_message(address, push.counter.clock())
                elif isinstance(event, EncoderTurned) and event.number == Encoder.TEMPO:
                    if shift_pressed:
                        push.counter.quantum = max(2, push.counter.quantum + event.amount)
                    else:
                        push.counter.set_tempo((round(push.counter.tempo * 2) + event.amount) / 2, timestamp=event.timestamp)
                elif isinstance(event, ButtonPressed) and event.number == Control.TAP_TEMPO:
                    tap_tempo_pressed = True
                elif isinstance(event, ButtonReleased) and event.number == Control.TAP_TEMPO:
                    tap_tempo.apply(push.counter, event.timestamp, backslip_limit=1)
                    tap_tempo_pressed = False
                elif isinstance(event, EncoderTurned) and event.number == Encoder.MASTER:
                    brightness = min(max(0, brightness + event.amount / 200), 1)
                    push.set_led_brightness(brightness)
                    push.set_display_brightness(brightness)
            else:
                async with push.screen_context() as ctx:
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
    finally:
        for n in range(64):
            push.set_pad_color(n, 0, 0, 0)
        for n in range(16):
            push.set_menu_button_color(n, 0, 0, 0)
        for n in BUTTONS:
            push.set_button_white(n, 0)


parser = argparse.ArgumentParser(description="Flight Server")
parser.add_argument('--debug', action='store_true', default=False, help="Debug logging")
parser.add_argument('--verbose', action='store_true', default=False, help="Informational logging")
args = parser.parse_args()
logging.basicConfig(level=logging.DEBUG if args.debug else (logging.INFO if args.verbose else logging.WARNING), stream=sys.stderr)
asyncio.run(run())
