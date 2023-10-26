"""
Ableton Push (and MIDI) constants
"""

import enum


class MIDI(enum.IntEnum):
    NOTE_OFF = 0x80
    NOTE_ON = 0x90
    POLYPHONIC_PRESSURE = 0xA0
    CONTROL_CHANGE = 0xB0
    CHANNEL_PRESSURE = 0xD0
    PITCH_BEND_CHANGE = 0xE0
    START_OF_SYSEX = 0xF0
    END_OF_SYSEX = 0xF7
    CLOCK = 0xF8
    START = 0xFA
    CONTINUE = 0xFB
    STOP = 0xFC
    ACTIVE = 0xFE
    RESET = 0xFF


class Command(enum.IntEnum):
    SET_COLOR_PALETTE_ENTRY = 0x03
    REAPPLY_COLOR_PALETTE = 0x05
    SET_LED_BRIGHTNESS = 0x06
    SET_DISPLAY_BRIGHTNESS = 0x08
    SET_MIDI_MODE = 0x0A
    SET_TOUCHSTRIP_CONFIG = 0x17
    SET_AFTERTOUCH_MODE = 0x1E


class Animation(enum.IntEnum):
    NONE = 0
    ONE_SHOT_SIXTH = 1
    ONE_SHOT_QUARTER = 2
    ONE_SHOT_HALF = 3
    ONE_SHOT_ONE = 4
    ONE_SHOT_TWO = 5
    PULSING_SIXTH = 6
    PULSING_QUARTER = 7
    PULSING_HALF = 8
    PULSING_ONE = 9
    PULSING_TWO = 10
    BLINKING_SIXTH = 11
    BLINKING_QUARTER = 12
    BLINKING_HALF = 13
    BLINKING_ONE = 14
    BLINKING_TWO = 15


class Note(enum.IntEnum):
    ENCODER_0 = 0
    ENCODER_1 = 1
    ENCODER_2 = 2
    ENCODER_3 = 3
    ENCODER_4 = 4
    ENCODER_5 = 5
    ENCODER_6 = 6
    ENCODER_7 = 7
    ENCODER_MASTER = 8
    ENCODER_METRONOME = 9
    ENCODER_TEMPO = 10
    TOUCH_STRIP = 12
    PAD_0_0 = 36
    PAD_0_1 = 37
    PAD_0_2 = 38
    PAD_0_3 = 39
    PAD_0_4 = 40
    PAD_0_5 = 41
    PAD_0_6 = 42
    PAD_0_7 = 43
    PAD_1_0 = 44
    PAD_1_1 = 45
    PAD_1_2 = 46
    PAD_1_3 = 47
    PAD_1_4 = 48
    PAD_1_5 = 49
    PAD_1_6 = 50
    PAD_1_7 = 51
    PAD_2_0 = 52
    PAD_2_1 = 53
    PAD_2_2 = 54
    PAD_2_3 = 55
    PAD_2_4 = 56
    PAD_2_5 = 57
    PAD_2_6 = 58
    PAD_2_7 = 59
    PAD_3_0 = 60
    PAD_3_1 = 61
    PAD_3_2 = 62
    PAD_3_3 = 63
    PAD_3_4 = 64
    PAD_3_5 = 65
    PAD_3_6 = 66
    PAD_3_7 = 67
    PAD_4_0 = 68
    PAD_4_1 = 69
    PAD_4_2 = 70
    PAD_4_3 = 71
    PAD_4_4 = 72
    PAD_4_5 = 73
    PAD_4_6 = 74
    PAD_4_7 = 75
    PAD_5_0 = 76
    PAD_5_1 = 77
    PAD_5_2 = 78
    PAD_5_3 = 79
    PAD_5_4 = 80
    PAD_5_5 = 81
    PAD_5_6 = 82
    PAD_5_7 = 83
    PAD_6_0 = 84
    PAD_6_1 = 85
    PAD_6_2 = 86
    PAD_6_3 = 87
    PAD_6_4 = 88
    PAD_6_5 = 89
    PAD_6_6 = 90
    PAD_6_7 = 91
    PAD_7_0 = 92
    PAD_7_1 = 93
    PAD_7_2 = 94
    PAD_7_3 = 95
    PAD_7_4 = 96
    PAD_7_5 = 97
    PAD_7_6 = 98
    PAD_7_7 = 99


class Control(enum.IntEnum):
    TAP_TEMPO = 3
    METRONOME = 9
    ENCODER_TEMPO = 14
    ENCODER_METRONOME = 15
    MENU_0_0 = 20
    MENU_0_1 = 21
    MENU_0_2 = 22
    MENU_0_3 = 23
    MENU_0_4 = 24
    MENU_0_5 = 25
    MENU_0_6 = 26
    MENU_0_7 = 27
    MASTER = 28
    STOP_CLIP = 29
    SETUP = 30
    LAYOUT = 31
    CONVERT = 35
    ONE_4 = 36
    ONE_4_T = 37
    ONE_8 = 38
    ONE_8_T = 39
    ONE_16 = 40
    ONE_16_T = 41
    ONE_32 = 42
    ONE_32_T = 43
    LEFT = 44
    RIGHT = 45
    UP = 46
    DOWN = 47
    SELECT = 48
    SHIFT = 49
    NOTE = 50
    SESSION = 51
    ADD_DEVICE = 52
    ADD_TRACK = 53
    OCTAVE_DOWN = 54
    OCTAVE_UP = 55
    REPEAT = 56
    ACCENT = 57
    SCALE = 58
    USER = 59
    MUTE = 60
    SOLO = 61
    PAGE_LEFT = 62
    PAGE_RIGHT = 63
    ENCODER_0 = 71
    ENCODER_1 = 72
    ENCODER_2 = 73
    ENCODER_3 = 74
    ENCODER_4 = 75
    ENCODER_5 = 76
    ENCODER_6 = 77
    ENCODER_7 = 78
    ENCODER_SETUP = 79
    PLAY = 85
    RECORD = 86
    NEW = 87
    DUPLICATE = 88
    AUTOMATE = 89
    FIXED_LENGTH = 90
    MENU_1_0 = 102
    MENU_1_1 = 103
    MENU_1_2 = 104
    MENU_1_3 = 105
    MENU_1_4 = 106
    MENU_1_5 = 107
    MENU_1_6 = 108
    MENU_1_7 = 109
    DEVICE = 110
    BROWSE = 111
    MIX = 112
    CLIP = 113
    QUANTIZE = 116
    DOUBLE_LOOP = 117
    DELETE = 118
    UNDO = 119


class Encoder(enum.IntEnum):
    ZERO = 0
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    SETUP = 8
    METRONOME = 9
    TEMPO = 10


class TouchStripFlags(enum.IntFlag):
    PUSH2_CONTROL = 0b00000000
    HOST_CONTROL = 0b00000001
    SEND_VALUES = 0b00000000
    SEND_SYSEX = 0b00000010
    PITCH_BEND = 0b00000000
    MOD_WHEEL = 0b00000100
    LEDS_BAR = 0b00000000
    LEDS_POINT = 0b00001000
    BAR_BOTTOM = 0b00000000
    BAR_CENTER = 0b00010000
    NO_AUTO_RETURN = 0b00000000
    AUTO_RETURN = 0b00100000
    RETURN_BOTTOM = 0b00000000
    RETURN_CENTER = 0b01000000


BUTTONS = frozenset({Control.TAP_TEMPO, Control.METRONOME, Control.SETUP, Control.USER, Control.DELETE, Control.UNDO, Control.ADD_DEVICE,
                     Control.ADD_TRACK, Control.DEVICE, Control.BROWSE, Control.MIX, Control.CLIP, Control.MUTE, Control.SOLO, Control.STOP_CLIP,
                     Control.MASTER, Control.ONE_4, Control.ONE_4_T, Control.ONE_8, Control.ONE_8_T, Control.ONE_16, Control.ONE_16_T,
                     Control.ONE_32, Control.ONE_32_T, Control.CONVERT, Control.DOUBLE_LOOP, Control.QUANTIZE, Control.DUPLICATE, Control.NEW,
                     Control.FIXED_LENGTH, Control.AUTOMATE, Control.RECORD, Control.PLAY, Control.LEFT, Control.RIGHT, Control.UP, Control.DOWN,
                     Control.REPEAT, Control.ACCENT, Control.SCALE, Control.LAYOUT, Control.NOTE, Control.SESSION, Control.OCTAVE_DOWN, Control.OCTAVE_UP,
                     Control.PAGE_LEFT, Control.PAGE_RIGHT, Control.SELECT, Control.SHIFT,
                     Control.MENU_0_0, Control.MENU_0_1, Control.MENU_0_2, Control.MENU_0_3,
                     Control.MENU_0_4, Control.MENU_0_5, Control.MENU_0_6, Control.MENU_0_7,
                     Control.MENU_1_0, Control.MENU_1_1, Control.MENU_1_2, Control.MENU_1_3,
                     Control.MENU_1_4, Control.MENU_1_5, Control.MENU_1_6, Control.MENU_1_7})

COLOR_BUTTONS = frozenset({Control.MUTE, Control.SOLO, Control.STOP_CLIP, Control.AUTOMATE, Control.RECORD, Control.PLAY,
                           Control.ONE_4, Control.ONE_4_T, Control.ONE_8, Control.ONE_8_T,
                           Control.ONE_16, Control.ONE_16_T, Control.ONE_32, Control.ONE_32_T,
                           Control.MENU_0_0, Control.MENU_0_1, Control.MENU_0_2, Control.MENU_0_3,
                           Control.MENU_0_4, Control.MENU_0_5, Control.MENU_0_6, Control.MENU_0_7,
                           Control.MENU_1_0, Control.MENU_1_1, Control.MENU_1_2, Control.MENU_1_3,
                           Control.MENU_1_4, Control.MENU_1_5, Control.MENU_1_6, Control.MENU_1_7})

ENCODER_CONTROLS = {Control.ENCODER_0: Encoder.ZERO, Control.ENCODER_1: Encoder.ONE, Control.ENCODER_2: Encoder.TWO, Control.ENCODER_3: Encoder.THREE,
                    Control.ENCODER_4: Encoder.FOUR, Control.ENCODER_5: Encoder.FIVE, Control.ENCODER_6: Encoder.SIX, Control.ENCODER_7: Encoder.SEVEN,
                    Control.ENCODER_SETUP: Encoder.SETUP, Control.ENCODER_METRONOME: Encoder.METRONOME, Control.ENCODER_TEMPO: Encoder.TEMPO}
