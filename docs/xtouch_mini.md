
# Behringer X-Touch mini controller

Enable MC mode for full control: send CC+11 127 1.

# MIDI mapping for user interaction

All buttons and encoders are channel 0.

|---------|---------|---------|---------|---------|---------|---------|---------|---------|
|Encoder 0|Encoder 1|Encoder 2|Encoder 3|Encoder 4|Encoder 5|Encoder 6|Encoder 7|  16383  |
|  CC 16  |  CC 17  |  CC 18  |  CC 19  |  CC 20  |  CC 21  |  CC 22  |  CC 23  |         |
|  N# 32  |  N# 33  |  N# 34  |  N# 35  |  N# 36  |  N# 37  |  N# 38  |  N# 39  |  Pitch  |
|---------|---------|---------|---------|---------|---------|---------|---------|  Wheel  |---------|
|Button 0 |Button 1 |Button 2 |Button 3 |Button 4 |Button 5 |Button 6 |Button 7 |         |Button A |
|  N# 89  |  N# 90  |  N# 40  |  N# 41  |  N# 42  |  N# 43  |  N# 44  |  N# 45  |channel 8|  N# 84  |
|---------|---------|---------|---------|---------|---------|---------|---------|         |---------|
|Button 8 |Button 9 |Button 10|Button 11|Button 12|Button 13|Button 14|Button 15|         |Button B |
|  N# 87  |  N# 88  |  N# 91  |  N# 92  |  N# 86  |  N# 93  |  N# 94  |  N# 95  |    0    |  N# 85  |
|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|

Encoders send a **CONTROL CHANGE** with a relative click value, 1...64 for
clockwise movement and 65...127 for anti-clockwise, and a **NOTE ON** with
velocity 127 for press down and 0 for release.

Buttons send a **NOTE ON** message with velocity 127 for pressed and 0 for
released.

The fader sends a **PITCH BEND CHANGE** with a linear absolute position from 0
to 16256 - only the upper 7 bits of the pitch bend value change, the lower 7
bits are always 0. The "0dB" marking is approximately 12672 (`0x00 0x63`).


# MIDI mapping for lighting control

|---------|---------|---------|---------|---------|---------|---------|---------|---------|
|Encoder 0|Encoder 1|Encoder 2|Encoder 3|Encoder 4|Encoder 5|Encoder 6|Encoder 7|         |
|  CC 48  |  CC 49  |  CC 50  |  CC 51  |  CC 52  |  CC 53  |  CC 54  |  CC 55  |         |
|---------|---------|---------|---------|---------|---------|---------|---------|         |---------|
|Button 0 |Button 1 |Button 2 |Button 3 |Button 4 |Button 5 |Button 6 |Button 7 |         |Button A |
|  N# 89  |  N# 90  |  N# 40  |  N# 41  |  N# 42  |  N# 43  |  N# 44  |  N# 45  |         |  N# 84  |
|---------|---------|---------|---------|---------|---------|---------|---------|         |---------|
|Button 8 |Button 9 |Button 10|Button 11|Button 12|Button 13|Button 14|Button 15|         |Button B |
|  N# 87  |  N# 88  |  N# 91  |  N# 92  |  N# 86  |  N# 93  |  N# 94  |  N# 95  |         |  N# 85  |
|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|

Encoders take a **CONTROL CHANGE** message on channel 0 with value:

- 0 = all off
- 1..11 = single LED
- 17..27 = fill to middle ("pan" mode)
- 33..43 = fill from start
- 49..54 = fill both sides from middle
- 63 = all on

Numbers between these sub-ranges are ignored. Values 64..127 seem to be a
duplicate of the functionality in the 0..63 range. LEDs indexed clockwise from
second to second-last - first and last LEDs are *not* controllable in MC mode.

Buttons take a **NOTE ON** message on channel 0 with velocity:

- 0 = off
- 1 = flashing
- 127 = on
