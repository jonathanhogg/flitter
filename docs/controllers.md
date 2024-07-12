
# Using Controllers

:::{note}
This is currently just a placeholder to capture the full horror of everything I
have to document for `!controller`.
:::

**Flitter** supports interacting with running programs through the use of MIDI
control surfaces. Currently, the list of supported devices is just the Ableton
Push2 and the Behringer X-Touch mini, because those are the two devices that
the author owns.

## Controller Configuration

Controllers are configured by specifying an interface through a set of abstract
controls, and then linking the output of these controls into the [state
mapping](language.md#state) where the values can be accessed by the running
program. The configuration of the control interface can be dynamically
adjusted as the program runs.

The top-level node for controllers is `!controller`. This takes the single
attribute:

`driver=` *DRIVER*
: The controller driver to use. Currently one of `:push2` or `:xtouch_mini`.

The drivers are designed to be resilient to the control device being missing or
disconnecting and reconnecting while **Flitter** is running.

Inside the `!controller` node, there should be a set of individual nodes that
configure each available control.

### Common control attributes

There are a number of attributes that are common to all nodes.

`id=` *ID*
: A value that identifies the specific physical control (button, knob or
otherwise) on the controller. These are defined by the driver and are [listed
below](#supported-controllers).

`state=` *PREFIX*
: This provides a key/prefix to use in the state mapping for the outputs of this
control. Generally the key *PREFIX* will be the primary output value of the
control, but additional values may be available at keys such as
*PREFIX*`;:beat`. The individual control types list these.

`label=` *TEXT*
: A label to use for this control. This is only expected to be useful on
controllers that have some kind of display associated with particular controls.

`color=` *COLOR*
: A single brightness value, or a 3-vector of red, green and blue values. For
controls that have an associated LED, this will specify the output of that LED.
This may just be a `0`/`1`, on/off value if the LED does not support either
brightness or color.

`action=` *ACTION*
: A symbol specifying an internal engine action to take when this control is
activated. The valid actions will vary on whether the control is a button or a
position control (such as a slider or rotary) and are [documented
below](#actions).

### Positional controls

Positional controls are able to represent some value within a range. These
include knobs, pressure pads and sliders. They support the following generic
attributes:

`lower=` *LOWER*
: The lower value of the output range. Defaults to `0`.

`upper=` *UPPER*
: The upper value of the output range. Defaults to `1` and must be greater than
`lower`.

`origin=` *ORIGIN*
: A value that specifies an "origin" value in the output range. This attribute
only makes sense where the physical control has some kind of associated visual
feedback that provides a bar graph style representation for the current value
with a variable "zero" point. This defaults to `0` if `lower` is less than zero
and `upper` is greater than zero, otherwise it will default to `lower`.

`lag=` *LAG*
: Specifies a coefficient to apply an exponential moving average to the
control's output value. This is used to "smooth out" the actual value and is
particularly useful for controls that step slightly, like encoders, or just to
limit the effect of jerky movements on the program. The default is specific to
the driver.

The *processed* output value of a positional control is stored to the `state`
key.

### Rotaries

Rotary controls are knobs, including potentiometers and encoders, that can be
mapped to a value in a range. They are configured with the `!rotary` node, which
accepts the [positional control attrbutes](#positional-controls) above as well
as the following rotary-specific attributes:

`turns=` *TURNS*
: The number of turns of an encoder that represents the full range of the
output. Defaults to `1`.

`wrap=` ( `true` | `false` )
: Specifies whether this value will wrap around between the `upper` and `lower`
ends of the range. This only makes sense for encoders that do not have a
physical limit. Defaults to `false`.

`initial=` *INITIAL*
: Specifies a value in the output range that this control will start at (or
reset back to if resetting is supported). This only makes sense for encoders.

`style=` [ `:volume` | `:pan` | `:continuous` ]
: Specifies a behaviour for the control and a display style if the controller
supports visual feedback. This indicates a preference for a volume style bar
that fills as the knob is turned, a pan style that fills in either direction
from the `origin` point, or a continuous style that indicates a position with a
single mark (most useful with encoders and `wrap=true`). This defaults to
`:volume` unless `origin` is greater than `lower`, in which case it defaults to
`:pan`.

Setting `style=:continuous` and `wrap=false` on an encoder control will allow
the output value to range beyond `lower` and `upper`. The `lower` and `upper`
values will be used for display purposes, but turning the knob "beyond" these
points will cause the output value to continue to increase or decrease.

### Buttons

A button represents

`toggle=`

`group=`

`initial=`

### Pads

`lower=`

`upper=`

`threshold=`

### Sliders

`lower=`

`upper=`

`origin=`

`lag=`

`initial=`

### Touch-sensitive controls



## Actions

## Supported Controllers

### Ableton Push 2

### Behringer X-Touch mini
