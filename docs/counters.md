
# Counters

Counters provide a simple way of creating custom state values that increment
or decrement at a controllable rate. A counter is specified at the top-level
of a **Flitter** program with the `!counter` node.

The supported attributes are:

`state=`*state*
: A required attribute that specifies the state key this counter value will be
stored under.

`initial=`*initial*
: An optional initial value for this counter. If not specified, the counter will
be initialised to `0`.

`rate=`*rate*
: An attribute that specifies the rate of change of this counter. This defaults
to `0` if not specified.

`time=`*time*
: An optional attribute that specifies the time value used for counter
calculations. This defaults to the frame time if not specified.

`minimum=`*minimum* (or `min=`*minimum*)
: An optional minimum value for the counter. If not specified, no minimum is
enforced.

`maximum=`*maximum* (or `max=`*minimum*)
: An optional maximum value for the counter. If not specified, no maximum is
enforced.

Each counter object keeps track of its current value and the last value of
`time`. On each frame, the counter calculates the delta between the new value
of `time` (or the frame time) and the last value. The counter will then be
incremented by *rate* multiplied by this delta.

The `time` value does not need to be a monotonically increasing clock. There
is nothing to stop `time` stopping or going backwards. In fact, it is perfectly
valid to use another counter as the value for `time`.

All of the numeric vectors, *initial*, *rate*, *time*, *minimum* and *maximum*
may be n-vectors. In this case, the normal piece-wise rules for vector
mathematics apply.

:::{note}
The `minimum` and `maximum` values also apply piece-wise. This means that if
the counter is an n-vector and `minimum` and/or `maximum` are 1-vectors, then
*each* element of the counter n-vector will be limited by these values
individually. The behaviour is simulat to that of the [`clamp()` built-in
function](builtins.md#mathematical-functions).
:::

## Counter State

Each counter stores two values in the state mapping:

*state*
: The current value of the counter.

*state*`;:time`
: The last value of the `time` attribute (or frame time).

Note that a counter will only be initialised when rendering occurs after
program execution. This means that on the very first execution of a program,
`$(`*state*`)` will return `null`. You may need to check for this and use a
default (or initial) value to ensure that the first frame of the program is
executed correctly.

## Example

```flitter
let N=10

!counter state=:foo \
         initial=100*uniform(:initial)[..N] \
         rate=noise(:rate, (..N)/100, clock/10) \
         minimum=0 \
         maximum=100
```

This creates a counter that tracks 10 independent values, initialised to
uniformly-distributed random numbers in the range $[0,100]$. A noise function is
used to move each of these values at a constantly changing rate. The counter is
limited to a minimum of $0$ and a maximum of $100$. The frame time is used as
the default clock for calculating the change of these values.

On the first execution of the program `$(:foo)` will be `null`. On the second
execution, it will be equal to the 10-item vector `100*uniform(:initial)[..N]`.
On each subsequent frame `$(:foo)` will be a 10-item vector with each item
slightly different to the previous frame by an amount related to the noise
function and the frame time delta. Each of these values will never go below $0$
or above $100$.
