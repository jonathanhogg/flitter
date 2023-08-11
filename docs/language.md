
# The Language

**flitter** is a declarative graph-construction language (`Node` in the model).
All values are arrays (`Vector` in the model), delimited with semicolons, and
all maths is piece-wise. Short vectors are repeated as necessary in binary
operations, i.e., `(1;2;3;4) * 2` is `2;4;6;8`. The `null` value is an empty
array and most expressions evaluate to this in event of an error. In particular,
all maths expressions involving a `null` will evaluate to `null`.

A-la Python, indentation represents block structuring. `let` statements name
constant values, everything else is largely about creating nodes to append to
the implicit *root* node of the graph. There are *no variables*. The language
is sort of pure-functional, if you imagine that the statements are monadic on
a monad encapsulating the current graph.

The simplest program would be something like:

```
-- Hello world!

let SIZE=1280;720

!window #top size=SIZE
    !canvas size=SIZE antialias=true composite=:add
        !group font="Helvetica" font_size=100 color=sine(beat/2)
            !text point=SIZE/2 text="Hello world!"
```

This contains a comment, a `let` statement and a node creation statement.
Indented statements below this represent child nodes. Any name with a `!` in
front of it creates a node of that *kind*; the bindings following this specify
attributes of the node. Nodes can also be followed by one or more `#tag`s to
add tags that can be later searched for with *queries* (see language features
below).

When the **flitter** engine is run with this file, it will evaluate the code
repeatedly (at an attempted 60fps) and render this to screen. Note that one
explicitly specifies a window to be drawn into. The engine supports multiple
windows.

```
flitter examples/hello.fl
```

A `!canvas` node creates a 2D drawing canvas that follows an SVG-like drawing
model. `!group` sets styles and transforms. `!text` draws text, centred at
`point` by default. `sine()` is a function that reproduces a sine wave ranging
over [0..1] with the argument expressed in waves (turns/circumferens). `beat`
is a global representing the current floating point beat of the main clock. The
default BPM is 120, so `beat/2` is effectively a number of seconds since the
clock started. `color`s are 3- or 4-vectors (RGB and RGBA) in the range [0..1],
but the return value from `sine()` here is automagically extended out to the
3-vector resulting in a varying brightness of white.

`true` and `false` are synonyms for `1` and `0`. Truthfulness is represented by
any non-empty vector that contains something other than 0 or the empty string.
Names prefixed with a `:` are *symbols*, which internally are just string
literals (so `"add"` is the same as `:add`) but are easier to write and read
when specifying an identifier. When a string is required, such as the `text`
attribute of `!text` in the example above, each element of the vector will be
cast to a string if necessary and then concatenated together, e.g.,
`"Hello ";name;"!"`.

So the end result of this should be the text "Hello world!" pulsing white in
the middle of the window. You can edit and re-save the code while the engine is
running and it will reload the code on-the-fly - this is usually fast enough
not to be noticeable.

The available global values are:

- `beat` - the current main clock beat (a monotonically-increasing floating
    point value)
- `quantum` - the beats per quantum (usually 4)
- `delta` - the difference between the current value of `beat` and the value
    at the last display frame
- `clock` - the time in seconds since the "start" of the main clock, this
    will adjust when the tempo or quantum is changed to keep the value of
    `beat` constant, so it's generally not a particularly useful value
- `performance` - a value in the range [0.5 .. 2.0] that increases fractionally
    if the engine has time to spare and decreases fractionally if it is missing
    the target frame rate; this value can be multiplied into variable loads
    in the code – e.g., number of things on screen – to maintain frame rate

## Ranges

```
start..stop|step
```

A range creates a vector beginning at `start` and incrementing (or
decrementing if negative) by `step` until the value is equal to or passes
`stop` - the last value is *not* included in the range, i.e., it is a
half-open range.

`start` and `|stop` may be omitted, in which case the vector will begin at 0
and increment by 1. Therefore:

- `..10` evaluates to the vector `0;1;2;3;4;5;6;7;8;9`
- `1..21|5` evaluates to the vector `1;6;11;16`
- `1..0|-0.1` evaluates to the vector `1;0.9;0.8;0.7;0.6;0.5;0.4;0.3;0.2;0.1`

Ranges are *not* lazy like they are in Python, so `0..1000000` will create a
vector with 1 million items.

## Loops

```
for name[;name...] in expression
    expression
    ...
```

For loops iterate over vectors binding the values to the name(s). The result of
evaluating the expressions within the loop body are concatenated into a single
vector that represents the result value of the loop. Normally this would be a
vector of nodes to be appended to some enclosing node.

When multiple names are given, each iteration will take another value from the
source vector. If there are not enough values left in the source vector to bind
all names in the last iteration, then the names lacking values will be bound to
`null`.

Iterating with multiple names is particularly useful combined with the `zip()`
function that merges multiple vectors together:

```
!group #clockface
    let theta=(..12)/12
    for x;y in zip(cos(theta), sin(theta)) * 100
        !ellipse point=x;y radius=10
    !fill color=1
```

Note that `cos()` and `sin()` here are taking a 12-vector and returning another
12-vector, `zip()` combines these into a 24-vector and the multiplication by
100 is applied to every element of this. Also of note here is that the flitter
versions of `cos()` and `sin()` take values in *turns* not in radians. The
dozen `!ellipse` nodes created by the loop are combined into a 12-vector, this
has the final `!fill` node concatenated with it and then all of these nodes are
appended to the `!group`.

There is actually a more convenient `polar(theta)` function that does the same
thing as `zip(cos(theta), sin(theta))`. Arguably, it would be even neater to
implement a clock face using `rotate` transforms instead.

## Conditionals

```
if test
    expression
    ...
[elif test
    expression
    ...]
[else
    expression
    ...]
```

*test* is any expression and it will be considered *true* if it evaluates to a
non-empty vector containing something other than 0s or empty strings. So `0` is
false, as is `null`, `""` and `0;0;0`. The result of evaluating the matching
indented expressions is the result value of the `if`. In the absence of an
`else` clause the result of an `if`/`elif` with no true tests is `null`.

## Queries

Queries allow the node graph so-far to be searched and manipulated. They use
a CSS-selector-like syntax that is best explained by example:

- `{*}` matches *all* nodes in the graph
- `{window}` matches any `!window` node
- `{#spot}` matches any node with the `#spot` tag
- `{shader#blur}` matches `!shader` nodes with the `#blur` tag
- `{ellipse|rect}` matches any `!ellipse` or `!rect` node
- `{group path}` matches `!path` nodes anywhere within a `!group` node
- `{path>line_to}` matches `!line_to` nodes *immediately* below a `!path` node
- `{group.}` returns all `!group` nodes reachable from the root, but *not* any
further `!group` nodes *within* those, i.e., it stops recursive descent on a
match

The result of a query expression is a vector of matching nodes. Note that nodes
are (currently) only appended to the main graph from the top-level after the
complete evaluation of each expression, including any indented append
operations. Thus a query evaluated within a nested expression cannot match any
node that makes up part of that expression.

A query expression may be combined with attribute-setting or node-appending
expressions to amend the current graph. For example:

```
let SIZE=1280;720

!window size=SIZE
    !canvas size=SIZE color=1 font_size=100
        !text point=SIZE/2 text="Hello world!"

if beat > 10
    {canvas} color=1;0;0
```

In this example, the text will turn red after 10 beats (5 seconds by default).

Appending nodes to a query will append those nodes to all matches:

```
...

if beat > 10
    {canvas}
        !text point=SIZE*(0.5;0.75) text="(RED)" color=1;0;0
```

Appending queries to a node will re-parent the matching nodes into this new
position in the graph. For example, the following code combines two queries
to wrap all nodes within the canvas in a new group that turns them upside down
and changes the default color to red:

```
...

if beat > 10
    {canvas}
        !group color=1;0;0 translate=SIZE rotate=0.5
            {canvas>*}
```

## Functions

```
func name(parameter[=default], ...)
    expression
    ...
```

`func` will create a new function and bind it to `name`. Default values may be
given for the parameters and will be used if the function is later called with
an insufficient number of matching arguments, otherwise any parameters lacking
matching arguments will be bound to `null`. The result of evaluating all
body expressions will be returned as a single vector.

## Template Functions

This is something of a hack, but the special `@` operator allows calling a
function using similar syntax to constructing a node. The name following `@`
should be the name of the function to be called, any named attributes placed
after this are passed as arguments to the respectively-named function
parameters. Any indented expressions are evaluated and the resulting
vector passed as the first argument to the function. Function parameters that
are not bound by a pseudo-attribute will have their default value if one was
specified in the function or `null` otherwise.

For example:

```
func shrink(nodes, percent=0)
    !transform scale=1-percent/100
        nodes

!canvas ...
    @shrink percent=25
        !ellipse radius=10
```

This (rather pointless) example shows using a template function call to shrink
a circle by 25%, by wrapping it in an equivalent `!transform scale=0.75`
node.

## Pseudo-random sources

**flitter** provides three useful sources of pseudo-randomness: `uniform()`,
`normal()` and `beta()`. These built-in functions return special "infinite"
vectors that may only be indexed. These infinite vectors provide a reproducible
stream of numbers from the *Uniform(0,1)*, *Normal(0,1)* and *Beta(2,2)*
distributions.

The single argument to both functions is a vector that acts as the
pseudo-random seed. Floating-point numbers within this seed vector are truncated
to whole numbers before the seed is calculated. This is deliberate to allow new
seeds to be generated at intervals, e.g.: `uniform(:foo;beat)` will create a new
stream of pseudo-random numbers for each beat of the main clock. Multiplying or
dividing this seed value then allows for different intervals, e.g., four times a
beat: `uniform(:foo;beat*4)`.

Similarly, the index value to the infinite-vector is truncated to pick a
specific number in the stream. For example:

```
let SIZE=1280;720

!window size=SIZE
    !canvas size=SIZE
        for i in ..10
            let x=uniform(:x;i)[beat] y=uniform(:y;i)[beat]
                r=10*beta(:r;i)[beat] h=uniform(:h;i)[beat]
            !path
                !ellipse point=(x;y)*SIZE radius=r*50
                !fill color=hsv(h;1;1)
```

This will create 10 circles distributed uniformly around the window with
different radii clustered around 50 pixels and different uniformly picked hues.
Every beat, the circles will be drawn in different places and with different
sizes and hues. Note the use of symbols (`:x`, `:y`, etc.) combined with the
index of the circle to create unique seeds. This code will draw exactly the
same sequence of circles *every* time it is run as the pseudo-random functions
are stable on their seed argument. There is no mechanism for generating true
random numbers in **flitter**.

The pseudo-random streams can be indexed with a range vector to generate a
vector of numbers, e.g.: `uniform(:some_seed)[..100]` will generate a 100-vector
of uniformly distributed numbers. The streams are arbitrarily long and are
unit-cost to call, so indexing the billionth number takes the same amount of
time as the 0th. Unlike normal vectors, the streams also extend into negative
indices.

Pseudo-random streams may be bound to a name list in a `let` expression to
pick off the first few values, e.g.:

```
let x;y;z = uniform(:position)
```

They are also considered to be `true` in conditional expressions (`if`, `and`,
etc.). In all aspects, pseudo-random streams appear like the `null` vector,
e.g., attempts to use them in mathematical expressions will evaluate to `null`.

## State

The outside world (only in the form of a Push 2 at the moment) communicates with
a running **flitter** program via a *state* mapping. This can be queried with
the `$` operator like so:

```
let SIZE=1280;720

!window size=SIZE
    !canvas size=SIZE translate=SIZE/2
        !ellipse radius=$:circ_radius
        !fill color=1;0;0

!encoder number=0 name="Radius" state=:circ_radius lower=0 upper=300 initial=100
```

This shows the first Push 2 encoder being configured as a knob that changes the
radius of a red filled-circle drawn in the middle of the window. The
`state=:circ_radius` attribute of the `!encoder` node specifies the state key to
write values to and the current value is retrieved with `$:circ_radius`.

A key is any vector of numbers and/or strings (/symbols). As the `;` operator
binds with very low precedence, a non-singleton key needs to be surrounded with
brackets, e.g., `$(:circle;:radius)`.

## Counters

Arbitrary counters can be managed with the `counter()` function. This takes
either two or three arguments and **has state side-effects**.

In the three-argument form, `counter(counter_id, clock, speed)`, the function
will create or update a counter with a specific `clock` value and `speed`,
storing the current state of the counter in the state mapping with the given
`counter_id` key. The function returns the current count.

Counters begin at zero and increment upwards by `speed` every unit increase of
`clock`. The counter speed can be changed at any point and the counter will
return the count at the previous speed before switching to counting with the
new speed. The stored state only changes when the speed changes. `clock`
*should* be a monotonically increasing number, such as `beat`. `speed` is
allowed to be zero to stop the counter or negative to count downwards.

Either `clock` or `speed` can be given as an n-vector to create a
multi-dimensional counter. This can be useful, for instance, to move an object
through space with a velocity vector:

```
let velocity = 0.2;-1.5;3.0
    position = counter(:thing_position, beat, velocity)
```

The two argument form of the function omits the speed and returns the count
matching the value of `clock` at the last counter speed. If the counter has not
already been initialised, then the speed will default to 1.

## Pragmas

There are three supported pragmas that may be placed at the top-level of a
source file:

```
%pragma tempo 110
%pragma quantum 3
%pragma fps 30
```

These respectively set the initial tempo and/or quantum of the main clock (the
defaults are 120 and 4, respectively), and the *current* target frame rate of
the engine (default is 60 or the value specified with the `--fps` command-line
option).

## Partial evaluation

When a source file is loaded it is parsed into an evaluation tree and then that
tree is *partially-evaluated*. This attempts to evaluate all literal/constant
expressions. The partial-evaluator is able to construct parts of node graphs,
unroll loops with a constant source vector, evaluate conditionals with constant
tests, call functions with constant arguments (including creating pseudo-random
streams), replace `let` names with constant values and generally reduce as much
of the evaluation tree as possible to literal values.

The partial-evaluator will also run again incorporating the state if it has
been stable for a period of time (configurable with the `--evalstate`
command-line option). If the state is touched again (i.e., a pad or encoder
is touched) then this partially-evaluated tree is discarded and the engine will
return to dynamically evaluating state expressions.

Unbound names (which includes all of the globals listed above, like `beat`) and
queries (`{...}`) are always dynamic, as are obviously any expressions that
then include these.
