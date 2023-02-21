# flitter

**flitter** is a 2D (presently) visuals language and engine designed for live
performances. While **flitter** supports a basic form of live-coding (live
reload of source files), it is designed primarily for driving via an Ableton
Push 2 controller.

The engine that runs the language is capable of: drawing in windows with Skia,
OpenGL shaders and video files; driving a LaserCube plugged in over USB (other
lasers probably easy to support); driving DMX lighting via a USB DMX interface
(currently via an Entec/-compatible interface or my own crazy hand-built
interfaces).

It is implemented in a mix of Python and Cython. I use and develop **flitter**
exclusively on macOS. It is notionally portable – in that there's no particular
reason why it wouldn't work on Linux or Windows – but I've not tested either of
those platforms.

## Background

This is probably my third implementation of a variant of these ideas.
Originally, I developed a simple visuals system as an embedding in Python
(using a crazy system of `with` statements) sending JSON graphs over a WebSocket
to a JavaScript web app that rendered the results in an HTML 2D canvas.

This current version was initially developed over a furious fortnight in
October 2021 leading up to a live performance at the Purcell Room, London
Southbank Centre, supporting Bishi at her 'Let My Country Awake' album launch.
The work was partially supported by Ableton, who gave me an artist discount on
a Push 2. I've been working on **flitter** off-and-on since then trying to
develop it as a live tool.

While I think live-reload is a hugely useful tool for testing ideas, I find the
idea of writing code live too terrifying. What I'm interested in is using
physical knobs and buttons to control parametric code. So **flitter** is
designed to provide different ways to manipulate scene graphs, including the
ability to search and alter a graph, and piece one together from sections.

Nothing about this is in any sense "finished". It is still a testbed for my
ideas. I've put this on GitHub in case someone else finds something in this that
is interesting. If you're thinking of using a Push 2 then you might find the
Python API for this useful – it provides complete support for all of the
controls and drawing stuff on the screen.

-- Jonathan Hogg <me@jonathanhogg.com>

## Requirements

At least Python 3.10 is *required* as the code uses `match`/`case` syntax. I
work exclusively in 3.11, so I may have introduced some other dependency on
this later version.

Install the required modules with:

```
pip3 install -r requirements.txt
```

For reference, they are:

- `cython` - because half of **flitter** is implemented in Cython for speed
- `numpy` - for fast memory crunching
- `lark` and `regex` - for the language parser
- `python-rtmidi` - for talking MIDI to the Push 2
- `pyusb` - for sending screen data to the Push 2 and for talking to LaserCubes
- `skia-python` - for 2D drawing
- `pyglet` - for OpenGL windowing
- `moderngl` - because the OpenGL API is too hard
- `av` - for decoding video
- `pyserial` - for talking to DMX interfaces and lasers

## The Language

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

!window size=SIZE
    !canvas size=SIZE antialias=true composite=:add
        !group font="Helvetica" font_size=100 color=sine(beat/2)
            !text point=SIZE/2 text="Hello world!"
```

This contains a comment, a `let` statement and a node creation statement.
Indented statements below this represent child nodes. Any name with a `!` in
front of it creates a node of that *kind*; the bindings following this specify
attributes of the node.

When the **flitter** engine is run with this file, it will evaluate the code
repeatedly (at an attempted 60fps) and render this to screen. Note that one
explicitly specifies a window to be drawn into. The engine supports multiple
windows.

```
./flitter.sh examples/hello.fl
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
running and it will reload the code on-the-fly. This is the best way to
experiment, but the reloading may be a little too janky for performing live.

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

`!window`s composite their children and a `!canvas` is transparent until drawn
into.

Other useful language features:

### Ranges

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

### Loops

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

### Conditionals

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

### Queries

Queries allow the node graph so-far to be searched and manipulated. They use
a CSS-selector-like syntax that is best explained by example:

- `{*}` matches *any* and *all* nodes in the graph
- `{window}` matches any `!window` node
- `{#spot}` matches any node with the `#spot` tag
- `{shader#blur}` combines these, matching only a `!shader` node with a `#blur`
tag
- `{ellipse|rect}` matches an `!ellipse` or a `!rect` node
- `{group path}` matches a `!path` node within an `!group` node
- `{path>line_to}` matches a `!line_to` node *immediately* below a `!path` node
- `{group.}` returns all `!group` nodes reachable from the root, but *not* any
further `!group` nodes *within* those, i.e., it stops recursive descent on a
match

The result of a query expression is a vector of matching nodes. Note that nodes
are (currently) only appended to the main graph from the top-level after the
complete evaluation of each expression, including any indent-append operations.
Thus a query evaluated within a nested expression cannot match any node that
makes up part of that expression.

A query expression may be combined with attribute-setting or node-appending
expressions to amend the current graph. For example:

```
let SIZE=1280;720

!window size=SIZE
    !canvas size=SIZE color=1 font_size=100
        !text point=SIZE/2 text="Hello world!"

if beat > 4
    {canvas} color=1;0;0
```

In this example, the text will turn red after 4 beats (2 seconds by default).

Appending nodes to a query will append those nodes to all matches:

```
...

if beat > 4
    {canvas}
        !text point=SIZE*(0.5;0.75) text="(RED)" color=1;0;0
```

Appending queries to a node will re-parent the matching nodes into this new
position in the graph. For example, the following code combines two queries
to wrap all nodes within the canvas in a new group that turns them upside down
and changes the default color to red:

```
...

if beat > 4
    {canvas}
        !group color=1;0;0 translate=SIZE rotate=0.5
            {canvas>*}
```

### Functions

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

### Pseudo-random sources

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
let SIZE=1920;1080

!window size=SIZE
    for i in ..10
        let x=uniform(:x;i)[beat] y=uniform(:y;i)[beat]
            r=10*beta(:r;i)[beat] h=uniform(:h;i)[beat]
        !path
            !ellipse point=(x;y)*SIZE radius=r*100
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

### Partial evaluation

When a source file is loaded is is parsed into an evaluation tree and then that
tree is *partically-evaluated*. This attempts to evaluate all literal/constant
expressions. The partial-evaluator is able to construct partial node graphs,
unroll loops with a constant source vector, evaluate conditionals with constant
tests, call functions with constant arguments (including creating pseudo-random
streams), replace `let` names with constant values and generally reduce as much
of the evaluation tree as possible to literal values.

Unbound names (which includes all of the globals listed above, like `beat`),
all state (`$...`) and queries (`{...}`) are irreducible, as are obviously any
expressions that include these.

## Controlling code with a Push 2

Assuming that you have an Ableton Push 2 connected,

```
./push.sh
```

will fire up the process that talks to it. This interfaces with the engine via
OSC messaging (on `localhost`) and is generally resilient to the engine starting
and stopping. You can also automatically start the Push 2 interface as a
managed subprocess of the engine by just adding the `--push` command-line
option to `flitter.sh`.

Other than tempo control, you won't have much in the way of interface until you
specify one in the program itself. `!pad` and `!encoder` nodes at the top level
in the graph will configure pads and encoders on the Push 2. Again, really
you'll need to look at the examples.

The outputs from the pads and encoders are put into a special environment map
in the engine that is read from with `$(:some_key)`. This allows one to
parameterise the program and live manipulate it.

If multiple code files are specified on the command line of the engine, then
these will be loaded as multiple "pages". The previous and next page buttons on
the Push 2 can be used to switch between the files. The state of each page is
maintained when switching, including the current tempo and start time of the
beat clock, and the state values of all of the pads and encoders.

## OpenGL Shaders

The `!shader` node allows insertion of an OpenGL shader into the scene graph.
This takes a `fragment` GLSL text code attribute (also an optional `vertex` code
attribute, but I've never tested this) and a `size` for the framebuffer. Any
children of this node will be bound to `sampler2D` uniforms called `texture0`,
`texture1`, etc. The special `uniform sampler2D last` (if present) will be bound
to the previous contents of the shader's output framebuffer – this allows for
feedback loops. The fragment shader executes over the framebuffer rectangle with
`in vec2 coord` being the standardised rectangle coordinate [0..1] (multiply by
`size` to get pixel coordinates) and `out vec4 color` being the color to write
to the framebuffer. All textures are RGBA with pre-multiplied alpha.

All of the standard language globals will be bound to `float` uniforms of
the same name. Any other uniforms will be bound to the value of `!shader`
attributes with the same name. `float`, `vec2`, `vec3` and `vec4` uniforms
expect 1-, 2- , 3- and 4-vector values respectively. Arrays of these types
will expect a vector with an appropriate multiple of those sizes. GLSL default
values for the uniforms can be given in the code and will be used if the
attribute value is not specified (or is invalid).

Generally one would use the built-in `read()` function to load text from a file
for the code attribute, a la:

```
let SIZE=1920;1080

!window size=SIZE
    !shader size=SIZE fragment=read('blur.frag') radius=5
        !canvas size=SIZE
            ...
```

Shader code will be reloaded on-the-fly if the file's modification timestamp
changes. On macOS the maximum OpenGL level is `#version 410`. Shaders can be
nested to apply multiple effects.
