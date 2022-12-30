# flitter

**flitter** is a 2D (presently) visuals language and engine designed for live
performances. While **flitter** supports a basic form of live-coding (live
reload of source files), it is designed primarily for driving via an Ableton
Push 2 controller.

It is implemented in a mix of Python and Cython.

## Background

This is probably my third implementation of a variant of these ideas.
Originally, I developed a simple visuals system as an embedding in Python
(using a crazy system of `with` statements to build the graph) sending JSON
graphs over a WebSocket to a JavaScript web app that rendered the results in an
HTML 2D canvas.

This version was initially developed over a furious fortnight in October 2021
leading up to a live performance at the Purcell Room, London Southbank Centre,
supporting Bishi at her 'Let My Country Awake' album launch. The work was
partially supported by Ableton, who gave me an artist discount on a Push 2. I've
been working on **flitter** off-and-on since then trying to develop it as a live
tool.

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

Python 3.10 is *required* as the code uses `match`/`case` syntax.

Install the required modules with:

```sh
pip3 install -r requirements.txt
```

For reference, they are:

- `cython` - because half of **flitter** is implemented in Cython for speed
- `numpy` - for fast memory crunching
- `lark` - for the language parser
- `regex` - for the language parser
- `python-rtmidi` - for talking MIDI to the Push 2
- `pyusb` - for sending the screen data to the Push 2
- `skia-python` - for 2D drawing
- `pyglet` - for OpenGL windowing
- `moderngl` - because the OpenGL API is too hard
- `posix_ipc` - for multiprocessing

## The Language

**flitter** is mainly a test-bed for my ideas. The language is a declarative
graph-construction language (`Node` in the model). All values are arrays
(`Vector` in the model), delimited with semicolons, and all maths is piece-wise.
Singleton vectors are generally extended out to the useful length, i.e.,
`(1;2;3;4) * 2` is `2;4;6;8`. The `null` value is an empty array and most
expressions evaluate to this in event of an error. In particular, all maths
expressions involving a `null` will evaluate to `null`.

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
    !canvas size=SIZE
        !group font="Helvetica" font_size=100 color=sine(beat/2)
            !text point=SIZE/2 text="Hello world!"
```

This contains a comment, a `let` statement and a node creation statement.
Indented statements below this represent child nodes. Any name with a `!` in
front of it creates a node of that *kind*; the bindings following this specify
attributes of the node.

When the **flitter** engine is run with this file, it will evaluate the code
repeatedly (at an attempted 60fps) and render this to screen. Note that one
explicitly specifies a window to be drawn into. The engine actually supports
multiple windows, although this is pretty untested so your mileage may vary.

```
./flitter.sh examples/hello.fl
```

A `!canvas` node creates a 2D drawing canvas that follows an SVG-like drawing
model. `!group` sets styles and transforms. `!text` draws text, centred at
`point` by default. `sine()` is a function that reproduces a sine wave ranging
over [0..1] with the argument expressed in waves (turns/circumferens). `beat` is
a global representing the current floating point beat of the master clock. The
default BPM is 120, so `beat/2` is effectively a number of seconds since the
clock started. `color`s are 3- or 4-vectors (RGB and RGBA) in the range [0..1],
but the return value from `sine()` here is automagically extended out to the
3-vector resulting in a varying brightness of white.

So the end result of this should be the text "Hello world!" pulsing white in
the middle of the window. Beyond this you'll need to look at the examples and
the code as I've not documented the language at all. You can edit and re-save
the code while the engine is running and it will reload the code on-the-fly.
This is the best way to experiment, but the reloading may be a little too janky
for performing live.

The available global values are:

- `beat` - the current master clock beat (a monotonically-increasing floating
    point value)
- `quantum` - the beats per quantum (usually 4)
- `delta` - the difference between the current value of `beat` and the value
    at the last display frame
- `clock` - the time in seconds since the "start" of the master clock, this
    will adjust when the tempo or quantum is changed to keep the value of `beat`
    constant, so it's generally not a particularly useful value
- `performance` - a value in the range [0.5 .. 1.5] that represents how well
    the engine is doing at maintaining the maximum frame rate (usually 60fps,
    but configurable with a command-line option) – above 1.0 means that the
    engine has cycles to spare and below this means the frame rate is dipping

`!window`s composite their children and a `!canvas` is transparent until drawn
into.

## Controlling code with a Push 2

Assuming that you have an Ableton Push 2 connected,

```
./push.sh
```

will fire up the process that talks to it. This interfaces with the engine via
OSC messaging (on `localhost`) and is generally resilient to the engine starting
and stopping.

Other than tempo control, you won't have much in the way of interface until you
specify one in the program itself. `!pad` and `!encoder` nodes at the top level
in the graph will configure pads and encoders on the Push 2. Again, really
you'll need to look at the examples.

The outputs from the pads and encoders are put into a special environment map
in the engine that is read from with `$("some_key")`. This allows one to
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
to the framebuffer. All textures are pre-multiplied alpha.

All of the standard language globals will be bound to `float` uniforms of
the same name. Any other uniforms will be bound to the value of `!shader`
attributes with the same name – `float`, `vec2`, `vec3` and `vec4` uniforms
expect 1-, 2- , 3- and 4-vector values respectively – including the standard
`size` attribute giving the size of the framebuffer. GLSL default values for the
uniforms can be given in the code and will be used if the attribute value is not
specified (or is invalid).

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
