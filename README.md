# flitter

**flitter** is a 2D and basic 3D visuals language and engine designed for live
performances. While **flitter** supports a basic form of live-coding (live
reload of source files), it is designed primarily for driving via an Ableton
Push 2 controller.

The engine that runs the language is capable of: 2D drawing with Skia; running
OpenGL shaders as image generators/filters; rendering videos; rendering basic
3D scenes in OpenGL with a few simple primitives, mesh model loading, ambient,
directional, point and spot- light sources with basic Phong lighting/shading;
driving a LaserCube plugged in over USB (other lasers probably easy to support);
driving DMX lighting via a USB DMX interface (currently via an Entec/-compatible
interface or my own crazy hand-built interfaces).

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

Install flitter and all of its requirements with:

```sh
pip3 install .
```

or some other suitable PEP 517 build mechanism. I'd recommend doing this in a
virtual env, but you do you.

For reference, the runtime dependencies are:

- `numpy` - for fast memory crunching
- `lark` and `regex` - for the language parser
- `python-rtmidi` - for talking MIDI to the Push 2
- `pyusb` - for sending screen data to the Push 2 and for talking to LaserCubes
- `skia-python` - for 2D drawing
- `pyglet` - for OpenGL windowing
- `moderngl` - because the OpenGL API is too hard
- `trimesh` - for generating/loading 3D triangular mesh models
- `scipy` - because `trimesh` needs it for some operations
- `av` - for encoding and decoding video
- `pillow` - for saving screenshots as image files
- `mako` - for templating of the GLSL source
- `pyserial` - for talking to DMX interfaces and lasers
- `loguru` - because the standard library `logging` is just too antiquated

and the install-time dependencies are:

- `cython` - because half of **flitter** is implemented in Cython for speed

## Running

Along with the `flitter` Python package, the `flitter` command-line script
should be installed. This can be used to execute a **flitter** script with:

```sh
flitter examples/hello.fl
```

You might want to add the `--verbose` option to get some more logging. You can
see a full list of available options with `--help`.

Everything else you might need to know can be found in the [docs folder](/docs)
(or by reading the source if that fails).

### Multi-processing

I love Python, but the global interpreter lock basically makes any kind of
serious multi-threading impossible. **flitter** supports a limited form of
multi-processing instead. Adding the `--multiprocess` option on the command line
will cause a separate renderer child process to be executed for each window,
laser or DMX bus.

The main process handles evaluating the script to produce an output tree. This
tree is then fed to each renderer and all renderers are waited on before moving
on to the next evaluation. This works well if the script and the renderers are
expensive to run, but the tree is small. As the tree grows large, the cost of
pickling and un-pickling it across the process boundaries becomes a bottleneck,
so your mileage may vary.
