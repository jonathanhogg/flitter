![Screenshot from a flitter program showing colourful distorted ellipse shapes
with trails moving outwards from the centre of the screen.](docs/header.jpg)

# flitter

**flitter** is a 2D and basic 3D visuals language and engine designed for live
performances. While **flitter** supports a basic form of live-coding (live
reload of source files), it is designed primarily for driving via a MIDI surface
(an Ableton Push 2 controller at the moment).

The engine that runs the language is capable of:

- 2D drawing (loosely based on the HTML canvas/SVG model)
- 3D rendering of primitive shapes and external triangular mesh models (in a
bunch of formats including OBJ and STL); ambient, directional, point and
spot- light sources with (currently shadowless) Phong lighting/shading; simple
fog; perspective and orthographic projections; texture-mapping with the output
of other visual units (like a drawing canvas or a video)
- playing videos at arbitrary speeds (including in reverse, although video will
stutter if it makes extensive use of P-frames)
- running GLSL shaders as stacked image generators and filters, with live
manipulation of uniforms and live reload/recompilation of source
- compositing all of the above and rendering to one or more windows
- saving output to image and video files
- driving arbitrary DMX fixtures via a USB DMX interface (currently via an
Entec-compatible interface or my own crazy hand-built devices)
- driving a LaserCube plugged in over USB (other lasers probably easy to
support)
- taking live inputs from Ableton Push 2 or Behringer X-Touch mini MIDI
surfaces (other controllers relatively easy to add)

Fundamentally, the system consists of an engine that repeatedly evaluates a
program with a beat counter and the current state of the MIDI surface. The
output of the program is a tree of nodes that describe the visuals. A series of
renderers turn these nodes into 2D and 3D drawing commands (or DMX packets, or
laser DAC values). It's sort of like using a functional language to build a
web page DOM - something akin to React.

It is implemented in a mix of Python and Cython. I use and develop **flitter**
exclusively on macOS. It is notionally portable – in that there's no particular
reason why it wouldn't work on Linux or Windows – but I've not tested either of
those platforms. Please give it a go and raise any issues you come across, or
get in touch if - amazingly - it just works.

## Background

This is probably my third implementation of a variant of these ideas.
Originally, I developed a simple visuals system as an embedding in Python
(using a crazy system of `with` statements) that sent JSON graphs over a
WebSocket to a JavaScript web app that rendered the results in an HTML 2D
canvas. This was actually a three-layer system that involved running the Python
to build a graph, with the graph using a small expression language that allowed
evaluating the attribute values with respect to time, and then the final static
graph would be rendered. I quickly found this too limiting and wanted a more
expressive, single combined language.

This current version was initially developed over a furious fortnight in
October 2021 leading up to a live performance at the Purcell Room, London
Southbank Centre, supporting Bishi at her 'Let My Country Awake' album launch.
The work was partially supported by Ableton, who gave me an artist discount on
a Push 2. I've been working on **flitter** off-and-on since then developing it
as a live tool.

While **flitter** supports live coding, that's not primarily why I designed
it. As a programmer, I enjoy using code to create visual artwork and I am
fascinated by how generating all of the visuals live affords me the ability to
improvise a performance. However, I'm interested in complex, narrative visuals
that do not lend themselves to coding in-the-moment. Thus, I have in no way
optimised the language for this purpose - meaning it is probably too verbose for
performative coding (that and I haven't bothered building a mechanism for
overlaying code onto the output).

I spend a huge amount of time in advance of a performance thinking and designing
the visuals. In this period I will constantly iterate on the code, and there I
really value the ability to immediately see the effects of the code changes I am
making. However, during a performance I prefer to use physical knobs and buttons
to riff on the themes I have developed in advance. So **flitter** is designed to
interface with a MIDI surface and then provide different ways to parameterise
the running code and manipulate and alter the graph that describes the visuals.

Nothing about **flitter** is in any sense "finished". It is still a testbed for
my ideas. The language and graph semantics are in near constant flux and I
semi-frequently rethink how something works and introduce breaking changes. I've
put this on GitHub in case anyone finds something in this interesting. You'll
find that the frequency of my commits is in direct proportion to how close I am
to a big visuals gig.

**Jonathan Hogg**
<me@jonathanhogg.com>

### A note about the name

Much like the tortured path of the software, the name has evolved over time.
The *very* original version of this idea was called **flight**, which stood for
"functional lighting", as my first 2D visuals system was actually an extension
of a framework I wrote to control DMX lights. The name **flitter** was a sort
of extension of this from the standard meaning of "flitter", which is to fly
back and forth. However, "flitter" has a number of other meanings in different
dialects of old Scots (I am a secret Scot), including "a small shiny piece of
metal" – like a sequin. I like the name encompassing both movement and light.

## Requirements

At least Python 3.10 is *required* as the code uses `match`/`case` syntax. I
work exclusively in 3.11, so I may have introduced some other hidden dependency
on this later version.

Install flitter and all of its requirements with:

```sh
pip3 install .
```

or some other suitable PEP 517 build mechanism. I'd recommend doing this in a
virtual env, but you do you.

For reference, the runtime dependencies are:

- `numpy` - for fast memory crunching
- `lark` and `regex` - for the language parser
- `rtmidi2` - for talking MIDI
- `pyusb` - for sending screen data to the Push 2 and for talking to LaserCubes
- `skia-python` - for 2D drawing
- `glfw` - for OpenGL windowing
- `moderngl` - because the raw OpenGL API is too low-level
- `trimesh` - for generating/loading 3D triangular mesh models
- `scipy` - because `trimesh` needs it for some operations
- `av` - for encoding and decoding video
- `pillow` - for saving screenshots as image files
- `mako` - for templating of the GLSL source
- `pyserial` - for talking to DMX interfaces and lasers
- `loguru` - because the standard `logging` library is just too antiquated

and the install-time dependencies are:

- `cython` - because half of **flitter** is implemented in Cython for speed

If you want to muck about with the code then ensure Cython is installed in
your runtime environment, do an editable package deployment, throw away the
built code and let `pyximport` (re)compile it on-the-fly as you go:

```sh
pip3 install --editable .
rm **/*.c **/*.so
```

You might also want to install `flake8` and `pytest`, which is what I use for
linting the code and running the (few) unit tests.

## Running

Along with the `flitter` Python package, the `flitter` command-line script
should be installed. This can be used to execute a **flitter** script with:

```sh
flitter examples/hello.fl
```

You might want to add the `--verbose` option to get some more logging. You can
see a full list of available options with `--help`.

Everything else there is to know can be found in the [docs folder](/docs) and
in the code.

### A note on multi-processing

I love Python, but the global interpreter lock basically makes any kind of
serious multi-threading impossible. **flitter** supports a limited form of
multi-processing instead. Adding the `--multiprocess` option on the command line
will cause a separate renderer child process to be executed for each window,
laser or DMX bus.

The main process handles evaluating the script to produce an output tree. This
tree is then fed to each renderer and all renderers are waited on before moving
on to the next iteration. This works well if the script and the renderers are
expensive to run, but the tree is small. As the tree grows large, the cost of
pickling and un-pickling it across the process boundaries becomes a bottleneck,
so your mileage may vary.
