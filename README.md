![Screenshot from a flitter program showing colourful distorted ellipse shapes
with trails moving outwards from the centre of the screen.](docs/header.jpg)

# flitter

**flitter** is a [2D and 3D visuals language](/docs/language.md) and engine
designed for live performances. While **flitter** supports basic live-coding
(live reload of source files while maintaining system state), it is designed
primarily for driving via a MIDI surface.

The engine that runs the language is capable of:

- 2D drawing (loosely based on the HTML canvas/SVG model)
- 3D rendering of primitive shapes and external triangular mesh models (in a
bunch of formats including OBJ and STL); ambient, directional, point and
spot- light sources with (currently shadowless) Phong lighting/shading; simple
fog; perspective and orthographic projections; texture-mapping with the output
of other visual units (like a drawing canvas or a video)
- simulating simple [physical particle systems](/docs/physics.md) (including
gravity, electrostatic charge, inertia, drag and collisions) and hooking the
results up to drawing instructions
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

Fundamentally, the system repeatedly evaluates a program with a beat counter
and the current state. The output of the program is a tree of nodes that
describe visuals to be rendered, systems to be simulated and interfaces to be
made available to the user. A series of renderers turn the nodes describing
the visuals into 2D and 3D drawing commands (or DMX packets, or laser DAC
values). It's sort of like using a functional language to build a web page
DOM - something akin to React.

## Installing/running

**flitter** is implemented in a mix of Python and Cython. I develop and use it
exclusively on macOS. It is notionally portable – in that there's no particular
reason why it wouldn't work on Linux or Windows – but I've not tested either of
those platforms. Please give it a go and raise any issues you come across, or
get in touch if - amazingly - it just works.

If you want to try it out without cloning the repo, then you can install and
try it **right now** with:

```sh
pip3 install https://github.com/jonathanhogg/flitter/archive/main.zip
```

and then:

```sh
flitter path/to/some/flitter/script.fl
```

I'd recommend doing this in a Python virtual env, but you do you. Sadly, you
won't have the examples handy doing it this way.

If you clone the repo, then you can install from the top level directory with:

```sh
pip3 install .
```

Then you can run one of the examples easily with:

```sh
flitter examples/hoops.fl
```

You might want to add the `--verbose` option to get some more logging. You can
see a full list of available options with `--help`.

Everything else there is to know can be found in the [docs folder](/docs),
examples or in the code.

## More examples

As well as the few [examples](/examples) in this repo, there is a dedicated
repo containing [more examples](https://github.com/jonathanhogg/flitter-examples).

## Requirements

At least Python 3.10 is *required* as the code uses `match`/`case` syntax.

The full runtime dependencies are:

- `av` - for encoding and decoding video
- `glfw` - for OpenGL windowing
- `lark` - for the language parser
- `loguru` - for enhanced logging
- `mako` - for templating of the GLSL source
- `moderngl` - for a higher-level API to OpenGL
- `numpy` - for fast memory crunching
- `pillow` - for saving screenshots as image files
- `pyserial` - for talking to DMX interfaces
- `pyusb` - for low-level communication with the Push 2 and LaserCube
- `regex` - used by `lark` for advanced regular expressions
- `rtmidi2` - for talking MIDI to control surfaces
- `scipy` - used by `trimesh` for some operations
- `skia-python` - for 2D drawing
- `trimesh` - for generating/loading 3D meshes

and the install-time dependencies are:

- `cython` - because half of **flitter** is implemented in Cython for speed
- `setuptools` - to run the build file

If you want to muck about with the code then ensure Cython is installed in
your runtime environment, do an editable package deployment, throw away the
built code and let `pyximport` (re)compile it on-the-fly as you go:

```sh
pip3 install cython
pip3 install --editable .
rm **/*.c **/*.so
```

You might also want to install `flake8` and `pytest`, which is what I use for
linting the code and running the (few) unit tests.

## Background

This is probably my third implementation of a variant of these ideas.
Originally, I developed a simple visuals system as an embedding in Python
(using a crazy system of `with` statements), which sent JSON graphs over a
WebSocket to a JavaScript web app that rendered the results in an HTML 2D
canvas. This was actually a three-layer system that involved running the Python
to build a graph, with the graph using a small expression language that allowed
evaluating the attribute values with respect to time, and then the final static
graph would be rendered. I quickly found this too limiting and wanted a more
expressive, single combined language.

This current version was initially developed over a furious fortnight in
October 2021 leading up to a live performance at the Purcell Room, London
Southbank Centre, doing visuals for Bishi at her 'Let My Country Awake' album
launch. The work was partially supported by Ableton, who gave me an artist
discount on a Push 2. I've been working on **flitter** off-and-on since then
developing it as a live tool.

While **flitter** supports live coding, that's not primarily why or how I
designed it. As a programmer, I enjoy using code to create visual artwork and I
am fascinated by how generating all of the visuals live affords me the ability
to improvise a performance. However, I'm interested in complex, narrative
visuals that do not lend themselves to coding in-the-moment. Thus, I have in no
way optimised the language for this purpose - meaning it is probably too verbose
for performative coding (and I haven't bothered building a mechanism for
overlaying code onto the output).

I spend a huge amount of time in advance of a performance thinking and designing
the visuals. In this period I will constantly iterate on the code, and there I
really value the ability to immediately see the effects of code changes.
However, during a performance I prefer to use physical knobs and buttons to riff
on the themes I have developed in advance. So **flitter** is designed to
interface with a MIDI surface and then provide different ways to parameterise
the running code, and manipulate and alter the graph that describes the visuals.

Nothing about **flitter** is in any sense "finished". It is still a testbed for
my ideas. The language and graph semantics are in near constant flux and I
semi-frequently rethink how something works and introduce breaking changes. I've
put this on GitHub in case anyone finds something in this interesting. You'll
find that the frequency of my commits is in direct proportion to how close I am
to a gig.

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
