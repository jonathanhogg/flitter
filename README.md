![Screenshot from a Flitter program showing colourful distorted ellipse shapes
with trails moving outwards from the centre of the screen.](docs/header.jpg)

# Flitter

**Flitter** is a functional programming language wrapped around a declarative
system for describing 2D and 3D visuals. [The language](/docs/language.md) is
designed to encourage an iterative, explorative, play-based approach to
constructing generative visuals. The engine that runs **Flitter** programs is
able to live reload all code and assets while retaining current system state
(thus supporting live-coding) while also having strong support for interacting
with running programs via MIDI surfaces.

**Flitter** is designed for expressivity and ease of engine development over
raw performance, but is sufficiently fast to be able to do interesting things.

The engine that runs the language is capable of:

- 2D drawing (loosely based on an HTML canvas/SVG model)
- 3D rendering of primitive shapes and external triangular mesh models (in a
variety of formats including OBJ and STL); ambient, directional, point and
spot- light sources with (currently shadowless) [PBR](https://en.wikipedia.org/wiki/Physically_based_rendering)
material shading; simple fog; perspective and orthographic projections;
texture-mapping - including with the output of other visual units (like a
drawing canvas or a video)
- simulating simple [physical particle systems](/docs/physics.md) (including
spring/rod/rubber-band constraints, gravity, electrostatic charge, inertia,
drag, barriers and particle collisions)
- playing videos at arbitrary speeds (including in reverse, although video will
stutter if it makes extensive use of P-frames)
- running GLSL shaders as stacked image generators and filters, with live
manipulation of uniforms and live reload of source
- compositing all of the above and rendering to one or more windows
- saving rendering output to image and video files (including the lockstep
frame-by-frame video output suitable for producing perfect loops)
- driving arbitrary DMX fixtures via a USB DMX interface (currently via an
Entec-compatible interface or my own crazy hand-built devices)
- driving a LaserCube plugged in over USB (other lasers probably easy-ish to
support)
- taking live inputs from Ableton Push 2 or Behringer X-Touch mini MIDI
surfaces (other controllers relatively easy to add)

Fundamentally, the system repeatedly evaluates a program with a beat counter
and the current system state. The output of the program is a tree of nodes that
describe visuals to be rendered, systems to be simulated and control interfaces
to be made available to the user. A series of renderers turn the nodes
describing the visuals into 2D and 3D drawing commands (or DMX packets, or laser
DAC values).

## Installing/running

**Flitter** is implemented in a mix of Python and Cython and requires OpenGL
3.3 or above. At least Python 3.10 is *required* as the code uses `match`/`case`
syntax.

I develop and use it exclusively on Intel macOS. I have done some limited
testing on an Intel Ubuntu VM and on Apple Silicon and it seems to run fine on
both of those platforms. I've not heard of anyone trying it on Windows yet, but
there's no particular reason why it shouldn't work. If you have success or
otherwise on another platform please let me know / raise an issue.

If you want to try it out without cloning the repo, then you can install and
try it **right now** with:

```sh
pip3 install https://github.com/jonathanhogg/flitter/archive/main.zip
```

and then:

```sh
flitter path/to/some/flitter/script.fl
```

I'd recommend doing this in a Python [virtual env](https://docs.python.org/3/library/venv.html),
but you do you. Sadly, you won't have the examples handy doing it this way.

If you clone this repo, then you can install from the top level directory:

```sh
git clone https://github.com/jonathanhogg/flitter.git
cd flitter
pip3 install .
```

Then you can run one of the examples easily with:

```sh
flitter examples/hoops.fl
```

You might want to add the `--verbose` option to get some more logging. You can
see the full list of available options with `--help`.

### Install and runtime dependencies

The first-level runtime dependencies are:

- `av` - for encoding and decoding video
- `glfw` - for OpenGL windowing
- `lark` - for the language parser
- `loguru` - for enhanced logging
- `mako` - for templating of the GLSL source
- `manifold3d` - for sold-model boolean operations
- `moderngl` - for a higher-level API to OpenGL
- `numpy` - for fast memory crunching
- `pillow` - for saving screenshots as image files
- `pyserial` - for talking to DMX interfaces
- `pyusb` - for low-level communication with the Push 2 and LaserCube
- `regex` - used by `lark` for advanced regular expressions
- `rtmidi2` - for talking MIDI to control surfaces
- `scipy` - for computing convex hulls
- `skia-python` - for 2D drawing
- `trimesh` - for loading 3D meshes

and the install-time dependencies are:

- `cython` - because half of **Flitter** is implemented in Cython for speed
- `setuptools` - to run the build file

### Editable installations

If you want to muck about with the code then ensure that `cython` and
`setuptools` are installed in your runtime environment, do an editable
package deployment, and then throw away the built code and let `pyximport`
(re)compile it on-the-fly as you go:

```sh
pip3 install cython setuptools
pip3 install --editable .
rm **/*.c **/*.so
```

You might also want to install `flake8` and `pytest`, which is what I use for
linting the code and running the (few) unit tests.

## Learning Flitter

There is *some* documentation available in the [docs folder](/docs) and a few
quick [examples](/examples) ready to run out-of-the-box.

However, there is also a separate repo containing [many more interesting
examples](https://github.com/jonathanhogg/flitter-examples) that are worth
checking out.
