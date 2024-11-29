![Screenshot from a Flitter program showing colourful distorted ellipse shapes
with trails moving outwards from the centre of the screen.](https://github.com/jonathanhogg/flitter/raw/main/docs/header.jpg)

# Flitter

[![CI lint](https://github.com/jonathanhogg/flitter/actions/workflows/ci-lint.yml/badge.svg?)](https://github.com/jonathanhogg/flitter/actions/workflows/ci-lint.yml)
[![CI test](https://github.com/jonathanhogg/flitter/actions/workflows/ci-test.yml/badge.svg?)](https://github.com/jonathanhogg/flitter/actions/workflows/ci-test.yml)
[![CI coverage](https://gist.githubusercontent.com/jonathanhogg/b7237d8b4e7ff50c3f284cb939e949d0/raw/badge.svg?)](https://github.com/jonathanhogg/flitter/actions/workflows/ci-coverage.yml)
[![docs](https://readthedocs.org/projects/flitter/badge/?version=latest)](https://flitter.readthedocs.io/en/latest/?badge=latest)

**Flitter** is a functional programming language and declarative system for
describing 2D and 3D visuals. It is designed to encourage an iterative,
explorative, play-based approach to constructing visuals.

[The language](https://flitter.readthedocs.io/en/latest/language.html) supports
the basic range of functional language features: first-class recursive and
anonymous functions, comprehensions, let/where, conditional expressions, lists
("vectors"). However, unusually, all values are vectors and all operators are
element-wise, and the language is built around constructing trees of attributed
nodes. The language is designed to be familiar to Python programmers.

The engine is able to live reload all code and assets (including any shaders,
images, videos, models, etc.) while retaining the current system state - thus
supporting live-coding. It also has support for interacting with running
programs via MIDI surfaces (plus basic pointer and keyboard support).

**Flitter** is implemented in a mix of Python and Cython and requires at least
OpenGL 3.3 (Core Profile) or OpenGL ES 3.0. At least Python 3.10 is also
required as the code uses `match`/`case` syntax.

**Flitter** is designed for expressivity and ease of engine development over
raw performance, but is fast enough to be able to do interesting things.

The engine that runs the language is capable of:

- [2D drawing](https://flitter.readthedocs.io/en/latest/canvas.html) (loosely
  based on an HTML canvas/SVG model)
- [3D rendering](https://flitter.readthedocs.io/en/latest/canvas3d.html),
  including:
  - primitive box, sphere, cylinder and cone shapes
  - external triangular mesh models in a variety of formats including OBJ
    and STL
  - planar trimming, union, difference and intersection of solid models
  - construction of meshes from signed distance fields, including common
    combinators and blending functions, and the ability to specify custom
    functions
  - ambient, directional, point/sphere, line/capsule and spotlight sources
    (currently shadowless)
  - multiple (simultaneous) cameras with individual control over location,
    field-of-view, clip planes, render buffer size, color depth, MSAA samples,
    perspective/orthographic projection, fog, conversion to monochrome and
    colour tinting
  - PBR forward-rendering pipeline with emissive objects, transparency and
    translucency, plus the ability to plug in custom GLSL shaders for arbitrary
    groups of objects
  - texture mapping, including with the output of other visual units (e.g., a
    drawing canvas or a video)
- simulating [physical particle
  systems](https://flitter.readthedocs.io/en/latest/physics.html), including
  spring/rod/rubber-band constraints, gravity, electrostatic charge, adhesion,
  buoyancy, inertia, drag (including in flowing media), Brownian motion,
  uniform electric fields, barriers and particle collisions
- [playing videos](https://flitter.readthedocs.io/en/latest/windows.html#video)
  at arbitrary speeds (including in reverse)
- running [GLSL
  shaders](https://flitter.readthedocs.io/en/latest/shaders.html) as
  stacked image filters and generators, with per-frame control of arbitrary
  uniforms, and support for multi-pass and downsampling
- built-in filters for: scaling/translating/rotating, Gaussian blurring, bloom,
  edge detection, vignetting, video feedback, lens flares, color and exposure
  adjustments, tone-mapping with the Reinhard and ACES Filmic functions, and 2D
  noise-map generation
- compositing all of the above and rendering to one or more windows
- [saving rendered
  output](https://flitter.readthedocs.io/en/latest/windows.html#record) to
  image and video files (including lockstep frame-by-frame video output
  suitable for producing perfect loops and direct generation of animated GIFs)
- taking live inputs from Ableton Push 2 or Behringer X-Touch mini MIDI
  surfaces (other controllers relatively easy to add)
- driving arbitrary DMX fixtures via an Entec-compatible USB DMX interface
- driving a LaserCube plugged in over USB (other lasers probably easy-ish to
  support)

**Flitter** also has a plug-in architecture that allows extension with new
image and 3D mesh generators, MIDI and DMX interfaces, or completely novel
input and output systems.

## Installation

**Flitter** can be installed from the [`flitter-lang` PyPI
package](https://pypi.org/project/flitter-lang/)  with:

```sh
pip3 install flitter-lang
```

and then run as:

```sh
flitter path/to/some/flitter/script.fl
```

More details can be found in the [installation
documentation](https://flitter.readthedocs.io/en/latest/install.html).

## Documentation

The documentation is available on [**Read** *the*
**Docs**](https://flitter.readthedocs.io/).

There are a few quick
[examples](https://github.com/jonathanhogg/flitter/blob/main/examples)
in the main repository. However, there is also a separate repo containing [many
more interesting examples](https://github.com/jonathanhogg/flitter-examples)
that are worth checking out.

## License

**Flitter** is copyright © Jonathan Hogg and licensed under a [2-clause
"simplified" BSD
license](https://github.com/jonathanhogg/flitter/blob/main/LICENSE)
except for the OpenSimplex 2S noise implementation, which is based on
[code](https://code.larus.se/lmas/opensimplex) copyright © A. Svensson and
licensed under an [MIT
license](https://code.larus.se/lmas/opensimplex/src/branch/master/LICENSE).
