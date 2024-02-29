![Screenshot from a Flitter program showing colourful distorted ellipse shapes
with trails moving outwards from the centre of the screen.](https://github.com/jonathanhogg/flitter/raw/main/docs/header.jpg)

# Flitter

[![CI lint](https://github.com/jonathanhogg/flitter/actions/workflows/ci-lint.yml/badge.svg?)](https://github.com/jonathanhogg/flitter/actions/workflows/ci-lint.yml)
[![CI test](https://github.com/jonathanhogg/flitter/actions/workflows/ci-test.yml/badge.svg?)](https://github.com/jonathanhogg/flitter/actions/workflows/ci-test.yml)
[![docs](https://readthedocs.org/projects/flitter/badge/?version=latest)](https://flitter.readthedocs.io/en/latest/?badge=latest)

**Flitter** is a functional programming language and declarative system for
describing 2D and 3D visuals. [The
language](https://flitter.readthedocs.io/en/latest/language.html) is designed
to encourage an iterative, explorative, play-based approach to constructing
visuals.

The engine is able to live reload all code and assets (including shaders,
images, videos, models, etc.) while retaining the current system state - thus
supporting live-coding. It also has support for interacting with running
programs via MIDI surfaces.

**Flitter** is designed for expressivity and ease of engine development over
raw performance, but is fast enough to be able to do interesting things.

The engine that runs the language is capable of:

- 2D drawing (loosely based on an HTML canvas/SVG model)
- 3D rendering, including:
  - primitive box, sphere, cylinder and cone shapes
  - external triangular mesh models in a variety of formats including OBJ
    and STL
  - texture mapping, including with the output of other visual units (e.g., a
    drawing canvas or a video)
  - planar slicing and union, difference and intersection of solid models
  - ambient, directional, point/sphere, line/capsule and spotlight sources
    (currently shadowless)
  - Physically-based rendering material shading, emissive objects and
    transparency
  - multiple cameras with individual control over location, field-of-view, near
    and far clip planes, render buffer size, color depth, MSAA samples,
    perspective/orthographic projection, fog, conversion to monochrome and
    colour tinting
- simulating [physical particle
systems](https://flitter.readthedocs.io/en/latest/physics.html), including
spring/rod/rubber-band constraints, gravity, electrostatic charge, adhesion,
buoyancy, inertia, drag, barriers and particle collisions
- playing videos at arbitrary speeds, including in reverse (although video will
stutter if it makes extensive use of P-frames)
- running GLSL shaders as stacked image filters and generators, with per-frame
control of arbitrary uniforms
- compositing all of the above and rendering to one or more windows
- saving rendered output to image and video files (including lockstep
frame-by-frame video output suitable for producing perfect loops and direct
generation of animated GIFs)
- driving arbitrary DMX fixtures via a USB DMX interface (currently via an
Entec-compatible interface or my own crazy hand-built devices)
- driving a LaserCube plugged in over USB (other lasers probably easy-ish to
support)
- taking live inputs from Ableton Push 2 or Behringer X-Touch mini MIDI
surfaces (other controllers relatively easy to add)

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

The documentation is available on the [Flitter **Read** *the* **Docs**
pages](https://flitter.readthedocs.io/).

There are a few quick
[examples](https://github.com/jonathanhogg/flitter/blob/main/examples)
in the main repository. However, there is also a separate repo containing [many
more interesting examples](https://github.com/jonathanhogg/flitter-examples)
that are worth checking out.
