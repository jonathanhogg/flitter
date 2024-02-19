
# Installing and Running

**Flitter** is implemented in a mix of Python and Cython and requires OpenGL
3.3 or above. At least Python 3.10 is *required* as the code uses `match`/`case`
syntax.

It is developed exclusively on macOS Intel. Some limited testing has been done
on an Ubuntu Intel VM and on Apple Silicon and it seems to run fine on
both of those platforms. There have been no reports of it having been tried
on Windows yet, but there's no particular reason why it shouldn't work. If you
have success or otherwise on another platform please get in touch.

## Install precursors

**Flitter** is a command-line tool. It is assumed that you are comfortable using
the command line on your OS of choice. You will also obviously need to be able
to ensure that you have a recent Python install. Sadly, even on macOS Sonoma
the system installed Python is only at version 3.9. You can normally download
Python for your OS from the [Python website](https://www.python.org/downloads/).
However, you may want to explore a package manager that can manage this sort of
thing for you, like [Homebrew](https://brew.sh) or
[MacPorts](https://www.macports.org/install.php). As **Flitter** uses Cython
under-the-hood, you'll also need a working compiler environment. On macOS this
means downloading [Xcode](https://developer.apple.com/xcode/).

On a recent Linux box, all you should need to do is make sure that you have a
Python3 development environment. On a Debian-variant, like Ubuntu, this would
just be:

```console
$ sudo apt install python3-dev
```

It is recommended to do the install into a Python [virtual
environment](https://docs.python.org/3/library/venv.html) as **Flitter** draws
on quite a few dependencies (see below). This is especially important if you are
using the latest Python 3.12 as it will simply refuse to install packages into
the system environment. This is generally something as simple as:

```console
$ python3 -m venv ~/.virtualenvs/flitter
```

However, then ensuring that `~/.virtualenvs/flitter/bin` is in your `PATH` is
left as an exercise for the reader.

## Installing the `flitter-lang` package

If you've safely navigated getting a working Python development environment, you
can install you can install the latest [`flitter-lang` PyPI
package](https://pypi.org/project/flitter-lang/) with:

```console
$ pip3 install flitter-lang
```

and then run it with:

```console
$ flitter path/to/some/flitter/script.fl
```

If you want to live on the bleeding edge, then you can install from the current
head of the `main` branch with:

```console
$ pip3 install https://github.com/jonathanhogg/flitter/archive/main.zip
```

However, if you clone the repo instead, then you can install from the top
level directory:

```console
$ git clone https://github.com/jonathanhogg/flitter.git
$ cd flitter
$ pip3 install .
```

keep up-to-date with developments and have direct access to the example
programs:

```console
$ flitter examples/hoops.fl
```

## Python package dependencies

The first-level runtime Python package dependencies are listed below. These will
all be installed for you by `pip`, but it's useful to know what you're getting
into.

- `av` - for encoding and decoding video
- `glfw` - for OpenGL windowing
- `lark` - for the language parser
- `loguru` - for enhanced logging
- `mako` - for templating of the GLSL source
- `manifold3d` - used by `trimesh` for 3D boolean operations
- `mapbox_earcut` - used by `trimesh` for triangulating polygons
- `moderngl` - for a higher-level API to OpenGL
- `networkx` - used by `trimesh` for graph algorithms
- `numpy` - for fast memory crunching
- `pillow` - for saving screenshots as image files
- `pyserial` - for talking to DMX interfaces
- `pyusb` - for low-level communication with the Push 2 and LaserCube
- `regex` - used by `lark` for advanced regular expressions
- `rtmidi2` - for talking MIDI to control surfaces
- `rtree` - used by `trimesh` for spatial tree intersection
- `scipy` - used by `trimesh` for computing convex hulls
- `shapely` - used by `trimesh` for polygon operations
- `skia-python` - for 2D drawing
- `trimesh` - for loading 3D meshes

During install, `pip` will also use:

- `cython` - because half of **Flitter** is implemented in Cython for speed
- `setuptools` - to run the build file

## Editable installations

If you want to edit the code then ensure that `cython` and `setuptools` are
installed in your runtime environment, do an editable package deployment, and
then throw away the built code. The code automatically makes use of `pyximport`
to (re)compile Cython code on-the-fly as **Flitter** runs:

```console
$ pip3 install cython setuptools
$ pip3 install --editable .
$ rm **/*.c **/*.so
```

If you want to lint the code and run the tests then you might also want to
install `flake8`, `cython-lint` and `pytest`.

## Flitter command-line options

**Flitter** takes one or more program filenames as the main command-line
arguments. If multiple programs are given then they are loaded into separate
*pages* and the first will be executed. By default, you can switch between
pages by pressing the left- and right-arrow keys while a window has focus.
However, full control over page switching is actually part of the [controller
infrastructure](controllers.md) of Flitter.

The supported command-line options are:

`--help`
: See a list of these options.

`--quiet` | `--verbose` | `--debug` | `--trace`
: Change the amount of logging that is written out, from least to most noisy.
`--verbose` is useful for getting some basic runtime statistics, `--debug`
shows a little of what is going on under-the-hood, `--trace` is incredibly
noisy as it logs lots of object allocation and deallocation activity.

`--fps` *FPS*
: Specify a target frame-rate for the engine to try and maintain. This defaults
to 60fps, but it's useful to be able to lower it if you are running an
especially complex script and would prefer to run evenly at a slower rate than
to have stuttering. It's also useful when recording video output. You can also
control frame-rate via a [pragma](language.md#pragmas).

`--define` *NAME*`=`*VALUE*
: This allows a name to bound to a constant value in the language interpreter.
This is useful for controlling conditional behaviour in a program, such as
enabling video output or changing the window size. *VALUE* may be either a
string or a numeric vector, with items separated by semicolons. You will need to
appropriately quote it on the command-line if using semicolons.

`--screen` *SCREEN*
: Allows you to ask **Flitter** to open windows on a distinct screen number.
This depends on your desktop setup, but generally screen 0 is the main one and
they count upwards from there. Specific windows can also be placed on specific
screens by adding a `screen=N` attribute to the `!window` node. This option
controls the default if that is not done.

`--fullscreen`
: Requests that windows are opened full-screen by default. Again, this can be
controlled on a per-window basis with the `fullscreen=true` attribute to a
`!window` node.

`--vsync`
: Requests that windows use vertical-sync double-buffering. Also the
`vsync=true` attribute.

`--state` *FILENAME*
: Asks **Flitter** to save the system state periodically to a file and load
again from this. This allows you to retain state across multiple sessions. This
was primarily designed as a safety measure for reacting quickly to a crash
during a live performance. Thankfully, **Flitter** is pretty resilient to
crashes, but there are still some edge-cases where a code change can confuse
the engine.

`--autoreset` *SECONDS*
: Throws away the internal system state after a period of it not changing. This
was designed to be used as a demo mode, enabling users to interact with a
running program via a controller and then have their fiddlings discarded after
they walk away.

`--simplifystate` *SECONDS*
: Calls the [language simplifier](language.md#simplification) after a period of
state idleness.

`--lockstep`
: Turns on *non-realtime mode*. In this mode, the engine will generate frames
as fast as possible (which may be quite slowly) while maintaining an evenly
incrementing beat counter and frame clock. This is designed for saving video
output from non-interactive programs. It also has a specific [effect on how the
physics system runs](physics.md#non-realtime-mode).

`--runtime` *SECONDS*
: Run for a specific period of time and then exit automatically. This is
particularly useful for capturing videos of a specific length unattended. In
`--lockstep` mode, this is a period of the internal frame clock not the wall
clock.

`--offscreen`
: Requests that all `!window` nodes actually be silently interpreted as
`!offscreen` nodes. This has the effect of running **Flitter** without any
visual output. It assumes offscreen access to the GPU. On macOS this just works,
on Linux you'll still need to provide a windowing environment even if it's
a virtual one with something like `Xvfb`.

`--profile`
: Runs the Python profiler around the main loop. This will slow down **Flitter**
significantly. If you exit with an interrupt (ctrl-C) then a summary of the
internal functions called will be shown. This won't be a great deal of use to
you unless you are hacking on the **Flitter** code.

`--vmstats`
: Turns on logging of **Flitter** virtual machine statistics. This will slow
down the interpreter by quite a lot. The statistics are written out on program
interrupt at the `INFO` logging level, which means you will also need to on
at least `--verbose` logging to see the results.

`--gamma` *GAMMA*
: This option specifies a gamma correction to apply as the final step of
compositing window contents. It should not be necessary on a decently-configured
desktop environment, but is available in case the output device is less well set
up (perhaps some random dodgy projector in a venue) and the visual output has a
bad gamma curve. Numbers smaller than 1 will brighten the image and numbers
greater than 1 will darken it. In particular, if the output looks way too dark
then it's possible that the standard sRGB transfer function is being applied
twice, you can undo this with `--gamma 0.45`. Too bright and you might want to
start with `--gamma 2.2`. As usual, you can control this per-window with the
`gamma=N` attribute of `!window`.
