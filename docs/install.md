
# Installing and Running

**Flitter** is implemented in a mix of Python and Cython and requires at least
OpenGL 3.3 (Core Profile) or OpenGL ES 3.0. At least Python 3.10 is also
required as the code uses `match`/`case` syntax. It is supported and tested on
Apple Silicon macOS, Intel macOS, x86_64 Linux, AArch64 Linux and x86_64
Windows.

**Flitter** is a command-line tool. It is assumed that you are comfortable using
the command line on your OS of choice. You will also obviously need to be able
to ensure that you have a recent Python install. Sadly, even on macOS Sequoia
the system installed Python is only at version 3.9. You can normally download
Python for your OS from the [Python website](https://www.python.org/downloads/).
On a Mac you may want to explore a package manager that can manage this sort of
thing for you, like [Homebrew](https://brew.sh) or
[MacPorts](https://www.macports.org/install.php).

## Installing Flitter

### Installing from the pre-built wheels

If you are installing on one of the supported platforms then, good news!, you
can install one of the pre-built wheels in the [`flitter-lang` PyPI
package](https://pypi.org/project/flitter-lang/) with just:

```console
$ pip3 install flitter-lang
```

Otherwise you are going to need to [install from the source
distribution](#installing-from-the-source-package).

It is recommended to do the install into a Python [virtual
environment](https://docs.python.org/3/library/venv.html) as **Flitter** draws
on quite a few dependencies (see [below](#python-package-dependencies)). This is
especially important if you are using Python 3.12 or later as it will simply
refuse to install packages into the system environment. This is generally
something as simple as:

```console
$ python3 -m venv ~/.virtualenvs/flitter
$ ~/.virtualenvs/flitter/bin/pip3 install flitter-lang
```

Alternatively, do a *user* install with:

```console
$ pip3 install --user flitter-lang
```

Assuming your Python install is set up well for your operating system, this
should just put the `flitter` script into your path for you. However, if this
isn't working (on Windows, for example, you may need to have checked a box
during install) then you can also run **Flitter** as a Python package with:

```console
$ python3 -m flitter.engine path/to/some/flitter/script.fl
```

### macOS and sRGB

**Flitter** assumes that OpenGL windows are in the sRGB colorspace. However,
modern Macs generally have displays that support the wider Display-P3
colorspace, and macOS OpenGL windows will default to this.

This results in colors on macOS appearing more vivid than on other platforms.
Colors in windows will not match output saved to image and video files (which
are always tagged as being in the sRGB colorspace), and loaded images or video
will not display accurately. Colors calculated from calibrated color functions
(like `colortemp()` or `oklch()`) will also not be displayed accurately.

**Flitter** is capable of telling macOS to change the window colorspace to
sRGB, but will only do so if the optional, macOS-only libraries `pyobjc` and
`pyobjc-framework-Cocoa` are installed. You can install these manually, or
automatically if you install **Flitter** with:

```console
$ pip3 install "flitter-lang[macos]"
```

### Installing from the source package

As **Flitter** uses Cython under-the-hood, you'll need a working build
environment with a C compiler. On macOS this means downloading
[Xcode](https://developer.apple.com/xcode/). On Windows you'll need [Visual
Studio](https://visualstudio.microsoft.com). On a Debian-variant Linux box, all
you should need to do is install the `python3-dev` package:

```console
$ sudo apt install python3-dev
```

If you've safely navigated getting a working Python development environment, you
can do the build from source with exactly the same install command as above.

### Installing from the repo:

If you want to live on the bleeding edge, then you can build the current
head of the `main` branch with:

```console
$ pip3 install https://github.com/jonathanhogg/flitter/archive/main.zip
```

## Running Flitter

If you've installed **Flitter** in a virtual environment, then you'll need to
ensure that the scripts/bin folder for that environment is in your system path.
You can then run **Flitter** with a command like:

```console
$ flitter path/to/some/flitter/script.fl
```

If you've done a local `pip` install instead then – with a little luck – the
scripts folder is already in your path. Otherwise, you might want to try running
**Flitter** as a Python package:

```console
$ python3 -m flitter.engine path/to/some/flitter/script.fl
```

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
: This allows a name to be bound to a constant value in the language interpreter.
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

`--resetonswitch`
: Throws away the internal system state when switching between different pages.
Normally state is saved and restored when switching pages. Throwing the last
state away can be useful if running in a *kiosk* mode where each page should
begin again fresh.

`--simplifystate` *SECONDS*
: Sets the period after which the [language
simplifier](language.md#simplification) will simplify on static state values, in
seconds (may be specified as a [time-code](language.md#time-codes)). The default
is 10 seconds. Set this to `0` to disable simplifying on state.

`--nosimplify`
: Completely disable the language simplifier. Generally there is no reason to
do this, but running the simplifier is the slowest part of loading **Flitter**
code and so it may be useful if very fast switching is required – particularly
if a program has very large loops that are being unrolled and there is
insufficient gain from doing the unrolling.

`--modelcache`
: How long to cache unused 3D models for, in seconds (may be specified as a
[time-code](language.md#time-codes)). Set this to `0` to disable clearing of the
model cache completely – this may be a useful performance tweak for long-running
programs that reference a fixed number of models.

`--lockstep`
: Turns on *non-realtime mode*. In this mode, the engine will generate frames
as fast as possible (which may be quite slowly) while maintaining an evenly
incrementing beat counter and frame clock. This is designed for saving video
output from non-interactive programs. It also has a specific [effect on how the
physics system runs](physics.md#non-realtime-mode).

`--runtime` *SECONDS*
: Run for a specific period of time and then exit automatically, in seconds
(may be specified as a [time-code](language.md#time-codes)). This is
particularly useful for capturing videos of a specific length unattended. In
`--lockstep` mode, this is a period of the internal frame clock not the wall
clock.

`--offscreen`
: Requests that all `!window` nodes actually be silently interpreted as
`!offscreen` nodes. This has the effect of running **Flitter** without any
visual output. You'll still need a windowed environment. You can conspire to run
on Linux without a real windowed environment using `Xvfb` (and the CI tests are
run just this way), but you should know that the Mesa software rasterizer is
*very, very slow*. **Flitter** really wants a GPU.

`--opengles`
: This tells *Flitter* to request the OpenGL ES API instead of OpenGL Core. You
can use this on devices that do not support the full OpenGL API, or that are
using a compatibility layer like ANGLE. At the moment, `!canvas` 2D drawing is
not supported on OpenGL ES.

`--profile`
: Runs the Python profiler around the main loop. See [Profiling](#profiling)
below for details.

`--vmstats`
: Turns on logging of **Flitter** virtual machine statistics. This will slow
down the interpreter by quite a lot. The statistics are written out on exit
at the `INFO` logging level, which means you will also need to have at least
`--verbose` logging turned on to see the results. Be wary of interpreting these
statistics: the overhead of doing the timing can make very simple instructions
that are executed millions of times appear to be a larger source of execution
cost than they really are.

## Developing Flitter

If you want to edit the **Flitter** code – or just want to keep up-to-date with
current developments – ensure that `cython` and `setuptools` are installed in
your runtime Python environment, clone the repo, do an *editable* package
deployment directly from the clone directory, and then throw away the compiled
files:

```console
$ pip3 install cython setuptools
$ git clone https://github.com/jonathanhogg/flitter.git
$ cd flitter
$ pip3 install --editable .
$ rm src/**/*.c src/**/*.so
```

**Flitter** automatically makes use of `pyximport` to (re)compile Cython code
on-the-fly as it runs. This allows you to edit the code (or pull a new version
from the repo) and just re-run `flitter` to pick up changes.

The code is linted with `flake8` and `cython-lint`:

```console
$ pip3 install flake8 cython-lint
$ flake8 src tests scripts
$ cython-lint src
```

And the test suite can be run with `pytest`:

```console
$ pip3 install pytest
$ pytest
```

### Profiling

**Flitter** has built-in support for running itself with profiling turned on:
just add the `--profile` command line option when running it. This is best
combined with `--lockstep` and `--runtime=N` to run for exactly *N* seconds of
frame time. At the end of execution (including a keyboard interrupt), the
standard Python profiler output will be printed, ordered by `tottime`.

By default, the **Flitter** Cython modules are *not* compiled with profiling
support. This means that only pure Python functions will be listed in the
profiler, with all time spent inside Cython code aggregated into those
functions' runtime. This is not particularly useful as much of the work is
happening in the Cython modules.

Assuming that you are already using an editable install (as described above),
then all of the Cython modules can be compiled with profiling by setting the
`FLITTER_BUILD_PROFILE` environment variable to `1` and running `setup.py`:

```console
$ env FLITTER_BUILD_PROFILE=1 python3 setup.py build_ext --inplace
```

:::{warning}
Profiling of the Cython modules **is not supported in Python 3.12** due to
internal changes in the profiling API. Profiling *is* supported in Python 3.13
and above as long as you have Cython 3.1 or later.
:::

Recompiling all Cython modules with profiling enabled may not be the best thing
to do as the overhead of adding profiling support to Cython is quite high and
the results will be skewed by the cost of the profiling. It is probably better
to *only* compile profiling support into the module(s) that you are considering.
This can be done with `cythonize`:

```console
$ cythonize --build --inplace --force -3 --annotate -X profile=True \
    src/flitter/render/window/canvas3d.pyx
```

Even then, take the profiler results with a pinch of salt. Because of the
additional cost of profiling, small functions that are called very frequently
will show a greater proportion of the runtime cost than they probably actually
contribute.

Remove the compiled versions of any modules that are not being profiled to allow
`pyximport` to generate and use an optimized version.

### Checking code coverage

If you want to run code coverage analysis, then you will need to do a special
in-place build with coverage enabled (Cython line-tracing):

```console
$ env FLITTER_BUILD_COVERAGE=1 python3 setup.py build_ext --inplace
```

This also enables (and requires) profiling support, and so the same caveats
about Python and Cython versions noted above apply.

Once you have a coverage-enabled build, you can generate a code coverage report
for the test suite with:

```console
$ pip3 install coverage
$ coverage run -m pytest
$ coverage report
```

You will need to re-run `setup.py` if you change any of the Cython code. You
will need to delete all of these object files if you want to go back to using
the normal `pyximport` automatic recompile (and you will want to do so as the
coverage-enabled version of the code is *significantly* slower).

### Generating the documentation

To generate a local HTML copy of the documentation, install `sphinx`,
`myst_parser` and the separate [`flitter-pygments`
package](https://github.com/jonathanhogg/flitter-pygments) (which adds syntax
highlighting support for **Flitter** to `pygments`):

```console
$ pip3 install sphinx myst_parser flitter-pygments
```

You can then generate the docs with:

```console
$ sphinx-build docs build/html
```

They can then be read from `build/html/index.html`.

### Python package dependencies

The first-level runtime Python package dependencies are listed below. These will
all be installed for you by `pip`, but it's useful to know what you're getting
into.

- `av` - for encoding and decoding video
- `glfw` - for OpenGL windowing
- `lark` - for the language parser
- `loguru` - for enhanced logging
- `mako` - for templating of the GLSL source
- `manifold3d` - for constructive solid geometry operations
- `moderngl` - for a higher-level API to OpenGL
- `networkx` - used by `trimesh` for graph algorithms
- `numpy` - for fast memory crunching
- `pillow` - for saving screenshots as image files
- `pyserial` - for talking to DMX interfaces
- `pyusb` - for low-level communication with the Push 2 and LaserCube
- `regex` - used by `lark` for advanced regular expressions
- `rtmidi2` - for talking MIDI to control surfaces
- `scipy` - used by `trimesh` for computing convex hulls
- `skia-python` - for 2D drawing
- `trimesh` - for loading 3D meshes

During install, `pip` will also use:

- `cython` - because half of **Flitter** is implemented in Cython for speed
- `setuptools` - to run the build file
