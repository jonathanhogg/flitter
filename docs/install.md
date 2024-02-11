
# Installing and Running Flitter

**Flitter** is implemented in a mix of Python and Cython and requires OpenGL
3.3 or above. At least Python 3.10 is *required* as the code uses `match`/`case`
syntax.

I develop and use it exclusively on Intel macOS. I have done some limited
testing on an Intel Ubuntu VM and on Apple Silicon and it seems to run fine on
both of those platforms. I've not heard of anyone trying it on Windows yet, but
there's no particular reason why it shouldn't work. If you have success or
otherwise on another platform please let me know / raise an issue.

If you want to try **Flitter** then you can install the latest [`flitter-lang`
PyPI package](https://pypi.org/project/flitter-lang/) with:

```sh
pip3 install flitter-lang
```

and then run it with:

```sh
flitter path/to/some/flitter/script.fl
```

It is recommended to do this in a Python [virtual env](https://docs.python.org/3/library/venv.html).

If you want to live on the bleeding edge, then you can install from the current
head of the `main` branch with:

```sh
pip3 install https://github.com/jonathanhogg/flitter/archive/main.zip
```

However, if you clone this repo instead, then you can install from the top
level directory:

```sh
git clone https://github.com/jonathanhogg/flitter.git
cd flitter
pip3 install .
```

and have direct access to the example programs:

```sh
flitter examples/hoops.fl
```

You might want to add the `--verbose` option to get some more logging. You can
see the full list of available options with `--help`.

## Install and runtime dependencies

The first-level runtime dependencies are:

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

and the install-time dependencies are:

- `cython` - because half of **Flitter** is implemented in Cython for speed
- `setuptools` - to run the build file

## Editable installations

If you want to muck about with the code then ensure that `cython` and
`setuptools` are installed in your runtime environment, do an editable
package deployment, and then throw away the built code and let `pyximport`
(re)compile it on-the-fly as you go:

```sh
pip3 install cython setuptools
pip3 install --editable .
rm **/*.c **/*.so
```

If you want to lint the code and run the tests then you might also want to
install `flake8`, `cython-lint` and `pytest` (plus `pytest-xvfb` on Linux if
you want to run the functional tests without a windowing environment).
