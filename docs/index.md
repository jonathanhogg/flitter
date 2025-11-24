
![Screenshot from a Flitter program showing colourful distorted ellipse shapes
with trails moving outwards from the centre of the screen.](header.jpg)

# Flitter

**Flitter** is a functional programming language and declarative system for
describing 2D and 3D visuals. [The language](language.md) is designed to
encourage an iterative, explorative, play-based approach to constructing
visuals.

The engine is able to live reload all code and assets (including shaders,
images, videos, models, etc.) while retaining the current system state - thus
supporting live-coding. It also has support for interacting with running
programs via MIDI surfaces.

**Flitter** is designed for expressivity and ease of engine development over
raw performance, but is fast enough to be able to do interesting things.

## Documentation

```{toctree}
:maxdepth: 2
background.md
install.md
tutorial.md
language.md
builtins.md
windows.md
shaders.md
canvas.md
canvas3d.md
counters.md
physics.md
controllers.md
```

## Examples

Some simple examples are [included in the
source](https://github.com/jonathanhogg/flitter/tree/main/examples)
but many more can be found at the [Flitter examples
repo](https://github.com/jonathanhogg/flitter-examples).

## Papers and presentations

There is [a paper discussing the background and design of
Flitter](https://2025.algorithmicpattern.org/proceedings/BG8MYU/paper.html)
that was presented at the Alpaca 2025 conference. You can also [watch a video of
the accompanying talk](https://www.youtube.com/watch?v=D9khHD9sB7M&list=PLxqmZjMvoVzw773-Fo9ajkujFfOThuFOP&index=9),
which demonstrates writing a simple Flitter program from scratch to generate
some animated 3D visuals.

## License and Development

**Flitter** is copyright © Jonathan Hogg and licensed under a [2-clause
"simplified" BSD
license](https://github.com/jonathanhogg/flitter/blob/main/LICENSE)
except for the OpenSimplex 2S noise implementation, which is based on
[code](https://code.larus.se/lmas/opensimplex) copyright © A. Svensson and
licensed under an [MIT
license](https://code.larus.se/lmas/opensimplex/src/branch/master/LICENSE).

The complete source code is available in the [GitHub Flitter
repo](https://github.com/jonathanhogg/flitter), which also hosts the
[issues tracker](https://github.com/jonathanhogg/flitter/issues). The official
release packages are available in the [PyPI `flitter-lang`
package](https://pypi.org/project/flitter-lang/).
