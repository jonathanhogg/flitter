
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
:maxdepth: 1
background.md
install.md
language.md
builtins.md
windows.md
physics.md
```

## Examples

Some simple examples are [included in the
source](https://github.com/jonathanhogg/flitter/tree/main/examples)
but many more can be found at the [Flitter examples
repo](https://github.com/jonathanhogg/flitter-examples).

## License and Development

**Flitter** is released under a [2-clause "simplified" BSD
license](https://github.com/jonathanhogg/flitter/blob/main/LICENSE).

The complete source code is available in the [GitHub Flitter
repo](https://github.com/jonathanhogg/flitter), which also hosts the
[issues tracker](https://github.com/jonathanhogg/flitter/issues). The official
release packages are available in the [PyPI `flitter-lang`
package](https://pypi.org/project/flitter-lang/).
