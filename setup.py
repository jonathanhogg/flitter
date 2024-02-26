
from setuptools import setup, Extension

setup(
    ext_modules=[
        Extension("flitter.language.functions", ["src/flitter/language/functions.pyx"]),
        Extension("flitter.language.noise", ["src/flitter/language/noise.pyx"]),
        Extension("flitter.language.tree", ["src/flitter/language/tree.pyx"]),
        Extension("flitter.language.vm", ["src/flitter/language/vm.pyx"]),
        Extension("flitter.model", ["src/flitter/model.pyx"]),
        Extension("flitter.render.counter", ["src/flitter/render/counter.pyx"]),
        Extension("flitter.render.dmx", ["src/flitter/render/dmx.pyx"]),
        Extension("flitter.render.laser", ["src/flitter/render/laser.pyx"]),
        Extension("flitter.render.physics", ["src/flitter/render/physics.pyx"]),
        Extension("flitter.render.window.canvas", ["src/flitter/render/window/canvas.pyx"]),
        Extension("flitter.render.window.canvas3d", ["src/flitter/render/window/canvas3d.pyx"]),
        Extension("flitter.render.window.models", ["src/flitter/render/window/models.pyx"]),
    ]
)
