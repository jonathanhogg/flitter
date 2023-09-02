
from setuptools import setup, Extension

setup(
    ext_modules=[
        Extension("flitter.model", ["src/flitter/model.pyx"]),
        Extension("flitter.interact.physics", ["src/flitter/interact/physics.pyx"]),
        Extension("flitter.language.functions", ["src/flitter/language/functions.pyx"]),
        Extension("flitter.language.noise", ["src/flitter/language/noise.pyx"]),
        Extension("flitter.language.tree", ["src/flitter/language/tree.pyx"]),
        Extension("flitter.language.vm", ["src/flitter/language/vm.pyx"]),
        Extension("flitter.render.window.canvas", ["src/flitter/render/window/canvas.pyx"]),
        Extension("flitter.render.window.canvas3d", ["src/flitter/render/window/canvas3d.pyx"]),
        Extension("flitter.render.window.models", ["src/flitter/render/window/models.pyx"]),
        Extension("flitter.render.dmx", ["src/flitter/render/dmx.pyx"]),
        Extension("flitter.render.laser", ["src/flitter/render/laser.pyx"]),
    ]
)
