
from setuptools import setup, Extension

setup(
    ext_modules=[
        Extension("flitter.model", ["src/flitter/model.pyx"]),
        Extension("flitter.language.functions", ["src/flitter/language/functions.pyx"]),
        Extension("flitter.language.tree", ["src/flitter/language/tree.pyx"]),
        Extension("flitter.render.canvas", ["src/flitter/render/canvas.pyx"]),
        Extension("flitter.render.canvas3d", ["src/flitter/render/canvas3d.pyx"]),
        Extension("flitter.render.dmx", ["src/flitter/render/dmx.pyx"]),
        Extension("flitter.render.laser", ["src/flitter/render/laser.pyx"]),
    ]
)
