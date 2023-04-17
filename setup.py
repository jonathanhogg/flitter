
from setuptools import setup, Extension

setup(
    ext_modules=[
        Extension("flitter.model", ["flitter/model.pyx"]),
        Extension("flitter.langage.functions", ["flitter/language/functions.pyx"]),
        Extension("flitter.langage.tree", ["flitter/language/tree.pyx"]),
        Extension("flitter.render.canvas", ["flitter/render/canvas.pyx"]),
        Extension("flitter.render.canvas3d", ["flitter/render/canvas3d.pyx"]),
        Extension("flitter.render.dmx", ["flitter/render/dmx.pyx"]),
        Extension("flitter.render.laser", ["flitter/render/laser.pyx"]),
    ]
)
