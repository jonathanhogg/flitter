
import os
from pathlib import Path
from setuptools import setup, Extension

from Cython.Build import cythonize


def find_cython_files(dirpath):
    for filepath in dirpath.iterdir():
        if filepath.is_dir():
            yield from find_cython_files(filepath)
        elif filepath.suffix == '.pyx':
            yield filepath


if int(os.environ.get('FLITTER_BUILD_COVERAGE', '0')):
    print("Building for coverage testing")
    define_macros = [("CYTHON_TRACE_NOGIL", "1")]
    compiler_directives = {'linetrace': True}
else:
    define_macros = []
    compiler_directives = {}

ext_modules = []
for filepath in find_cython_files(Path('src')):
    module_name = '.'.join(filepath.with_suffix('').parts[1:])
    ext_modules.append(Extension(module_name, [str(filepath)], define_macros=define_macros))

setup(ext_modules=cythonize(ext_modules, language_level=3, compiler_directives=compiler_directives))
