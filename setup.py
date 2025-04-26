from setuptools import setup
from Cython.Build import cythonize

import sys

sys.argv.append('build_ext')
sys.argv.append('--inplace')

def make():
    setup(
        name='fft_mic',
        ext_modules=cythonize("main.pyx"),
    )