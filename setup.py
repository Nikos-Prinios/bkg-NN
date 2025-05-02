# setup.py
from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name='bkg_speedups',
    ext_modules=cythonize(
        "speedups.pyx",
        compiler_directives={'language_level': '3'}
    ),
    include_dirs=[np.get_include()],
    zip_safe=False,
)