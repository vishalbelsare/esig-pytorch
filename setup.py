from distutils.core import setup
from Cython.Build import cythonize

setup(name="cython_sig_distance", ext_modules=cythonize('cython_sig_distance.pyx'),)
