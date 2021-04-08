from setuptools import setup
from Cython.Build import cythonize
from numpy import get_include
setup(
	ext_modules = cythonize("c_corr2d.pyx", compiler_directives={'language_level':3,'cdivision':True}),
	include_dirs=[get_include()]
)
