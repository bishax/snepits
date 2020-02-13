from distutils.extension import Extension

import numpy
# from distutils.core import setup
from Cython.Build import cythonize
from setuptools import find_packages, setup

CYgil_AR_gen = Extension(
    "CYgil_AR_gen",
    ["snepits/models/simulation/CYgil_AR_gen.pyx"],
    # define_macros=[("CYTHON_TRACE", "0")],
    include_dirs=[numpy.get_include(), "/usr/include"],
    library_dirs=["/usr/lib"],
    libraries=["gsl", "gslcblas"],
)

models_spec = Extension(
    "_models_spec",
    ["snepits/models/_models_spec.pyx"],
    include_dirs=[numpy.get_include(), "/usr/include"],
    library_dirs=["/usr/lib"],
    libraries=["gsl", "gslcblas"],
)

pyovpyx = Extension(
    "pyovpyx",
    ["snepits/models/pyovpyx.pyx"],
    include_dirs=[numpy.get_include(), "/usr/include"],
    library_dirs=["/usr/lib"],
    libraries=["gsl", "gslcblas"],
)
setup(
    name="snepits",
    packages=find_packages(),
    package_dir={'snepits': 'snepits/', '_models_spec': 'snepits/'},
    version="0.1.0",
    description="Inference for stochastic epidemic household models",
    author="Alex Bishop",
    license="MIT",
    ext_modules=cythonize([models_spec, CYgil_AR_gen, pyovpyx]),
    include_dirs=[numpy.get_include()],
)
