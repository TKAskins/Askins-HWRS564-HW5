from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np  # If NumPy is needed

extensions = [
    Extension(
        name="_splitter_cython",                    # Name of the extension
        sources=[
            "_splitter_cython.pyx",
            ],           # Source files
        language="c++",                    # Specify C++ compilation
    ),
    Extension(
        name="_tree_cython",                    # Name of the extension
        sources=[
            "_tree_cython.pyx",
            ],           # Source files
        language="c++",                    # Specify C++ compilation
    ),    
    Extension(
        name="_criterion_cython",                    # Name of the extension
        sources=[
            "_criterion_cython.pyx",
            ],           # Source files
        language="c",                    # Specify C++ compilation
    ),
    Extension(
        name="_utils_cython",                    # Name of the extension
        sources=[
            "_utils_cython.pyx",
            ],           # Source files
        language="c",                    # Specify C++ compilation
    ),    
    # Extension(
    #     name="_random",                    # Name of the extension
    #     sources=[
    #         "_random.pyx"          # Source files
    #     ],
    #     # language="c++",
    #     extra_compile_args=["-std=c++17"],
    #     include_dirs=[np.get_include()],                    # Specify C++ compilation
    # )    
]

setup(
    ext_modules=cythonize(extensions),  # Replace with your .pyx file
    include_dirs=[np.get_include()],       # Include NumPy headers if needed
)