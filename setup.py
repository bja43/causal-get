from setuptools import setup, Extension


# Define the extension
module = Extension('causalget',
                   sources=['causalget.c'],
                   extra_compile_args=['-Wall', '-Wextra', '-pedantic', '-O3'])

# Setup script
setup(
    name='causal-get',
    version='0.1.0',
    description='Causal Graph Estimation Toolbox',
    ext_modules=[module],
)
