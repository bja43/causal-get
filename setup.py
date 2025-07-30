from os.path import join
from setuptools import setup, Extension


ext_modules = [
  Extension(
    name="causalget.c_backend",
    sources=[join("causalget", "c_backend.c")],
    extra_compile_args=["-Wall", "-O3"],
  )
]

setup(
  name="causal-get",
  version="0.0.1",
  description="Causal Graph Estimation Toolbox",
  packages=["causalget"],
  ext_modules=ext_modules,
)
