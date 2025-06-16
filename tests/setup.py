from os.path import join
from setuptools import setup, Extension


ext_modules = [
  Extension(
    name="causalget.c_backend",
    sources=[join("causalget", "c_backend.c")],
    include_dirs=["causalget"],
    extra_compile_args=["-Wall", "-Wextra", "-pedantic", "-O3"],
  )
]

setup(
  name="causal-get",
  version="0.1",
  description="Causal Graph Estimation Toolbox",
  packages=["causalget"],
  ext_modules=ext_modules,
)
