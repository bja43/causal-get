from setuptools import setup, Extension

setup(
    ext_modules=[
        Extension("causalget", sources=["causalget.c"])
    ]
)
