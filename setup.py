from setuptools import setup, Extension

setup(
    ext_modules=[
        # Extension("mymodule", sources=["mymodule.c"])
        Extension("causalget", sources=["causalget.c"])
    ]
)
