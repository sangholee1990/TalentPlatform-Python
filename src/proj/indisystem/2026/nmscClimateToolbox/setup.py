from setuptools import setup
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="nmsc-climate-toolbox",
    version="0.1.1",
    description="NMSC Climate Toolbox for processing climate data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="nmsc",
    py_modules=["nmsc_climate_toolbox"],
    install_requires=[
        "numpy",
        "scipy",
        "xarray",
        "rioxarray",
        "matplotlib",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.11',
)