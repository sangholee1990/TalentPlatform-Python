from setuptools import setup

setup(
    name="nmsc-climate-toolbox",
    version="0.1.0",
    description="NMSC Climate Toolbox for processing and visualizing climate data",
    author="TalentPlatform",
    py_modules=["nmsc_climate_toolbox", "app_qt_material"],
    install_requires=[
        "numpy",
        "scipy",
        "xarray",
                "rioxarray",
        "matplotlib",
                "PyQt6",
        "qfluentwidgets",
        "PyQt6-WebEngine",
    ],
    entry_points={
        'console_scripts': [
            'nmsc-climate-toolbox=app_qt_material:main',
        ],
    },
)
