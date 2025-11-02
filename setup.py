# ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD

from distutils.core import setup

setup(
    name='oct_levitation',
    version='1.0.0',
    packages=['oct_levitation'],
    package_dir={'': 'src'},
    install_requires = [
        "alive_progress",
        "control",
        "matplotlib",
        "mayavi==4.7.2",
        "numba==0.51.0",
        "numpy==1.23.0",
        "pandas",
        "scipy==1.10.1",
        "vtk==9.0.1",
    ]
)