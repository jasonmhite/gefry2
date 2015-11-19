from setuptools import setup

setup(
    name="gefry",
    version="0.0.2",
    author="Jason M. Hite",
    license="BSD",
    packages=["gefry2"],
    install_requires=['shapely'],
    package_data={'gefry2': ['data/*.h5']}
)
