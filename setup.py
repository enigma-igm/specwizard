from os import path
from setuptools import find_packages, setup

if path.exists('README.md'):
    with open('README.md') as readme:
        long_description = readme.read()


setup(
    name="SpecWizard",
    version='0.0.dev0',
    description="Code to create spectra from simulation output.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/enigma-igm/specwizard.git",
    author='AAG, CCD, and more',
    include_package_data=True,
    packages=find_packages(exclude=["tests"]),
)

