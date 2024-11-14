import os
from pathlib import Path

from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='EMG decoders',
    author='John Luoyu Zhou',
    license='MIT',
)


# Set up directories
repo_dir = os.path.dirname(os.path.realpath(__file__))

dir_names = ["data/processed", "data/interim", "data/raw", "data/external", "models"]

for dir_name in dir_names:
    dir_path = Path(f"{repo_dir}/{dir_name}")
    dir_path.mkdir(parents=True, exist_ok=True)
