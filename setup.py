#!/usr/bin/env python

import os
import shutil
import sys

from setuptools import setup
from setuptools import find_packages

requirements = [
    #'numpy',
    #'six',
    #'torch',
    #'torchvision',
    #'horovod',
]

exec(open('deeptensor/_version_.py').read())

setup(
    name='deeptensor',
    version=__version__,
    description='Concise DL framework for both research and production',
    author='cicicici',
    author_email='cicicici@gmail.com',
    url='https://github.com/cicicici/deeptensor',
    download_url='https://github.com/cicicici/deeptensor/tarball/' + __version__,
    license='MIT',
    keywords=['deeptensor'],

    packages=find_packages(exclude=('test',)),

    zip_safe=True,
    install_requires=requirements,
    python_requires='>=3.6',
)
