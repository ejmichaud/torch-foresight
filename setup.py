# -*- coding: UTF-8 -*-

from setuptools import setup, find_packages

setup(
    name='torch-foresight',
    version='0.1.2',
    packages=find_packages(include=['foresight']),
    author='Eric J. Michaud',
    author_email='ericjmichaud@berkeley.edu',
    license='MIT',
    url='https://github.com/ejmichaud/torch-foresight',
    description='Tools for characterizing and predicting the dynamics of neural nets built with PyTorch',
    setup_requires=['pytest-runner'],
    tests_require=['pytest']
)
