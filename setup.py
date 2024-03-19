from setuptools import setup, find_packages

setup(
    name='LinAlgLib',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'sympy',
        'scipy'
    ],
)
