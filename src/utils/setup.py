import os
from setuptools import setup, find_packages

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="Medical Image Segmentation",
    version="0.0.1",
    author="Pierre-Marc Jodoin & Hassan Kajila",
    author_email="",
    description="A Pytorch implementation of a advanced CNN model for image classification",
    license="BSD",
    keywords="Pytorch deep learning cnn",
    url="",
    packages=find_packages(exclude=['contrib', 'doc', 'unit_tests']),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
)
