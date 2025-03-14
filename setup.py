from setuptools import find_packages, setup

setup(
    name="ssma",
    version="0.1.0",
    author="SVECTOR",
    author_email="research@svector.co.in",
    description="SSMA: Structured State Matrix Architecture library for efficient transformer-like models.",
    url="https://github.com/SVECTOR-CORPORATION/SSMA/",  # Update as needed
    packages=find_packages(),
    install_requires=[
        "torch>=1.8",
        "numpy",
        "tqdm",
        "pyyaml"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
