import io
import os
import re

from setuptools import find_packages, setup

# Adapted from https://stackoverflow.com/a/39671214
this_directory = os.path.dirname(os.path.realpath(__file__))
version_matches = re.search(r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
                            io.open(f"{this_directory}/frag_gt/__init__.py", encoding="utf_8_sig").read())
if version_matches is None:
    raise Exception("Could not determine FragGT version from __init__.py")
__version__ = version_matches.group(1)

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="frag_gt",
    version=__version__,
    author="Joshua Meyers",
    author_email="joshua.meyers@benevolent.ai",
    description="FragGT is an evolutionary algorithm for molecule generation distributed with guacamol_baselines",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BenevolentAI/guacamol_baselines",
    packages=find_packages(include=["frag_gt", "frag_gt.*"]),
    install_requires=[
        "numpy>=1.15.2",
        "tqdm>=4.26.0",
        "rdkit-pypi>=2021.9.3",
        "pandas",
        "joblib>=0.12.5",
        "pytest>=3.8.2",
    ],
    python_requires=">=3.6",
    extras_require={
        "rdkit": ["rdkit>=2020.03.3"],
    },
    include_package_data=True,
    zip_safe=False,
    classifiers=(
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ),
)
