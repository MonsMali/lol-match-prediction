"""
Setup script for the League of Legends Match Prediction System.

Install in development mode with:
    pip install -e .
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README for the long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    requirements = [
        line.strip()
        for line in requirements_path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
else:
    requirements = []

setup(
    name="lol-match-predictor",
    version="1.0.0",
    author="Luis Conceicao",
    author_email="luis.viegas.conceicao@gmail.com",
    description="Machine learning system for predicting professional League of Legends match outcomes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/luisconceicao/lol-match-predictor",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Games/Entertainment",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
    },
    entry_points={
        "console_scripts": [
            "lol-predict=src.prediction.predictor:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.joblib", "*.csv"],
    },
)
