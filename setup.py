"""
Setup script for Pharma Context Engine
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    long_description = readme_file.read_text(encoding="utf-8")

setup(
    name="pharma-context-engine",
    version="1.0.0",
    description="Intelligent pharmaceutical label extraction and verification system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/pharma-context-engine",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-python",
        "pillow",
        "torch",
        "torchvision",
        "ultralytics",
        "easyocr",
        "pytesseract",
        "pyzbar",
        "python-dotenv",
        "requests",
        "fuzzywuzzy",
        "python-Levenshtein",
        "scikit-image",
        "scipy",
        "pandas",
        "tqdm",
        "roboflow",
        "datasets",
        "transformers",
        "pydantic",
        "pytest"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
