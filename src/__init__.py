"""
Intelligent Pharma-Context Engine
Main package initialization
"""

__version__ = "1.0.0"
__author__ = "Pharma Context Team"

from .config import Config
from .pipeline import PharmaContextPipeline
from .preprocessing import ImagePreprocessor
from .detection import MedicineDetector
from .extraction import TextExtractor
from .verification import DataVerifier
from .enrichment import DataEnricher

__all__ = [
    "Config",
    "PharmaContextPipeline",
    "ImagePreprocessor",
    "MedicineDetector",
    "TextExtractor",
    "DataVerifier",
    "DataEnricher"
]
