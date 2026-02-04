"""
Configuration Management Module
Handles all configuration settings, environment variables, and constants.
"""

import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv
import logging
import sys
import io

# FIX Windows Unicode logging
if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
CACHE_DIR = DATA_DIR / "cache"

# Create directories if they don't exist
for directory in [MODEL_DIR, INPUT_DIR, OUTPUT_DIR, CACHE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


class Config:
    """Central configuration class for the Pharma Context Engine."""
    
    # API Configuration
    ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
    OPENFDA_API_KEY = os.getenv("OPENFDA_API_KEY", "")
    
    # Model Configuration
    YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", str(MODEL_DIR / "best.pt"))
    OCR_ENGINE = os.getenv("OCR_ENGINE", "easyocr")
    OCR_LANGUAGES = os.getenv("OCR_LANGUAGES", "en").split(",")
    
    # Processing Configuration
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
    FUZZY_MATCH_THRESHOLD = int(os.getenv("FUZZY_MATCH_THRESHOLD", "85"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "4"))
    USE_GPU = os.getenv("USE_GPU", "True").lower() == "true"
    
    # Data Paths
    INPUT_DIR = Path(os.getenv("INPUT_DIR", str(INPUT_DIR)))
    OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(OUTPUT_DIR)))
    CACHE_DIR = Path(os.getenv("CACHE_DIR", str(CACHE_DIR)))
    
    # API Endpoints
    OPENFDA_BASE_URL = os.getenv("OPENFDA_BASE_URL", "https://api.fda.gov/drug/label.json")
    RXNORM_BASE_URL = os.getenv("RXNORM_BASE_URL", "https://rxnav.nlm.nih.gov/REST")
    
    # Entity Field Names
    ENTITY_FIELDS = {
        "drug_name": ["drug_name", "medicine_name", "product_name"],
        "manufacturer": ["manufacturer", "company", "mfr"],
        "composition": ["composition", "active_ingredients", "ingredients"],
        "dosage": ["dosage", "strength", "dose"],
        "barcode": ["barcode", "data_matrix", "code"]
    }
    
    # Image Preprocessing Configuration
    IMAGE_SIZE = (640, 640)
    DENOISE_STRENGTH = 10
    SHARPEN_KERNEL = [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]
    
    # Logging Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @classmethod
    def get_config_dict(cls) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return {
            key: value for key, value in cls.__dict__.items()
            if not key.startswith("_") and not callable(value)
        }


def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL),
        format=Config.LOG_FORMAT,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(BASE_DIR / "pharma_engine.log")
        ]
    )
    return logging.getLogger(__name__)


logger = setup_logging()
