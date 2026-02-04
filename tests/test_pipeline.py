"""
Basic tests for the pipeline
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.pipeline import PharmaContextPipeline


def test_pipeline_initialization():
    """Test that pipeline initializes without errors"""
    try:
        pipeline = PharmaContextPipeline(Config)
        assert pipeline is not None
        print("✓ Pipeline initialization test passed")
    except Exception as e:
        pytest.fail(f"Pipeline initialization failed: {str(e)}")


def test_config_loading():
    """Test configuration loading"""
    assert Config.CONFIDENCE_THRESHOLD > 0
    assert Config.FUZZY_MATCH_THRESHOLD > 0
    print("✓ Config loading test passed")


if __name__ == "__main__":
    test_config_loading()
    test_pipeline_initialization()
    print("\n✓ All tests passed!")
