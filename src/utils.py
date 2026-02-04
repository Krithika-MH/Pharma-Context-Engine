"""
Utility Functions
Helper functions for file I/O, metrics calculation, and visualization.
"""

import json
import cv2
import numpy as np
from typing import Dict, Any, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_image(image_path: str) -> Optional[np.ndarray]:
    """Load image from file path."""
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return None
        return image
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {str(e)}")
        return None


def save_json(data: Dict[str, Any], output_path: str, indent: int = 2):
    """Save dictionary as JSON file."""
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
        
        logger.debug(f"Saved JSON to: {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to save JSON: {str(e)}")


def load_json(file_path: str) -> Optional[Dict]:
    """Load JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Failed to load JSON from {file_path}: {str(e)}")
        return None


def calculate_metrics(predictions: Dict, ground_truth: Dict) -> Dict:
    """Calculate evaluation metrics."""
    metrics = {
        "accuracy": 0,
        "precision": 0,
        "recall": 0,
        "f1_score": 0
    }
    
    try:
        # Compare extracted vs ground truth
        fields = ["drug_name", "manufacturer", "dosage"]
        
        correct = 0
        total = len(fields)
        
        for field in fields:
            pred = predictions.get(field, "").lower()
            gt = ground_truth.get(field, "").lower()
            
            if pred == gt:
                correct += 1
        
        metrics["accuracy"] = (correct / total * 100) if total > 0 else 0
        
    except Exception as e:
        logger.error(f"Metrics calculation failed: {str(e)}")
    
    return metrics


def format_output_json(data: Dict) -> str:
    """Format output JSON for pretty printing."""
    return json.dumps(data, indent=2, ensure_ascii=False, default=str)


def create_visualization(image: np.ndarray, 
                        detections: list,
                        text_results: list,
                        output_path: Optional[str] = None) -> np.ndarray:
    """Create visualization of detections and text extraction."""
    vis_image = image.copy()
    
    # Draw detections
    for det in detections:
        bbox = det.get("bbox", [])
        if len(bbox) == 4:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            label = f"{det.get('class_name', 'object')}: {det.get('confidence', 0):.2f}"
            cv2.putText(vis_image, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Draw text regions
    for text_res in text_results:
        bbox = text_res.get("bbox", [])
        if bbox:
            points = np.array(bbox, dtype=np.int32)
            cv2.polylines(vis_image, [points], True, (255, 0, 0), 1)
    
    if output_path:
        cv2.imwrite(output_path, vis_image)
    
    return vis_image
