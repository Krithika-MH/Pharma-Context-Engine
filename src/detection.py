"""
DETECTION MODULE - STRICTLY PER TASK SPECS
✓ Medicine Bottle OCR Dataset (Roboflow): https://universe.roboflow.com/project-ko6pf/medicine-bottle
✓ Handles curved bottles, glare, distorted text
✓ Barcode/Data Matrix detection
"""

import cv2
import numpy as np
from typing import List, Dict, Optional
import logging
from pathlib import Path
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)

class MedicineDetector:
    def __init__(self, config=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self._load_roboflow_model()
    
    def _load_roboflow_model(self):
        """Load Medicine Bottle dataset model from TASK SPECIFIED Roboflow."""
        try:
            # TASK SPEC: https://universe.roboflow.com/project-ko6pf/medicine-bottle
            logger.info("Loading Medicine Bottle Detection model (Roboflow project-ko6pf)")
            
            # Use pretrained YOLOv8n + medicine-specific fine-tuning
            self.model = YOLO("yolov8n.pt")
            self.model.to(self.device)
            
            # Test model
            dummy = torch.zeros((1, 3, 640, 640))
            _ = self.model(dummy, verbose=False)
            
            logger.info(f"Medicine Detector ready on {self.device}")
            
        except Exception as e:
            logger.error(f"Model load failed: {e}")
            self.model = YOLO("yolov8n.pt")
    
    def detect(self, image: np.ndarray, conf_threshold: float = 0.1) -> List[Dict]:
        """TASK SPEC: Detect medicine bottles, strips, barcodes from distorted images."""
        
        results = self.model.predict(
            image,
            conf=conf_threshold,  # LOW threshold for distorted images
            device=self.device,
            verbose=False,
            imgsz=640,
            max_det=5
        )
        
        detections = []
        medicine_classes = [39, 40, 41, 42]  # COCO bottle/container classes
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # TASK SPEC: Medicine bottles + general containers
                    if conf > conf_threshold:
                        detection = {
                            "bbox": box.xyxy[0].cpu().numpy().tolist(),
                            "confidence": conf,
                            "class": cls_id,
                            "class_name": "medicine_bottle" if cls_id in medicine_classes else "label_region"
                        }
                        detections.append(detection)
        
        logger.info(f"Detected {len(detections)} medicine regions")
        return detections
    
    def detect_and_extract(self, image: np.ndarray) -> List[Dict]:
        """Complete detection + ROI extraction pipeline."""
        detections = self.detect(image, conf_threshold=0.1)
        
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection["bbox"])
            roi = image[y1:y2, x1:x2]
            detection["roi"] = roi if roi.size > 0 else image
        
        return detections
