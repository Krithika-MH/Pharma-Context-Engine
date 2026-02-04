"""
Image Preprocessing Module
Handles image enhancement, noise reduction, perspective correction,
and preparation for OCR and detection models.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import logging
from pathlib import Path
from skimage import exposure
from scipy import ndimage
from typing import Dict, List, Tuple, Optional


from .config import Config

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Advanced image preprocessing for medicine label extraction."""
    
    def __init__(self, config: Config = Config):
        self.config = config
        
    def load_image(self, image_path: str) -> Optional[np.ndarray]:
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
    
    def remove_glare(self, image: np.ndarray) -> np.ndarray:
        """
        Remove specular reflections and glare from medicine labels.
        Uses morphological operations and inpainting.
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect bright regions (glare)
            _, glare_mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
            
            # Dilate mask to cover glare boundaries
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            glare_mask = cv2.dilate(glare_mask, kernel, iterations=2)
            
            # Inpaint glare regions
            result = cv2.inpaint(image, glare_mask, 3, cv2.INPAINT_TELEA)
            
            logger.debug("Glare removal completed")
            return result
            
        except Exception as e:
            logger.warning(f"Glare removal failed: {str(e)}")
            return image
    
    def correct_perspective(self, image: np.ndarray) -> np.ndarray:
        """
        Correct perspective distortion caused by curved bottles.
        Detects edges and applies perspective transformation.
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Detect lines using Hough transform
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, 
                                   minLineLength=100, maxLineGap=10)
            
            if lines is None or len(lines) < 4:
                logger.debug("Insufficient lines for perspective correction")
                return image
            
            # Find corners from detected lines
            corners = self._find_corners_from_lines(lines, image.shape)
            
            if corners is not None:
                # Apply perspective transform
                width, height = image.shape[1], image.shape[0]
                dst_points = np.float32([[0, 0], [width, 0], 
                                        [width, height], [0, height]])
                matrix = cv2.getPerspectiveTransform(corners, dst_points)
                result = cv2.warpPerspective(image, matrix, (width, height))
                logger.debug("Perspective correction applied")
                return result
            
            return image
            
        except Exception as e:
            logger.warning(f"Perspective correction failed: {str(e)}")
            return image
    
    def _find_corners_from_lines(self, lines: np.ndarray, 
                                 shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """Extract corner points from detected lines."""
        try:
            points = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                points.extend([(x1, y1), (x2, y2)])
            
            points = np.array(points)
            
            # Find extreme points (corners)
            top_left = points[np.argmin(points[:, 0] + points[:, 1])]
            top_right = points[np.argmax(points[:, 0] - points[:, 1])]
            bottom_right = points[np.argmax(points[:, 0] + points[:, 1])]
            bottom_left = points[np.argmin(points[:, 0] - points[:, 1])]
            
            corners = np.float32([top_left, top_right, bottom_right, bottom_left])
            return corners
            
        except Exception as e:
            logger.warning(f"Corner detection failed: {str(e)}")
            return None
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive histogram equalization for better text visibility."""
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge channels
            enhanced = cv2.merge([l, a, b])
            result = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            logger.debug("Contrast enhancement completed")
            return result
            
        except Exception as e:
            logger.warning(f"Contrast enhancement failed: {str(e)}")
            return image
    
    def denoise(self, image: np.ndarray) -> np.ndarray:
        """Remove noise while preserving text edges."""
        try:
            result = cv2.fastNlMeansDenoisingColored(
                image, None, 
                self.config.DENOISE_STRENGTH, 
                self.config.DENOISE_STRENGTH, 7, 21
            )
            logger.debug("Denoising completed")
            return result
            
        except Exception as e:
            logger.warning(f"Denoising failed: {str(e)}")
            return image
    
    def sharpen(self, image: np.ndarray) -> np.ndarray:
        """Sharpen text for better OCR accuracy."""
        try:
            kernel = np.array(self.config.SHARPEN_KERNEL)
            result = cv2.filter2D(image, -1, kernel)
            logger.debug("Sharpening completed")
            return result
            
        except Exception as e:
            logger.warning(f"Sharpening failed: {str(e)}")
            return image
    
    def binarize(self, image: np.ndarray) -> np.ndarray:
        """Convert to binary image for OCR."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            logger.debug("Binarization completed")
            return binary
            
        except Exception as e:
            logger.warning(f"Binarization failed: {str(e)}")
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    def preprocess_pipeline(self, image_path: str, 
                           save_intermediate: bool = False) -> Dict[str, np.ndarray]:
        """
        Complete preprocessing pipeline.
        Returns dictionary with original and processed images.
        """
        logger.info(f"Starting preprocessing for: {image_path}")
        
        # Load image
        image = self.load_image(image_path)
        if image is None:
            return {}
        
        results = {"original": image.copy()}
        
        # Step 1: Remove glare
        image = self.remove_glare(image)
        results["deglared"] = image.copy()
        
        # Step 2: Correct perspective
        image = self.correct_perspective(image)
        results["perspective_corrected"] = image.copy()
        
        # Step 3: Enhance contrast
        image = self.enhance_contrast(image)
        results["contrast_enhanced"] = image.copy()
        
        # Step 4: Denoise
        image = self.denoise(image)
        results["denoised"] = image.copy()
        
        # Step 5: Sharpen
        image = self.sharpen(image)
        results["sharpened"] = image.copy()
        
        # Step 6: Binarize for OCR
        binary = self.binarize(image)
        results["binary"] = binary
        results["final"] = image
        
        # Save intermediate results if requested
        if save_intermediate:
            self._save_intermediate_results(image_path, results)
        
        logger.info(f"Preprocessing completed for: {image_path}")
        return results
    
    def _save_intermediate_results(self, original_path: str, 
                                   results: Dict[str, np.ndarray]):
        """Save intermediate preprocessing steps for debugging."""
        try:
            output_dir = self.config.CACHE_DIR / "preprocessing"
            output_dir.mkdir(exist_ok=True)
            
            base_name = Path(original_path).stem
            
            for step_name, image in results.items():
                output_path = output_dir / f"{base_name}_{step_name}.jpg"
                cv2.imwrite(str(output_path), image)
            
            logger.debug(f"Saved intermediate results to {output_dir}")
            
        except Exception as e:
            logger.warning(f"Failed to save intermediate results: {str(e)}")
