#!/usr/bin/env python3
"""
OCR Utility Script using PaddleOCR
Supports multiple languages including English and Hindi
"""

import os
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path

try:
    from paddleocr import PaddleOCR
    import cv2
    import numpy as np
    from PIL import Image
except ImportError as e:
    print(f"Error: Missing required dependencies. Please install them first: {e}")
    print("Run: pip install -r requirements.txt")
    exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OCRProcessor:
    """Main OCR processor class using PaddleOCR"""
    
    def __init__(self, languages: List[str] = None, use_gpu: bool = False):
        """
        Initialize OCR processor
        
        Args:
            languages: List of language codes (e.g., ['en', 'hi'])
            use_gpu: Whether to use GPU acceleration
        """
        if languages is None:
            languages = ['en', 'hi']  # Default: English + Hindi
        
        self.languages = languages
        self.use_gpu = use_gpu
        self.ocr_models = {}
        
        # Initialize OCR models for each language
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize PaddleOCR models for specified languages"""
        try:
            for lang in self.languages:
                logger.info(f"Initializing PaddleOCR model for language: {lang}")
                self.ocr_models[lang] = PaddleOCR(
                    use_angle_cls=True,
                    lang=lang,
                    use_gpu=self.use_gpu,
                    show_log=False
                )
            logger.info("All OCR models initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OCR models: {e}")
            raise
    
    def process_image(self, image_path: str, language: str = None) -> List[Dict]:
        """
        Process a single image with OCR
        
        Args:
            image_path: Path to the image file
            language: Specific language to use (if None, uses all initialized languages)
            
        Returns:
            List of dictionaries containing text detection results
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Validate image file
        if not self._is_valid_image(image_path):
            raise ValueError(f"Invalid image file: {image_path}")
        
        # Determine which languages to use
        if language:
            if language not in self.ocr_models:
                raise ValueError(f"Language '{language}' not initialized. Available: {list(self.ocr_models.keys())}")
            languages_to_use = [language]
        else:
            languages_to_use = self.languages
        
        results = []
        
        for lang in languages_to_use:
            try:
                logger.info(f"Processing image with {lang} language model")
                ocr_result = self.ocr_models[lang].ocr(image_path, cls=True)
                
                # Process results
                processed_results = self._process_ocr_results(ocr_result, lang)
                results.extend(processed_results)
                
            except Exception as e:
                logger.error(f"Error processing image with {lang} language: {e}")
                continue
        
        return results
    
    def _process_ocr_results(self, ocr_result: List, language: str) -> List[Dict]:
        """
        Process raw OCR results into structured format
        
        Args:
            ocr_result: Raw OCR result from PaddleOCR
            language: Language code used for detection
            
        Returns:
            List of processed results
        """
        processed_results = []
        
        if not ocr_result or not ocr_result[0]:
            return processed_results
        
        for line in ocr_result[0]:
            if len(line) >= 2:
                # Extract bounding box coordinates
                bbox = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                
                # Extract text and confidence
                text_info = line[1]
                if isinstance(text_info, tuple) and len(text_info) >= 2:
                    text = text_info[0]
                    confidence = text_info[1]
                else:
                    text = str(text_info)
                    confidence = 1.0
                
                # Calculate bounding box center and dimensions
                bbox_array = np.array(bbox)
                center_x = np.mean(bbox_array[:, 0])
                center_y = np.mean(bbox_array[:, 1])
                width = np.max(bbox_array[:, 0]) - np.min(bbox_array[:, 0])
                height = np.max(bbox_array[:, 1]) - np.min(bbox_array[:, 1])
                
                processed_results.append({
                    'text': text,
                    'confidence': float(confidence),
                    'language': language,
                    'bbox': bbox,
                    'center': (float(center_x), float(center_y)),
                    'dimensions': (float(width), float(height))
                })
        
        return processed_results
    
    def _is_valid_image(self, image_path: str) -> bool:
        """Check if the file is a valid image"""
        try:
            with Image.open(image_path) as img:
                img.verify()
            return True
        except Exception:
            return False
    
    def extract_text_only(self, image_path: str, language: str = None) -> List[str]:
        """
        Extract only the text content from OCR results
        
        Args:
            image_path: Path to the image file
            language: Specific language to use
            
        Returns:
            List of detected text strings
        """
        results = self.process_image(image_path, language)
        return [result['text'] for result in results if result['text'].strip()]
    
    def get_detailed_results(self, image_path: str, language: str = None) -> Dict:
        """
        Get detailed OCR results with statistics
        
        Args:
            image_path: Path to the image file
            language: Specific language to use
            
        Returns:
            Dictionary containing detailed results and statistics
        """
        results = self.process_image(image_path, language)
        
        if not results:
            return {
                'text_count': 0,
                'languages_detected': [],
                'average_confidence': 0.0,
                'results': [],
                'summary': 'No text detected'
            }
        
        # Calculate statistics
        text_count = len(results)
        languages_detected = list(set(result['language'] for result in results))
        confidences = [result['confidence'] for result in results]
        average_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Group by language
        results_by_language = {}
        for result in results:
            lang = result['language']
            if lang not in results_by_language:
                results_by_language[lang] = []
            results_by_language[lang].append(result)
        
        return {
            'text_count': text_count,
            'languages_detected': languages_detected,
            'average_confidence': round(average_confidence, 3),
            'results': results,
            'results_by_language': results_by_language,
            'summary': f"Detected {text_count} text elements in {len(languages_detected)} languages with {average_confidence:.1%} average confidence"
        }


def create_sample_image(output_path: str = "sample.jpg"):
    """
    Create a sample image with English and Hindi text for testing
    
    Args:
        output_path: Path where to save the sample image
    """
    try:
        # Create a white background image
        img = np.ones((400, 600, 3), dtype=np.uint8) * 255
        
        # Add English text
        cv2.putText(img, "Hello World!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        cv2.putText(img, "This is a sample image", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(img, "for testing OCR functionality", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Add Hindi text (नमस्ते दुनिया = Hello World in Hindi)
        cv2.putText(img, "नमस्ते दुनिया", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
        cv2.putText(img, "यह एक नमूना छवि है", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Save the image
        cv2.imwrite(output_path, img)
        logger.info(f"Sample image created: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create sample image: {e}")
        return False


if __name__ == "__main__":
    # Test the OCR functionality
    print("Testing OCR functionality...")
    
    # Create sample image if it doesn't exist
    if not os.path.exists("sample.jpg"):
        create_sample_image()
    
    # Initialize OCR processor
    try:
        ocr = OCRProcessor(languages=['en', 'hi'])
        
        # Test with sample image
        if os.path.exists("sample.jpg"):
            print("\n=== Testing OCR with sample image ===")
            
            # Extract text only
            texts = ocr.extract_text_only("sample.jpg")
            print(f"Detected texts: {texts}")
            
            # Get detailed results
            detailed = ocr.get_detailed_results("sample.jpg")
            print(f"\nDetailed results: {detailed['summary']}")
            print(f"Languages detected: {detailed['languages_detected']}")
            print(f"Average confidence: {detailed['average_confidence']}")
            
        else:
            print("Sample image not found. Please create one manually or check the create_sample_image function.")
            
    except Exception as e:
        print(f"Error during testing: {e}")
        print("Please ensure all dependencies are installed: pip install -r requirements.txt") 