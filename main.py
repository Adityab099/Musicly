#!/usr/bin/env python3
"""
Main script demonstrating OCR utility usage
Provides command-line interface and examples
"""

import os
import sys
import argparse
from pathlib import Path

# Import our OCR utility
from ocr_utils import OCRProcessor, create_sample_image


def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(
        description="OCR Utility using PaddleOCR - Extract text from images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py sample.jpg                    # Process sample.jpg with all languages
  python main.py image.png --lang en          # Process with English only
  python main.py image.jpg --detailed         # Get detailed results
  python main.py --create-sample              # Create sample image for testing
        """
    )
    
    parser.add_argument(
        'image_path',
        nargs='?',
        help='Path to the image file to process'
    )
    
    parser.add_argument(
        '--lang', '--language',
        choices=['en', 'hi', 'ch', 'ko', 'ja', 'ar', 'de', 'fr', 'es', 'ru'],
        help='Specific language to use for OCR (default: all initialized languages)'
    )
    
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Show detailed OCR results with bounding boxes and confidence scores'
    )
    
    parser.add_argument(
        '--create-sample',
        action='store_true',
        help='Create a sample image with English and Hindi text for testing'
    )
    
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Use GPU acceleration if available'
    )
    
    parser.add_argument(
        '--languages',
        nargs='+',
        default=['en', 'hi'],
        help='Languages to initialize (default: en hi)'
    )
    
    args = parser.parse_args()
    
    # Handle create sample option
    if args.create_sample:
        print("Creating sample image...")
        if create_sample_image():
            print("✓ Sample image 'sample.jpg' created successfully!")
            print("You can now test OCR with: python main.py sample.jpg")
        else:
            print("✗ Failed to create sample image")
        return
    
    # Check if image path is provided
    if not args.image_path:
        parser.print_help()
        print("\nNo image path provided. Use --create-sample to create a test image.")
        return
    
    # Check if image file exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found: {args.image_path}")
        print("Use --create-sample to create a test image, or provide a valid image path.")
        return
    
    # Initialize OCR processor
    try:
        print(f"Initializing OCR processor with languages: {args.languages}")
        ocr = OCRProcessor(languages=args.languages, use_gpu=args.gpu)
        print("✓ OCR processor initialized successfully!")
        
    except Exception as e:
        print(f"✗ Failed to initialize OCR processor: {e}")
        print("Please ensure all dependencies are installed: pip install -r requirements.txt")
        return
    
    # Process the image
    try:
        print(f"\nProcessing image: {args.image_path}")
        
        if args.detailed:
            # Get detailed results
            results = ocr.get_detailed_results(args.image_path, args.lang)
            print_results_detailed(results)
        else:
            # Get text only
            texts = ocr.extract_text_only(args.image_path, args.lang)
            print_results_simple(texts)
            
    except Exception as e:
        print(f"✗ Error processing image: {e}")


def print_results_simple(texts):
    """Print simple text results"""
    print(f"\n=== OCR Results ===")
    if texts:
        print(f"Detected {len(texts)} text elements:")
        for i, text in enumerate(texts, 1):
            print(f"  {i}. {text}")
    else:
        print("No text detected in the image.")


def print_results_detailed(results):
    """Print detailed OCR results"""
    print(f"\n=== Detailed OCR Results ===")
    print(f"Summary: {results['summary']}")
    print(f"Languages detected: {', '.join(results['languages_detected'])}")
    print(f"Average confidence: {results['average_confidence']:.1%}")
    
    if results['results']:
        print(f"\nDetailed results:")
        for i, result in enumerate(results['results'], 1):
            print(f"  {i}. Text: '{result['text']}'")
            print(f"     Language: {result['language']}")
            print(f"     Confidence: {result['confidence']:.1%}")
            print(f"     Position: Center at ({result['center'][0]:.1f}, {result['center'][1]:.1f})")
            print(f"     Size: {result['dimensions'][0]:.1f} x {result['dimensions'][1]:.1f}")
            print()
    else:
        print("No text detected in the image.")


def run_examples():
    """Run example usage scenarios"""
    print("=== OCR Utility Examples ===\n")
    
    # Example 1: Basic usage
    print("1. Basic OCR Usage:")
    print("   from ocr_utils import OCRProcessor")
    print("   ocr = OCRProcessor(languages=['en', 'hi'])")
    print("   texts = ocr.extract_text_only('image.jpg')")
    print("   print(texts)")
    print()
    
    # Example 2: Language-specific processing
    print("2. Language-Specific Processing:")
    print("   # Process with English only")
    print("   texts_en = ocr.extract_text_only('image.jpg', language='en')")
    print("   # Process with Hindi only")
    print("   texts_hi = ocr.extract_text_only('image.jpg', language='hi')")
    print()
    
    # Example 3: Detailed results
    print("3. Detailed Results:")
    print("   detailed = ocr.get_detailed_results('image.jpg')")
    print("   print(f\"Detected {detailed['text_count']} text elements\")")
    print("   print(f\"Average confidence: {detailed['average_confidence']}\")")
    print()
    
    # Example 4: Batch processing (future extension)
    print("4. Batch Processing (Future Extension):")
    print("   # This could be extended to process multiple images")
    print("   image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']")
    print("   for img_path in image_paths:")
    print("       texts = ocr.extract_text_only(img_path)")
    print("       print(f\"{img_path}: {texts}\")")
    print()


if __name__ == "__main__":
    # Check if running with --examples flag
    if len(sys.argv) > 1 and sys.argv[1] == '--examples':
        run_examples()
    else:
        main() 