# OCR Utility with PaddleOCR

A powerful, multi-language OCR (Optical Character Recognition) utility built with PaddleOCR that supports English, Hindi, and many other languages.

## Features

- **Multi-language Support**: English, Hindi, Chinese, Korean, Japanese, Arabic, German, French, Spanish, Russian, and more
- **High Accuracy**: Powered by PaddleOCR's state-of-the-art models
- **Flexible Output**: Get simple text extraction or detailed results with bounding boxes and confidence scores
- **GPU Acceleration**: Optional GPU support for faster processing
- **Clean API**: Easy-to-use Python classes and functions
- **Extensible**: Designed for easy extension (batch processing, custom preprocessing, etc.)

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Install Dependencies

1. **Clone or download the project files**
2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

**Note**: PaddleOCR will automatically download the necessary model files on first use (this may take a few minutes depending on your internet connection).

### Supported Languages

The utility supports these language codes:
- `en` - English
- `hi` - Hindi
- `ch` - Chinese (Simplified)
- `ko` - Korean
- `ja` - Japanese
- `ar` - Arabic
- `de` - German
- `fr` - French
- `es` - Spanish
- `ru` - Russian

## Quick Start

### 1. Create a Sample Image

```bash
python main.py --create-sample
```

This creates `sample.jpg` with English and Hindi text for testing.

### 2. Basic OCR Usage

```bash
# Process the sample image
python main.py sample.jpg

# Process with specific language
python main.py sample.jpg --lang en

# Get detailed results
python main.py sample.jpg --detailed
```

### 3. Python Code Usage

```python
from ocr_utils import OCRProcessor

# Initialize OCR processor
ocr = OCRProcessor(languages=['en', 'hi'])

# Extract text from image
texts = ocr.extract_text_only('image.jpg')
print(texts)

# Get detailed results
detailed = ocr.get_detailed_results('image.jpg')
print(f"Detected {detailed['text_count']} text elements")
```

## Usage Examples

### Command Line Interface

```bash
# Basic usage
python main.py image.jpg

# Language-specific processing
python main.py image.jpg --lang hi

# Detailed results with bounding boxes
python main.py image.jpg --detailed

# Use GPU acceleration
python main.py image.jpg --gpu

# Initialize with specific languages
python main.py image.jpg --languages en hi ch

# Show examples
python main.py --examples
```

### Python API

```python
from ocr_utils import OCRProcessor

# Initialize with English and Hindi
ocr = OCRProcessor(languages=['en', 'hi'])

# Simple text extraction
texts = ocr.extract_text_only('document.jpg')
for text in texts:
    print(text)

# Detailed results
results = ocr.get_detailed_results('document.jpg')
for result in results['results']:
    print(f"Text: {result['text']}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"Language: {result['language']}")
    print(f"Position: {result['center']}")
    print(f"Size: {result['dimensions']}")
    print()

# Language-specific processing
english_texts = ocr.extract_text_only('document.jpg', language='en')
hindi_texts = ocr.extract_text_only('document.jpg', language='hi')
```

## File Structure

```
├── ocr_utils.py      # Main OCR utility class and functions
├── main.py           # Command-line interface and examples
├── requirements.txt  # Python dependencies
├── README.md         # This file
└── sample.jpg        # Sample image (created with --create-sample)
```

## API Reference

### OCRProcessor Class

#### Constructor
```python
OCRProcessor(languages=['en', 'hi'], use_gpu=False)
```

#### Methods

- **`process_image(image_path, language=None)`**: Process image and return raw OCR results
- **`extract_text_only(image_path, language=None)`**: Extract only text content
- **`get_detailed_results(image_path, language=None)`**: Get detailed results with metadata
- **`_is_valid_image(image_path)`**: Validate image file

### Output Format

#### Simple Text Output
```python
['Hello World!', 'This is a sample image', 'नमस्ते दुनिया']
```

#### Detailed Results
```python
{
    'text_count': 3,
    'languages_detected': ['en', 'hi'],
    'average_confidence': 0.95,
    'results': [
        {
            'text': 'Hello World!',
            'confidence': 0.98,
            'language': 'en',
            'bbox': [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
            'center': (150.5, 100.0),
            'dimensions': (200.0, 50.0)
        },
        # ... more results
    ],
    'summary': 'Detected 3 text elements in 2 languages with 95.0% average confidence'
}
```

## Performance Tips

1. **GPU Acceleration**: Use `--gpu` flag if you have a CUDA-compatible GPU
2. **Language Selection**: Only initialize languages you need to reduce memory usage
3. **Image Quality**: Higher resolution images generally provide better OCR results
4. **Text Orientation**: PaddleOCR automatically detects and corrects text orientation

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`
2. **Model Download**: First run may take time to download language models
3. **Memory Issues**: Reduce number of initialized languages if experiencing memory problems
4. **GPU Issues**: Fall back to CPU mode if GPU acceleration causes problems

### Error Messages

- **"Image file not found"**: Check file path and ensure image exists
- **"Invalid image file"**: Ensure file is a valid image format (JPG, PNG, etc.)
- **"Language not initialized"**: Check language code and ensure it's in the initialized list

## Extending the Utility

The code is designed to be easily extensible:

### Adding New Languages
```python
# Add new language to the languages list
ocr = OCRProcessor(languages=['en', 'hi', 'new_lang'])
```

### Batch Processing
```python
def process_multiple_images(image_paths):
    ocr = OCRProcessor()
    results = {}
    for path in image_paths:
        results[path] = ocr.extract_text_only(path)
    return results
```

### Custom Preprocessing
```python
def custom_preprocess(image_path):
    # Add your image preprocessing logic here
    # e.g., resize, enhance contrast, etc.
    return processed_image_path
```

## Dependencies

- **paddlepaddle**: Deep learning framework
- **paddleocr**: OCR toolkit
- **opencv-python**: Image processing
- **Pillow**: Image handling
- **numpy**: Numerical operations

## License

This project is open source and available under the MIT License.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the utility.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the examples in `main.py --examples`
3. Check PaddleOCR documentation for advanced usage 