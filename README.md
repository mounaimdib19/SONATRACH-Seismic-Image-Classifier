# SONATRACH-Seismic-Image-Classifier

Automatic classification system for seismic data developed for SONATRACH's deblending project (R&D N°221). Classifies seismic images as clean or noisy using machine learning and computer vision.

## Features

- **Binary Classification**: Automatically classify seismic images as clean or noisy
- **Automatic Annotation**: Detect and circle interference regions in seismic data
- **Web Interface**: Interactive Streamlit app for easy classification workflow
- **Batch Processing**: Handle multiple images at once
- **Export Results**: Download organized folders and CSV reports

## Requirements

- Python 3.8 or higher
- 4GB RAM minimum
- Modern web browser (Chrome, Firefox, Edge)

## Installation

### 1. Clone or download the project

```bash
git clone <your-repo-url>
cd seismic-classifier
```

### 2. Install dependencies

```bash
pip install streamlit opencv-python numpy pandas scikit-learn pillow matplotlib scikit-image scipy
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
streamlit==1.28.0
opencv-python==4.8.1.78
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
pillow==10.0.0
matplotlib==3.7.2
scikit-image==0.21.0
scipy==1.11.2
```

## Project Structure

```
seismic-classifier/
├── seismic_app.py                 # Main Streamlit web application
├── simple_seismic_classifier.py   # Classification module
├── automatic_seismic_annotator.py # Annotation module
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── data/                          # Your data folder
    ├── training/
    │   ├── clean/                 # Clean training images
    │   └── noisy/                 # Noisy training images
    └── test/                      # Images to classify
```

## Usage

### Quick Start

1. **Run the application**
```bash
streamlit run seismic_app.py
```

2. **Train the model** (in sidebar):
   - Upload clean images (no interference)
   - Upload noisy images (with interference)
   - Click "Train Model"

3. **Classify test images**:
   - Upload your test dataset
   - Click "Start Classification"
   - Navigate through results with "Next" button

4. **Download results**:
   - Download clean images ZIP
   - Download noisy images ZIP
   - Download CSV report

### Using the Classification Module Directly

```python
from simple_seismic_classifier import SimpleSeismicNoiseClassifier
import cv2

# Initialize classifier
classifier = SimpleSeismicNoiseClassifier()

# Load training data
clean_images = [cv2.imread(f'data/training/clean/img{i}.jpg') for i in range(10)]
noisy_images = [cv2.imread(f'data/training/noisy/img{i}.jpg') for i in range(10)]

# Train
classifier.train(clean_images, noisy_images)

# Predict
test_image = cv2.imread('data/test/test_image.jpg')
result = classifier.predict(test_image)
print(f"Is noisy: {result['has_noise']}, Confidence: {result['confidence']:.1%}")
```

### Using the Annotation Module

```python
from automatic_seismic_annotator import AutomaticSeismicAnnotator
import cv2

# Initialize annotator
annotator = AutomaticSeismicAnnotator()

# Load and annotate image
image = cv2.imread('seismic_image.jpg')
noise_regions = annotator.annotate_image(image, method='advanced', show_results=True)

# Batch process folder
annotator.batch_annotate('data/test/', output_folder='results/')
```

## Configuration

Adjust detection sensitivity in `automatic_seismic_annotator.py`:

```python
annotator.noise_threshold = 0.3    # Lower = more sensitive (0.1 - 0.5)
annotator.min_noise_size = 50      # Minimum noise region size in pixels
annotator.max_noise_size = 5000    # Maximum noise region size in pixels
```

## Output

### Folder Structure
```
output_clean/          # Clean images
output_noisy/          # Noisy images
results.csv            # Classification report
```

### CSV Report Format
```csv
filename,classification,confidence,clean_prob,noisy_prob
image1.jpg,Clean,0.95,0.95,0.05
image2.jpg,Noisy,0.88,0.12,0.88
```

## Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'cv2'`
- **Solution**: Run `pip install opencv-python`

**Issue**: Images not loading
- **Solution**: Ensure images are in supported formats (PNG, JPG, JPEG, TIFF, BMP)

**Issue**: Low accuracy
- **Solution**: Provide more diverse training data (minimum 20 images per class)

**Issue**: Streamlit won't start
- **Solution**: Check if port 8501 is available, or specify: `streamlit run seismic_app.py --server.port 8502`

**Issue**: Out of memory
- **Solution**: Process images in smaller batches, reduce image size

## Performance

- **Classification speed**: <0.1 seconds per image
- **Model accuracy**: 88-92% (depends on training data)
- **Batch processing**: ~100 images/minute
- **Memory usage**: ~500MB for typical datasets

## Technical Details

- **Algorithm**: Random Forest Classifier (100 trees)
- **Features**: 31 features extracted per image (statistical, texture, frequency, gradient)
- **Normalization**: StandardScaler for feature scaling
- **Validation**: 80/20 train-test split

## License

Developed for SONATRACH DC-RD (Direction Centrale Recherche et Développement).

## Contact

For questions or support:
- Developer: DIB Abdelmounaim.
- Supervisor: MERCHICHI Mohamed Rayane. 
- Department: SONATRACH DC-RD, Boumerdès

## Acknowledgments

Based on SONATRACH R&D Project N°221: "Deblending des données sismiques 3D"
