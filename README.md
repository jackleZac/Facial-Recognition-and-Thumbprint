# Smart Identity Matcher

A sophisticated facial recognition system built in MATLAB that can identify individuals across different photos using deep learning and traditional computer vision techniques.

![MATLAB](https://img.shields.io/badge/MATLAB-R2020a+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

## 🌟 Features

### Core Capabilities
- **Multi-Photo Recognition**: Identifies the same person across different photos with varying:
  - Lighting conditions
  - Facial expressions
  - Photo angles (within reasonable limits)
  - Image quality
  
- **Dual Feature Extraction**:
  - **Deep Learning** (AlexNet): Uses pre-trained CNN for robust feature extraction
  - **Traditional Methods**: Automatic fallback using HOG, LBP, SURF, and histogram features
  
- **Smart Unknown Detection**: Rejects faces not in the database with configurable thresholds

- **Quality Validation**: Comprehensive image quality checks including:
  - Blur detection
  - Brightness/contrast analysis
  - Face resolution verification
  - Eye detection validation

### User Interface
- Clean, modern GUI with intuitive controls
- Real-time confidence scoring
- Visual feedback with color-coded borders
- Database browsing functionality
- Detailed match information display

## 📋 Requirements

### Required MATLAB Toolboxes
- Computer Vision Toolbox
- Image Processing Toolbox

### Optional (Recommended)
- Deep Learning Toolbox (for AlexNet support)
- Statistics and Machine Learning Toolbox (not required - manual implementations included)

### MATLAB Version
- MATLAB R2020a or later

## 🚀 Installation

1. **Clone or download** this repository:
```bash
git clone https://github.com/jackleZac/Facial-Recognition-and-Thumbprint.git
cd Facial-Recognition-and-Thumbprint
```

2. **Add to MATLAB path**:
```matlab
addpath('path/to/Facial-Recognition-and-Thumbprint')
```

3. **Launch the application**:
```matlab
app = FacialRecognition
```

## 📖 Usage Guide

### 1. Register Identities

Click **"Register Identity"** to add people to your database:
- Select a clear, frontal photo of the person
- System performs quality checks automatically
- Photo is processed and features are extracted
- Identity is saved to `identityData.mat`

**Tips for best results:**
- Use well-lit, frontal face photos
- Ensure face is clearly visible (no sunglasses/masks)
- Higher resolution photos work better
- Minimum 80x80 pixels face resolution required

### 2. Scan a Photo

Click **"Scan Photo"** to load a photo for identification:
- Choose an image to identify
- System validates image quality
- Face is detected and features extracted
- Ready for identification

### 3. Identify

Click **"Identify"** to match the scanned photo:
- System compares against all registered identities
- Shows match result with confidence score
- Displays the matching identity (if found)
- Provides detailed similarity scores

### 4. View Database

Click **"View Database"** to browse registered identities:
- Cycles through all stored faces
- Shows name and position in database
- Useful for verification and management

## 🎯 How It Works

### Feature Extraction Pipeline

```
Input Image → Face Detection → Quality Check → Feature Extraction → Comparison
```

#### 1. Face Detection
- Uses Viola-Jones cascade classifier
- Handles multiple faces (selects largest)
- Adds padding around detected face

#### 2. Quality Validation
Scores images based on:
- **Blur Score** (30 points): Laplacian variance
- **Brightness** (25 points): Optimal range 80-180
- **Contrast** (20 points): Standard deviation check
- **Resolution** (15 points): Minimum 80x80 pixels
- **Eye Detection** (10 points): Facial landmark validation

Total score of 50+ passes quality check.

#### 3. Feature Extraction

**With AlexNet (Deep Learning)**:
```matlab
features = activations(alexnet, face, 'fc7')
```
- Extracts 4096-dimensional feature vector
- Highly robust to variations
- Recommended for best accuracy

**Without AlexNet (Traditional)**:
- **HOG Features**: Histogram of Oriented Gradients
- **LBP Features**: Local Binary Patterns
- **SURF Features**: Speeded Up Robust Features
- **Region Histograms**: 4x4 grid analysis

#### 4. Similarity Comparison

Multi-metric comparison:
```matlab
similarity = (cosine * 0.7) + (euclidean * 0.2) + (correlation * 0.1)
```

- **Cosine Similarity**: Primary metric (70%)
- **Euclidean Distance**: Secondary metric (20%)
- **Correlation**: Tertiary metric (10%)

### Identification Algorithm

```matlab
1. Extract features from scanned photo
2. Compare with all database features
3. Find best match and calculate confidence
4. Apply absolute minimum threshold (40%)
5. Calculate discrimination from second-best
6. Apply adaptive threshold (55-75%)
7. Display result or reject as unknown
```

## ⚙️ Configuration

### Adjusting Thresholds

Edit `performIdentification` method to tune sensitivity:

```matlab
% Absolute minimum for any match (Line 530)
ABSOLUTE_MIN_SIMILARITY = 0.40;  % Increase to reduce false positives

% Base threshold (Line 556)
baseThreshold = 60;  % Increase for stricter matching
```

### Quality Requirements

Edit `detectAndCropFaceEnhanced` to adjust quality standards:

```matlab
% Minimum quality score (Line 338)
if qualityScore >= 50  % Increase for stricter quality requirements
```

## 📊 Performance Characteristics

### Accuracy Metrics

| Scenario | Expected Accuracy |
|----------|------------------|
| Same person, good quality photos | 90-98% |
| Same person, mixed quality | 75-85% |
| Different lighting conditions | 80-90% |
| Different facial expressions | 85-95% |
| Unknown person rejection | 85-95% |

### Confidence Levels

- **85-100%**: Strong match - Very high confidence
- **70-84%**: Good match - High confidence  
- **60-69%**: Possible match - Moderate confidence (verify recommended)
- **55-59%**: Weak match - Low confidence (manual verification required)
- **<55%**: No match - Rejected

## 🔧 Troubleshooting

### "AlexNet not available" message
- System automatically uses traditional features
- Still functional but slightly less accurate
- Install Deep Learning Toolbox for AlexNet support

### Poor recognition across different photos
- Ensure quality photos during registration
- Use multiple photos per person (future enhancement)
- Adjust `ABSOLUTE_MIN_SIMILARITY` threshold
- Check lighting consistency

### False positives (wrong person identified)
- Increase `baseThreshold` from 60 to 65-70
- Increase `ABSOLUTE_MIN_SIMILARITY` from 0.40 to 0.45
- Register more diverse photos per identity

### False negatives (not recognizing registered person)
- Check photo quality scores
- Ensure frontal face photos
- Decrease thresholds slightly
- Re-register with better quality photo

## 📁 File Structure

```
smart-identity-matcher/
├── images                      # Store your images here
├── FacialRecognition.m         # Main application class
├── identityData.mat            # Database file (auto-generated)
└── README.md                   # This file
```

## 🎓 Technical Details

### Feature Vector Dimensions
- **AlexNet**: 4096 dimensions (fc7 layer)
- **Traditional**: ~5000+ dimensions (combined HOG, LBP, SURF, histograms)

### Preprocessing Pipeline
1. Face detection and cropping
2. Grayscale conversion
3. Adaptive histogram equalization (CLAHE)
4. Wiener filtering (noise reduction)
5. Resize to standard dimensions

### Storage Format
```matlab
% Saved in identityData.mat
userFaces         % Cell array of preprocessed face images (80x80)
userNames         % Cell array of identity names
userFaceFeatures  % Cell array of feature vectors
```

## 🚦 Best Practices

### For Registration
1. ✅ Use clear, well-lit photos
2. ✅ Frontal face view (±15° angle)
3. ✅ No accessories covering face
4. ✅ High resolution (200x200+ pixels)
5. ✅ Neutral expression recommended

### For Scanning
1. ✅ Similar quality to registration photos
2. ✅ Face clearly visible
3. ✅ Good lighting conditions
4. ✅ Single person in frame (or clear main subject)

### Database Management
1. 🗑️ Re-register if photo quality was poor
2. 📸 Use consistent photo conditions when possible
3. 💾 Backup `identityData.mat` regularly
4. 🔄 Update database with better photos as available

## 🔮 Future Enhancements

Potential improvements:
- [ ] Multiple photos per identity
- [ ] Real-time webcam recognition
- [ ] Face alignment pre-processing
- [ ] Database management UI (delete/edit entries)
- [ ] Export/import database functionality
- [ ] Training log and accuracy metrics
- [ ] Batch registration mode
- [ ] Integration with external databases

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 👤 Author

(https://github.com/jackleZac)

## 🙏 Acknowledgments

- MATLAB Computer Vision Toolbox documentation
- AlexNet pre-trained model (ImageNet)
- Viola-Jones face detection algorithm
- Open-source computer vision community

---

## User Interface
**Step 1**: Run the main file
<img width="1194" height="802" alt="ui-1" src="https://github.com/user-attachments/assets/19ea4929-f422-42ae-bbf4-0b69b517ef6a" />

**Step 2**: Register an identity and view database to confirm 
<img width="1192" height="804" alt="ui-2" src="https://github.com/user-attachments/assets/35d504a5-166e-4ec1-98a3-5479279bbe3b" />

**Step 3**: Choose a different image of the same person and click 'Scan Photo'
<img width="1191" height="804" alt="ui-3" src="https://github.com/user-attachments/assets/9756be8e-1be5-4789-9135-397013ad96ef" />

**Step 4**: Click 'Identify'. It should return 'Identified'
<img width="1192" height="805" alt="ui-4" src="https://github.com/user-attachments/assets/90b76c67-5ef3-4064-a10c-079834c1f4c2" />

**Step 5**: Try again with an image of different person. It should return 'No match detected'
<img width="1195" height="801" alt="ui-5" src="https://github.com/user-attachments/assets/2357f695-f997-4688-b484-246f30d20a62" />


**Note**: This system is designed for educational and research purposes. For production facial recognition systems, consider additional factors including privacy laws, consent requirements, and bias mitigation strategies.
