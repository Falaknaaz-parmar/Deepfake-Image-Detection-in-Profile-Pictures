# Deepfake Image Detection in Profile Pictures

A deep learning-based system for detecting AI-generated and manipulated images using transfer learning with EfficientNet-B0 architecture.

## 📋 Table of Contents
- [Problem Statement](#problem-statement)
- [Methodology](#methodology)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)

## 🎯 Problem Statement

With the rise of deepfake technology and AI-generated content, distinguishing authentic images from manipulated ones has become increasingly critical. This project addresses the challenge of identifying fake profile pictures through:

- **Automated Detection**: Binary classification of images as Real or Fake
- **High Accuracy**: Achieving 95%+ accuracy using state-of-the-art deep learning
- **Explainable AI**: Providing visual explanations for model predictions using Grad-CAM
- **Practical Deployment**: User-friendly web interface for real-world applications

## 🔬 Methodology

### Transfer Learning Approach
- Pre-trained EfficientNet-B0 model (ImageNet weights)
- Fine-tuned for binary classification (Real/Fake)
- Custom classifier head with dropout regularization

### Training Strategy
- **Data Augmentation**: Random flip, rotation, color jitter, affine transforms
- **Optimization**: Adam optimizer with learning rate scheduling
- **Regularization**: Dropout (30%), weight decay, early stopping
- **Hardware**: GPU-accelerated training (CUDA support)

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC Curve
- Confusion Matrix
- Per-class performance analysis

## 📊 Dataset

### Structure
```
data/
├── Train/          # Training set (70%)
│   ├── Fake/
│   └── Real/
├── Validation/     # Validation set (15%)
│   ├── Fake/
│   └── Real/
└── Test/           # Test set (15%)
    ├── Fake/
    └── Real/
```

### Dataset Specifications
- **Total Images**: ~140,000
- **Training**: 98,000 images
- **Validation**: 21,000 images
- **Test**: 21,000 images
- **Classes**: Real (authentic images), Fake (AI-generated/manipulated)

**Note**: Dataset not included in repository due to size constraints. Place your dataset in the `data/` directory following the structure above.

## 🧠 Model Architecture

### EfficientNet-B0 Backbone
```
Input (160×160×3)
    ↓
EfficientNet-B0 (Pretrained)
    ↓
Global Average Pooling
    ↓
Dropout (0.3)
    ↓
Linear (2 classes)
    ↓
Softmax → [P(Fake), P(Real)]
```

### Model Specifications
- **Parameters**: ~4 million trainable
- **Input Size**: 160×160 pixels
- **Architecture**: EfficientNet-B0
- **Output**: Binary classification with confidence scores

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM

### Setup
```bash
# Clone the repository
git clone https://github.com/Falaknaaz-parmar/Deepfake-Image-Detection-in-Profile-Pictures.git
cd Deepfake-Image-Detection-in-Profile-Pictures

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## 📖 Usage

### 1. Training
```bash
python train.py
```
- Trains model on your dataset
- Saves checkpoints to `outputs/checkpoints/`
- Logs training metrics to `outputs/logs/`
- Generates visualizations in `outputs/metrics/`

### 2. Evaluation
```bash
python evaluate.py
```
- Evaluates model on test set
- Generates confusion matrix, ROC curve
- Saves results to `outputs/evaluation/`

### 3. Single Image Prediction
```bash
python predict.py --image path/to/image.jpg
```

### 4. Web Application
```bash
python app.py
```
Open browser to `http://localhost:5000`

**Features**:
- Drag-and-drop image upload
- Real-time prediction with confidence scores
- Grad-CAM heatmap visualization
- Warning levels based on detection confidence

## 📈 Results

### Performance Metrics
| Metric | Value |
|--------|-------|
| Test Accuracy | 95.2% |
| Precision (Fake) | 94.8% |
| Recall (Fake) | 95.6% |
| F1-Score | 95.2% |
| ROC-AUC | 0.987 |

### Training Details
- **Epochs**: 30 (early stopping at epoch 25)
- **Batch Size**: 48
- **Learning Rate**: 0.0001
- **Training Time**: ~5-7 min/epoch (GPU)

### Visualizations
The system generates:
- Confusion matrices (raw and normalized)
- ROC curves
- Training history plots (loss & accuracy)
- Grad-CAM heatmaps for explainability

## 📁 Project Structure

```
deepfake_detection/
├── config/
│   └── config.yaml              # Configuration file
├── src/
│   ├── data/
│   │   ├── dataset.py           # Dataset loader
│   │   └── transforms.py        # Image transformations
│   ├── models/
│   │   └── model.py             # Model architecture
│   ├── training/
│   │   └── trainer.py           # Training loop
│   └── utils/
│       ├── explainable_ai.py    # Grad-CAM implementation
│       ├── logger.py            # Logging utilities
│       ├── metrics.py           # Evaluation metrics
│       └── visualize.py         # Visualization tools
├── templates/
│   └── index.html               # Web UI
├── static/
│   ├── style.css                # Styling
│   └── script.js                # Frontend logic
├── api/
│   └── index.py                 # API endpoints
├── outputs/
│   ├── checkpoints/             # Model checkpoints (not in repo)
│   ├── metrics/                 # Training visualizations
│   └── evaluation/              # Evaluation results
├── train.py                     # Training script
├── evaluate.py                  # Evaluation script
├── predict.py                   # Prediction script
├── app.py                       # Flask web app
└── requirements.txt             # Dependencies
```

## 🛠️ Technologies Used

- **Deep Learning**: PyTorch, torchvision
- **Model**: EfficientNet-B0 (transfer learning)
- **Web Framework**: Flask
- **Visualization**: Matplotlib, Seaborn
- **Explainability**: Grad-CAM
- **Data Processing**: NumPy, Pandas, OpenCV, PIL

## 📝 Key Features

✅ High-accuracy deepfake detection (95%+)  
✅ Transfer learning with EfficientNet-B0  
✅ Comprehensive data augmentation  
✅ Explainable AI with Grad-CAM visualizations  
✅ User-friendly web interface  
✅ RESTful API for integration  
✅ GPU acceleration support  
✅ Detailed metrics and evaluation

## 🎓 Academic Context

This project demonstrates:
- Modern deep learning techniques for image classification
- Transfer learning and fine-tuning strategies
- Practical deployment of AI models
- Ethical considerations in deepfake detection
- Professional software engineering practices

## 📧 Contact

**Author**: Falaknaaz Parmar  
**Repository**: [Deepfake-Image-Detection-in-Profile-Pictures](https://github.com/Falaknaaz-parmar/Deepfake-Image-Detection-in-Profile-Pictures)

---

**Note**: This project is for educational and research purposes. Model checkpoints are not included due to file size constraints. Train the model using your own dataset or contact for pretrained weights.
