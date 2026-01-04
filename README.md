# SkinCancerCapstone
Malignant or Benign

# Skin Cancer Classification: Malignant or Benign

A deep learning project for binary classification of skin lesions as malignant or benign using convolutional neural networks and transfer learning.

## 📋 Project Overview

This project implements a ResNet50-based deep learning model to classify skin lesion images into two categories:
- **Malignant** - Cancerous lesions
- **Benign** - Non-cancerous lesions

The model achieves **90.6% accuracy** on the test set, demonstrating strong performance in distinguishing between malignant and benign skin lesions.

## 🎯 Key Features

- **Transfer Learning**: Uses pre-trained ResNet50 architecture for feature extraction
- **Binary Classification**: Malignant vs. Benign skin lesion detection
- **Multiple Model Versions**: Iterative improvements with different architectures and hyperparameters
- **Comprehensive Training Pipeline**: Data preprocessing, augmentation, and evaluation
- **Model Deployment Ready**: Includes conversion tools for production deployment

## 📊 Model Performance

| Model | Accuracy | Details |
|-------|----------|---------|
| ResNet v3 | 90.2% | 12 epochs |
| ResNet v4 Large | **90.6%** | 15 epochs (best model) |

## 🗂️ Repository Structure

```
SkinCancerCapstone/
├── 07-neural-nets-train.ipynb    # Main training notebook
├── 07-neural-nets-test.ipynb     # Model evaluation notebook
├── skincancerresnet.ipynb        # ResNet experimentation
│
├── ResNet_v3_12_0.902.h5         # Trained model (90.2%)
├── ResNet_v4_large_15_0.906.h5   # Best model (90.6%)
│
├── data/                          # Dataset directory
├── model-conversion/              # Model format conversion tools
│   ├── Dockerfile                 # H5 → Keras/SavedModel
│   ├── Dockerfile.onnx            # SavedModel → ONNX
│   ├── Dockerfile.keras-onnx      # Keras → ONNX (direct)
│   ├── convert_h5_model.py
│   ├── convert_savedmodel_to_onnx.py
│   └── convert_keras_to_onnx.py
│
├── aws/                           # AWS deployment configurations
├── lambdaAWS/                     # AWS Lambda functions
├── tfserving/                     # TensorFlow Serving setup
│
├── resnet_v1_*.svg                # Training visualizations
└── Tracking                       # Experiment tracking logs
```

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.10+
TensorFlow 2.17+
Keras
NumPy
Pandas
Matplotlib
Scikit-learn
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/jg5xraydelta/SkinCancerCapstone.git
cd SkinCancerCapstone
```

2. Install dependencies:
```bash
pip install -r requirements.txt
# or using uv
uv sync
```

### Training the Model

Open and run the training notebook:
```bash
jupyter notebook 07-neural-nets-train.ipynb
```

The notebook includes:
- Data loading and preprocessing
- Image augmentation
- Model architecture definition
- Training with various learning rates
- Model evaluation and metrics

### Using Pre-trained Models

Load the best performing model:
```python
from tensorflow import keras

# Load the model
model = keras.models.load_model('ResNet_v4_large_15_0.906.h5')

# Make predictions
predictions = model.predict(your_image_data)
```

## 🔧 Model Conversion Tools

Convert models to different formats for deployment:

### H5 to Keras/SavedModel
```bash
docker build -t h5-converter -f model-conversion/Dockerfile .
docker run -v $(pwd):/models h5-converter /models/ResNet_v4_large_15_0.906.h5 --format savedmodel
```

### Keras to ONNX (for cross-platform deployment)
```bash
docker build -t keras-to-onnx -f model-conversion/Dockerfile.keras-onnx .
docker run -v $(pwd):/models keras-to-onnx /models/ResNet_v4_large_15_0.906.h5 -o /models/model.onnx
```

## 📈 Model Architecture

The model uses a **ResNet50** backbone with the following structure:

```
Input (150x150x3)
    ↓
ResNet50 (pre-trained, frozen)
    ↓
Global Average Pooling
    ↓
Dense (100 units, ReLU)
    ↓
Dropout (0.5)
    ↓
Dense (2 units, Softmax)
    ↓
Output (Malignant/Benign)
```

**Key Configuration:**
- Input Size: 150×150×3 RGB images
- Base Model: ResNet50 (ImageNet pre-trained)
- Trainable Parameters: ~205K (classification head only)
- Non-trainable Parameters: ~23.6M (frozen ResNet50 base)
- Total Parameters: ~23.8M

## 📊 Training Details

**Hyperparameters:**
- Optimizer: Adam
- Learning Rate: Experimented with 0.001, 0.01, and others
- Batch Size: 32
- Epochs: 12-15
- Loss Function: Binary Crossentropy
- Metrics: Accuracy, Precision, Recall

**Data Augmentation:**
- Random rotation
- Width and height shifts
- Horizontal flip
- Zoom range
- Brightness adjustment

## 📉 Training Visualizations

The repository includes training history visualizations:
- `resnet_v1_0_001.svg` - Training with learning rate 0.001
- `resnet_v1_0_01.svg` - Training with learning rate 0.01
- `resnet_v1_all_lr.svg` - Comparison across learning rates

## 🔬 Evaluation

Model evaluation is performed in `07-neural-nets-test.ipynb`:
- Accuracy metrics
- Confusion matrix
- Precision and recall
- ROC curve analysis
- Sample predictions visualization

## ☁️ Deployment

### AWS Lambda
The `lambdaAWS/` directory contains serverless deployment configurations for running inference on AWS Lambda.

### TensorFlow Serving
The `tfserving/` directory includes Docker configurations for deploying the model with TensorFlow Serving.

## 🛠️ Development Setup

The project uses:
- **Dev Containers**: `.devcontainer/` for consistent development environment
- **GitHub Actions**: `.github/` for CI/CD workflows
- **Python Package Management**: `pyproject.toml` and `uv.lock` for dependency management

## 📝 Experiment Tracking

Training experiments and hyperparameter tuning results are logged in the `Tracking` file.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Dataset: [Specify your dataset source here]
- ResNet50 architecture from Keras Applications
- Transfer learning techniques from the ML community

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This is an educational/research project. For medical diagnosis, always consult qualified healthcare professionals.