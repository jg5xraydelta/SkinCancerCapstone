Essential Instruction URLs:
https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/master/09-serverless/workshop

https://github.com/alexeygrigorev/workshops/tree/main/mlzoomcamp-k8s



Tech Notes:
I used codespace for this project.  There is a devcontainer.json folder & file that has the added Docker-in-Docker feature.  Codespace runs on docker.  I think if you launch a new codespace using this github(https://github.com/jg5xraydelta/SkinCancerCapstone), then it will look for the devcontainer file automatically.  If docker --version doesn't work, then maybe rebuild the codespace.

The root folder is /workspaces/SkinCancerCapstone.  Once the codespace is built.  You should uv sync.  That should set up the appropriate virtual environment.  

Git and Git LFS gave me fits throughout the project.  The data contains images and I wanted to push them once.  I pushed them and then downloaded git lfs.  There is a file called .gitattributes where large files are listed and git lfs will track and store them differently.  Github doesn't like files over 100MB.
The reason I think I had so much trouble is because once I pushed them, git in-a-sense wouldn't forget them despite git lfs and .gitattributes.  I had to unstage the file and remove it from repo tracking.  Afterwards, commit and a push cleared up my git issues.  Model files are large too and they are listed in .gitattributes.

Conversions were my greatest struggle during this project.  I used claude.ai extensively.  Finally generated the dockerfile and the python script for each conversion.  When running the file conversion dockerfiles, my codespace ran out of space quickly.  The command below ultimately fixed the issue.

$ docker system prune -a --volumes -f 

The model-conversion directory contains the dockerfile needed to convert h5 to keras or saved-model.  If you want Onnx format, then you need to take the keras model to the /to_Onnx/models directory and use the dockerimage in the /to_Onnx directory.

There is a kaggle notebook where I ran some more epochs and I could get access to gpus.  Surprisingly, I didn't get a higher accuracy than the models I ran in codespace on cpus.  Kaggle/Code was 10x faster though.  I have only included the notebook i used in Kaggle with mostly changes to the filepath to the data.  I uploaded the data folder to kaggle and named it skin-dataset.  That folder can be found in /kaggle/input directory once inside a kaggle notebook.

The ResNet_v4_large_15_0.906.hf is my highest accuracy model.  It is the model file used in the conversion folders.  

Deployment: The model was deployed locally, lambdaAWS, and kubernetes.

(1) The kubernetes deployment was carried out in a seperate github/codespace. Some of the files needed for the tf-serving component of that deployment are in the /tfserving directory. See (https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/10-kubernetes/07-kubernetes-tf-serving.md) for additional instructions.

(2) I believe there was a dockerfile similar to the one in /lambdaAWS.  It was used to deploy locally and the test.py script will send a request.  However, that dockerfile morphed into the dockerfile that is there now & the one needed for AWSlambda.

(3) So AWSlambda was tricky for me.  I'm pretty sure the files lambda_function.py, requirements.txt, and sc-model.onnx are what is needed.  An ecr repository has to be created and that is where a docker image will be pushed(see example below).    

I replaced: (you may pick a different 'my-app' name)
'my-app' with 'scmodelonnx'
'123456789012' with my AWS console ID
'us-east-1' with my region

''' 
# 1. Authenticate
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-east-1.amazonaws.com

# 2. Create repository (if needed)
aws ecr create-repository --repository-name my-app --region us-east-1

# 3. Tag your image
docker tag my-app:latest 123456789012.dkr.ecr.us-east-1.amazonaws.com/my-app:latest

# 4. Push to ECR
docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/my-app:latest
'''

Next, you will create a lambda function using a container image.  Then a popup box appears and you will select the container image in your ecr repo.
After your lambda function is created, you can test it with the url below.  

URL USED FOR TESTING: "url": "https://github.com/jg5xraydelta/SkinCancerCapstone/blob/main/data/test/malignant/1.jpg?raw=true"

Finally, to actual deploy the lambda function as a webservice, you'll need to 
create an API Gateway that exposes the Lambda Function.  For more detailed instructions, see (https://www.youtube.com/watch?v=wyZ9aqQOXvs&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR).




*********************************SkinCancerCapstone***********************************
Problem Statement:
Malignant or benign—this is the question that billions of men and women face each year when a new spot appears on their skin due to aging or sun exposure. While a biopsy remains the gold standard for determining whether a spot is cancerous and should be removed, dermatologists possess a keen eye and extensive experience in visually inspecting lesions and assessing their severity with very high accuracy. However, convolutional neural networks can achieve comparable accuracy and serve populations that lack access to highly trained dermatologists. Therefore, a skin cancer image classification model can address the problem of limited access to healthcare professionals and provide early detection and healthcare cost savings to patients.


# Skin Cancer Classification: Malignant or Benign

A deep learning project for binary classification of skin lesions as malignant or benign using convolutional neural networks and transfer learning.

## 📋 Project Overview (train and test only i.e. no validation)

This project implements a ResNet50-based deep learning model to classify skin lesion images into two categories:
- **Malignant** - Cancerous lesions
- **Benign** - Non-cancerous lesions

data url: https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign

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