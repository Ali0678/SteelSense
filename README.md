# SteelSense: Automated Surface Defect Detection

SteelSense is an end-to-end Deep Learning application designed to automate quality control in steel manufacturing. Utilizing a custom Convolutional Neural Network (CNN), the system detects and classifies surface anomalies in real-time, achieving >90% validation accuracy on industry-standard datasets.

## Project Overview

Manual inspection of steel surfaces is time-consuming and prone to human error. SteelSense addresses this by providing a robust, automated computer vision pipeline. The project encompasses the full machine learning lifecycle: data engineering, model architecture design, training optimization, and deployment via a web-based user interface.

## Key Features

- **Multi-Class Classification:** Accurately identifies 6 distinct defect topologies:
  - Crazing
  - Inclusion
  - Patches
  - Pitted Surface
  - Rolled-in Scale
  - Scratches
- **Custom Architecture:** A PyTorch-based CNN built from scratch, utilizing Adaptive Average Pooling to handle variable input dimensions and Batch Normalization to stabilize training.
- **Interactive Interface:** A Streamlit-powered web application allowing users to upload raw images and receive instantaneous inference and confidence metrics.

## Technical Stack

- **Language:** Python 3.10+
- **Deep Learning Framework:** PyTorch, Torchvision
- **Data Processing:** NumPy, PIL (Python Imaging Library)
- **Visualization:** Matplotlib
- **Web Framework:** Streamlit

## Dataset

The model is trained on the NEU Surface Defect Database. The dataset consists of 1,800 grayscale images (200×200 pixels), evenly distributed across 6 defect classes (300 images per class).

Find it here: https://www.kaggle.com/datasets/rdsunday/neu-urface-defect-database

> **Note:** This dataset is unorganized. To organize it into 6 distinct folders, run `prepare_data.py`.

## Installation and Usage

1. Clone the repository:

```bash
git clone https://github.com/Ali0678/SteelSense.git
cd SteelSense
```

2. Create and activate a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Prepare the dataset:

```bash
python src/prepare_data.py
```

5. Launch the web application:

```bash
streamlit run src/app.py
```

## Model Architecture

The core inference engine is a custom CNN featuring:

- **Feature Extraction:** 3 Convolutional Blocks (`Conv2d` → `BatchNorm2d` → `ReLU` → `MaxPool2d`)
- **Spatial Adaptation:** Adaptive Average Pooling (4×4) to ensure a fixed tensor size prior to the fully connected layers
- **Classification Head:** Linear layers with Dropout (p=0.5) for regularization, outputting raw logits for Cross-Entropy optimization