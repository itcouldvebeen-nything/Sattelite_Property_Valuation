# Satellite-Based Property Valuation Pipeline

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A multimodal machine learning system that predicts residential property values by combining traditional tabular features with satellite imagery analysis. The system employs a late-fusion stacking ensemble architecture, achieving state-of-the-art performance with an **R² score of 0.9044**.

## Table of Contents

- [What the Project Does](#what-the-project-does)
- [Why the Project is Useful](#why-the-project-is-useful)
- [Key Features](#key-features)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Usage Examples](#usage-examples)
- [Model Architecture](#model-architecture)
- [Performance](#performance)
- [Contributing](#contributing)
- [Support](#support)

## What the Project Does

This project implements a **Satellite-Based Property Valuation Pipeline** that predicts house prices using a hybrid approach:

- **Tabular Data**: Traditional property features (bedrooms, square footage, grade, location, etc.)
- **Satellite Imagery**: High-resolution satellite images (600×600) from Mapbox API
- **Late-Fusion Architecture**: Combines both modalities through a two-level stacking ensemble

The system extracts visual features using EfficientNet-B0 (pre-trained on ImageNet), reduces dimensionality with UMAP manifold learning, and combines them with engineered tabular features in a stacking ensemble of gradient-boosted trees.

## Why the Project is Useful

### Business Impact

- **Reduced Valuation Error**: ~$1,883 MAE reduction per property
- **Portfolio Risk Mitigation**: $18.8 Million in reduced valuation risk for 10,000 properties
- **Luxury Asset Accuracy**: Particularly effective for high-value properties with complex architectural features

### Technical Advantages

- **Multimodal Learning**: Leverages both structured and visual data for improved predictions
- **Interpretability**: Includes SHAP analysis and Grad-CAM visualizations
- **Production-Ready**: Efficient inference pipeline with ~80-120ms per-property latency
- **Scalable**: Handles 16,000+ properties with async image downloading

## Key Features

- 🛰️ **Satellite Image Processing**: Automated download and feature extraction from Mapbox API
- 🎯 **Advanced Feature Engineering**: "God Features" including neighborhood clustering, relative size metrics, and polynomial interactions
- 🧠 **Ensemble Learning**: Stacking ensemble with XGBoost, LightGBM, CatBoost, and RidgeCV meta-learner
- 📊 **Model Interpretability**: SHAP values and Grad-CAM visualizations for explainability
- 🔄 **UMAP Dimensionality Reduction**: Non-linear manifold learning preserving local structure
- ⚡ **GPU Acceleration**: Efficient PyTorch-based feature extraction with CUDA support

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training and inference)
- Mapbox API key (for satellite image downloads)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/property_valuation.git
cd property_valuation
```

### Step 2: Download Data

**Important**: Download the data zip folder from the [Google Drive link](https://drive.google.com/your-link-here) and extract it to the project root. The data should include:

- `data/raw/train(1).csv` - Training dataset with property features
- `data/raw/test2.csv` - Test dataset
- `data/images/` - Directory for satellite images (will be populated by the downloader)

### Step 3: Create Virtual Environment

```bash
python -m venv property_env
source property_env/bin/activate  # On Windows: property_env\Scripts\activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Configure Mapbox API Key

Edit `data_fetcher.py` and replace the placeholder with your Mapbox API key:

```python
API_KEY = "your_actual_mapbox_api_key"
```

You can obtain a free API key from [Mapbox](https://www.mapbox.com/).

## Getting Started

### 1. Download Satellite Images

Run the data fetcher script to download satellite images for all properties:

```bash
python data_fetcher.py
```

This script will:
- Read property coordinates from `data/raw/train(1).csv`
- Download 600×600 satellite images from Mapbox API
- Save images to `data/images/` directory
- Use multi-threaded async downloading for efficiency

**Note**: This process may take several hours depending on your API rate limits. The script includes rate limiting and retry logic.

### 2. Run Preprocessing

Open `notebooks/preprocessing.ipynb` and execute all cells to:

- Engineer tabular features (neighborhood clusters, relative sizes, polynomial features)
- Create geospatial visualizations
- Generate processed feature files

### 3. Extract Visual Embeddings

Open `notebooks/model_training.ipynb` and run the feature extraction cells to:

- Extract 1,280-dimensional feature vectors using EfficientNet-B0
- Apply UMAP dimensionality reduction to 15 dimensions
- Save embeddings to `data/processed/train_embeddings.pt`

### 4. Train the Model

Continue in `notebooks/model_training.ipynb` to:

- Combine tabular and visual features
- Train the stacking ensemble (XGBoost, LightGBM, CatBoost + RidgeCV)
- Evaluate model performance
- Generate SHAP explanations

## Project Structure

```
property_valuation/
├── data/
│   ├── raw/                    # Raw CSV files (download from Google Drive)
│   │   ├── train(1).csv
│   │   └── test2.csv
│   ├── images/                 # Satellite images (downloaded via data_fetcher.py)
│   └── processed/              # Processed features and embeddings
│       ├── train_embeddings.pt
│       └── train_engineered.csv
├── notebooks/
│   ├── preprocessing.ipynb   # Feature engineering and EDA
│   └── model_training.ipynb    # Model training and evaluation
├── output/                     # Generated visualizations and results
├── runs/                       # Training logs and checkpoints
├── data_fetcher.py            # Satellite image downloader
├── requirements.txt            # Python dependencies
├── Technical_Project_Report.md # Detailed technical documentation
└── README.md                   # This file
```

## Usage Examples

### Downloading Images for New Properties

```python
import pandas as pd
from data_fetcher import download_one_image

# Load your property data
df = pd.read_csv("data/raw/train(1).csv")

# Download image for a single property
row = df.iloc[0]
result = download_one_image(row)
print(f"Download status: {result}")
```

### Extracting Visual Features

```python
import torch
from torchvision import models, transforms
from PIL import Image

# Load pre-trained EfficientNet-B0
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
model.classifier = torch.nn.Identity()  # Remove classification head
model.eval()

# Load and preprocess image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image = Image.open("data/images/12345.jpg").convert('RGB')
image_tensor = transform(image).unsqueeze(0)

# Extract features
with torch.no_grad():
    features = model(image_tensor)  # Shape: [1, 1280]
```

### Making Predictions

```python
import pickle
import numpy as np

# Load trained model (saved from training notebook)
with open('models/stacking_ensemble.pkl', 'rb') as f:
    model = pickle.load(f)

# Prepare features (tabular + visual embeddings)
tabular_features = [...]  # Your engineered tabular features
visual_embeddings = [...]  # UMAP-reduced visual features (15-dim)
combined_features = np.concatenate([tabular_features, visual_embeddings])

# Predict
predicted_price = model.predict([combined_features])[0]
print(f"Predicted property value: ${predicted_price:,.2f}")
```

## Model Architecture

### Late-Fusion Stacking Ensemble

```
Input Layer
├── Tabular Data Stream
│   ├── Raw Features (bedrooms, sqft, grade, etc.)
│   └── Engineered Features (Size_Relative_to_Neighbors, Grade_Polynomials)
│
└── Satellite Imagery Stream
    ├── Mapbox API (600×600 resolution)
    └── EfficientNet-B0 Backbone (ImageNet pre-trained)
        └── Feature Extraction (1,280-dim vectors)
            └── UMAP Manifold Learning
                └── Dimensionality Reduction (15-dim embeddings)

Feature Fusion Layer
├── Concatenated Feature Vector (Tabular + Visual)
└── Level 1 Stacking Ensemble
    ├── XGBoost
    ├── LightGBM
    └── CatBoost

Meta-Learner Layer
└── RidgeCV
    └── Final Prediction (Property Value)
```

### Key Components

- **EfficientNet-B0**: Pre-trained CNN for visual feature extraction
- **UMAP**: Non-linear dimensionality reduction (1,280 → 15 dimensions)
- **Stacking Ensemble**: Level 1 (XGBoost, LightGBM, CatBoost) + Level 2 (RidgeCV)
- **Feature Engineering**: Neighborhood clustering, relative metrics, polynomial interactions

## Performance

### Model Metrics

| Model Configuration | R² Score | MAE Reduction |
|---------------------|----------|---------------|
| Tabular Baseline (with God Features) | 0.9018 | Baseline |
| **Multimodal Stacking Ensemble** | **0.9044** | **~$1,883** |

### Computational Requirements

- **Training Time**: ~5-7 hours (EfficientNet-B0: 3-4h, UMAP: 45-60min, Ensemble: 1.5-2h)
- **Inference Latency**: ~80-120ms per property
- **Storage**: ~20-25 GB (images: 18 GB, features: 80 MB, models: 800 MB)
- **Hardware**: NVIDIA GPU (V100/A100 recommended) for training, T4/V100 for inference

### Key Insights

- **Blue Pixel Premium**: Properties with visible water features trade at a 12.4% premium
- **Luxury Asset Focus**: Visual features provide disproportionate value for high-end properties
- **Model Interpretability**: Grad-CAM confirms the model focuses on built structures (pools, rooflines) rather than background noise

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Support

### Getting Help

- **Documentation**: See [Technical_Project_Report.md](Technical_Project_Report.md) for detailed technical documentation
- **Issues**: Open an issue on GitHub for bug reports or feature requests
- **Discussions**: Use GitHub Discussions for questions and community support

### Common Issues

**Issue**: Mapbox API rate limiting
- **Solution**: The `data_fetcher.py` script includes automatic rate limit handling with exponential backoff

**Issue**: CUDA out of memory during feature extraction
- **Solution**: Reduce batch size in `DataLoader` (default: 64) or use CPU mode

**Issue**: Missing images after download
- **Solution**: Check that property IDs in CSV match image filenames. The script handles ID normalization automatically.

## Acknowledgments

- Mapbox for satellite imagery API
- PyTorch team for EfficientNet pre-trained models
- UMAP developers for manifold learning implementation
- The open-source machine learning community

---

**Note**: This project is part of a research initiative. For detailed methodology, results, and business impact analysis, refer to [Technical_Project_Report.pdf](Technical_Project_Report.pdf).

