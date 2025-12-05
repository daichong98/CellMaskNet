# CellMaskNet
 A Nucleus Representation Aware Deep  Learning Model for Cell Segmentation and Classification  in Histopathological Images
## Introduction

**CellMaskNet** is a comprehensive deep learning pipeline designed for precise segmentation and classification of nuclei in pathology images. It leverages a multi-feature fusion strategy, combining:
*   **Local Features**: Extracted via MobileViT.
*   **Global Features**: Extracted via Swin Transformer.
*   **Morphological Features**: Geometric properties of the nuclei.
*   **Ring Features**: Texture information from the nuclear boundary.
*   **Graph Features**: Spatial relationships captured by Graph Attention Networks (GAT).
*   **Co-Attention Mechanism**: To effectively fuse local and global visual features.

This repository contains the complete workflow from data preprocessing and segmentation (using HoverNet) to feature extraction and final classification.

## Pipeline Overview

The project is structured into sequential steps:

1.  **Step 0: Data Preprocessing** (`step0_data_preprocessor.py`)
    *   Prepares the PanNuke dataset for processing.
2.  **Step 1: Segmentation** (`step1_hovernet_batch.py`)
    *   Runs HoverNet to generate instance segmentation masks for all images.
3.  **Step 2: Nuclei Extraction** (`step2_extract_nuclei.py`)
    *   Crops individual nucleus images based on segmentation masks.
4.  **Step 3: Local Feature Extraction** (`step3_batch_mobilevit.py`)
    *   Uses MobileViT to extract local visual features from nucleus crops.
5.  **Step 4: Global Feature Extraction** (`step4_batch_swin.py`)
    *   Uses Swin Transformer to extract global context features from whole slide images (or large patches).
6.  **Step 5: Feature Fusion** (`step5_batch_coattention.py`)
    *   Applies a Co-Attention mechanism to fuse MobileViT and Swin features.
7.  **Step 6: Morphological Features** (`step6_batch_morphological.py`)
    *   Calculates geometric features (area, perimeter, eccentricity, etc.).
8.  **Step 7: Graph Features** (`step7_gat_integrated.py`)
    *   Constructs a cell graph and extracts features using GAT.
9.  **Step 8: Ring Features** (`step8_batch_ring.py`)
    *   Extracts intensity patterns around the nuclear boundary.
10. **Step 9: Centroid Matching** (`step9_train_centroid_matcher.py`)
    *   Matches predicted centroids with ground truth centroids to assign labels.
11. **Training** (`train_nucleus_classifier_true.py`)
    *   Trains the final MLP classifier using the fused feature set.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/yourusername/CellMaskNet.git
    cd CellMaskNet
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the scripts in numerical order. Ensure you have configured the paths in each script to point to your PanNuke dataset location.

```bash
# 1. Preprocess Data
python step0_data_preprocessor.py

# 2. Run Segmentation
python step1_hovernet_batch.py

# ... (Run steps 2 through 9)

# 10. Train Classifier
python train_nucleus_classifier_true.py
```

## Requirements

*   Python 3.10+
*   PyTorch
*   Torchvision
*   NumPy
*   Pandas
*   OpenCV (opencv-python)
*   Scikit-learn
*   Scikit-image
*   Transformers (Hugging Face)
*   Timm
*   Matplotlib
*   Seaborn
*   Tqdm

## License

[MIT License](LICENSE)
