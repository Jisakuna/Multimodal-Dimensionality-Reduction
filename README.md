# Multimodal Dimensionality Reduction for Welding

A Python-based pipeline for processing, analyzing, and visualizing multimodal data collected from robotic welding operations.

This project extracts features from multiple data sources, including audio, images, video, and sensor signals. The extracted features are combined into a unified multimodal representation and reduced using PCA, t-SNE, and UMAP.

The pipeline supports exploratory data analysis, feature-space visualization, pattern discovery, and comparison of dimensionality-reduction methods for complex welding datasets.

## Overview

Robotic welding systems can generate large amounts of heterogeneous data from different sensors and monitoring devices. Analyzing these data sources independently may overlook important relationships between modalities.

This project provides a modular workflow that:

1. Processes each data modality independently.
2. Extracts representative features.
3. Combines the extracted features into multimodal vectors.
4. Reduces the dimensionality of the fused feature space.
5. Evaluates and visualizes the resulting representations.

## Supported Data Modalities

The pipeline supports feature processing for:

* Audio recordings
* Welding images
* Welding videos
* Sensor measurements

## Dimensionality-Reduction Methods

### Principal Component Analysis

PCA is a linear dimensionality-reduction method that transforms high-dimensional feature vectors into a smaller set of principal components while preserving as much variance as possible.

PCA is useful for:

* Data compression
* Noise reduction
* Feature-space analysis
* Preparing data for machine-learning models

### t-Distributed Stochastic Neighbor Embedding

t-SNE is a nonlinear dimensionality-reduction method designed primarily for visualizing high-dimensional data.

t-SNE is useful for:

* Cluster visualization
* Local pattern discovery
* Exploring relationships between welding samples
* Identifying potentially separable operating conditions

### Uniform Manifold Approximation and Projection

UMAP is a nonlinear manifold-learning technique that preserves both local and global structures more effectively than many traditional visualization methods.

UMAP is useful for:

* Fast high-dimensional visualization
* Cluster analysis
* Pattern discovery
* Preserving meaningful relationships between samples

## Key Features

* Modular preprocessing scripts for each modality
* Audio feature extraction
* Image feature extraction
* Video feature extraction
* Sensor-data processing
* Multimodal feature fusion
* PCA dimensionality reduction
* t-SNE dimensionality reduction
* UMAP dimensionality reduction
* Feature-space evaluation
* Two-dimensional and three-dimensional visualization
* NumPy-based feature storage
* Project-relative file paths
* Reusable processing workflow

## Project Workflow

```text
Raw Welding Dataset
        |
        v
Modality-Specific Processing
        |
        +--> Audio Features
        |
        +--> Image Features
        |
        +--> Video Features
        |
        +--> Sensor Features
        |
        v
Multimodal Feature Fusion
        |
        v
High-Dimensional Feature Vectors
        |
        +--> PCA
        |
        +--> t-SNE
        |
        +--> UMAP
        |
        v
Evaluation and Visualization
```

## Project Structure

```text
Multimodal-Dimensionality-Reduction/
├── project_paths.py
├── audio_process.py
├── image_process.py
├── viedo_process.py
├── sensor_process.py
├── compose.py
├── pca.py
├── PCA_eva.py
├── t_SNE.py
├── t_SNE_eva.py
├── UMAP.py
├── UMAP_eva.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Clone or download the repository, then install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

A virtual environment is recommended:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Dataset Setup

Place the welding dataset inside the project directory:

```text
Multimodal-Dimensionality-Reduction/
└── intel_robotic_welding_dataset/
```

The dataset directory is excluded from Git through `.gitignore` because datasets may be large or contain private research data.

Project paths are configured in `project_paths.py`, allowing the scripts to work without hard-coded absolute Windows paths.

## Usage

Run the modality-processing scripts first:

```bash
python audio_process.py
python image_process.py
python viedo_process.py
python sensor_process.py
```

Combine the extracted features:

```bash
python compose.py
```

Apply PCA:

```bash
python pca.py
python PCA_eva.py
```

Apply t-SNE:

```bash
python t_SNE.py
python t_SNE_eva.py
```

Apply UMAP:

```bash
python UMAP.py
python UMAP_eva.py
```

The exact execution order may depend on the input and output dependencies defined in the scripts.

## Generated Outputs

The pipeline may generate directories such as:

```text
Audio_process_output/
Image_process_output/
Video_process_output/
Sensor_process_output/
Fused_multimodal_vectors/
Fused_vectors_pca_32d/
Fused_vectors_tsne_32d/
Fused_vectors_umap_32d/
```

These generated directories are excluded from Git by default.

Common generated file formats include:

* `.npy`
* `.npz`
* `.png`
* `.jpg`
* Serialized model or feature files

## Potential Applications

This project can support research and development in:

* Robotic welding monitoring
* Welding-quality analysis
* Manufacturing-process visualization
* Multimodal machine learning
* Fault and anomaly detection
* Process-state classification
* Sensor-fusion research
* Industrial artificial intelligence
* Predictive quality assessment

## Requirements

The project may use the following Python libraries:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
torch
torchvision
opencv-python
Pillow
librosa
tqdm
umap-learn
```

Refer to `requirements.txt` for the complete dependency list.

## Data and Privacy

The included `.gitignore` excludes common datasets, generated outputs, virtual environments, model files, and environment-variable files.

## License

This project is licensed under the MIT License.

## Acknowledgments

This project was developed to support multimodal analysis and dimensionality-reduction research for robotic welding data.
