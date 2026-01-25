# Premature-Term-Birth-Classification
Premature vs. Term Birth Classification using PyTorch
This repository contains a deep learning pipeline designed to classify infants as Premature or Term based on brain MRI features. It utilizes PyTorch to build a Multi-Layer Perceptron (MLP) for binary classification


**Key components:**
* **Data Processing**: Loading and preprocessing tabular MRI feature data (`.pkl` format).
* **Neural Network**: A custom 3-layer MLP built in PyTorch.
* **3D Image Handling**: A `BrainSegmentationDataset` class for loading raw NIfTI (`.nii.gz`) brain volumes and segmentation masks.
* **Training Pipeline**: Full training loop with validation, accuracy logging, and loss visualization.

**Architecture Details:**
* **Input Layer**: Accepts **300 input features** (corresponding to the ROI metrics).
* **Hidden Layers**: Four dense layers, each containing **100 neurons**.
    * Activation Function: **ReLU** (Rectified Linear Unit) is applied after each hidden layer to introduce non-linearity.
* **Output Layer**: A single neuron (1 unit).
* **Final Activation**: **Sigmoid** function squashes the output to a probability range [0, 1] for binary classification (Preterm vs. Term).
