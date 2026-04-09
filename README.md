# Premature-Term-Birth-Classification
Premature vs. Term Birth Classification using PyTorch
This repository contains a deep learning pipeline designed to classify infants as Premature or Term based on brain MRI features. It utilizes PyTorch to build a Multi-Layer Perceptron (MLP) for binary classification


**Key components:**
* **Data Processing**: Loading and preprocessing tabular MRI feature data (`.pkl` format).
* **Neural Network**: A custom 4-layer MLP built in PyTorch with ReLU. Additionally Dropout layers added to prevent overfitting.
* **3D Image Handling**: A `BrainSegmentationDataset` class for loading raw NIfTI (`.nii.gz`) brain volumes and segmentation masks.
* **Training Pipeline**: Full training loop with validation, accuracy logging, and loss visualization.

**Architecture Details:**
* **Data Cleaning**: Utilizes `VarianceThreshold` to automatically detect and remove constant/zero-variance features prior to training.
* **Feature Selection**: Due to the small dataset with much more features than samples (300 feautres vs 80 samples) `SelectKBest` (ANOVA F-value) is used to isolate the 40 most highly correlated features.
* **Input Layer**: Accepts **40 input features**
* **Hidden Layers**: Four dense layers,
    * Activation Function: **ReLU** (Rectified Linear Unit) is applied after each hidden layer to introduce non-linearity.
* **Output Layer**: A single neuron (1 unit).
* **Loss Function**: Using `BCEWithLogitsLoss` which combines sigmoid with BCE loss function preventing vanishing gradient
* **Optimiser** Adam Optimiser used which adapts the learning rates for each parameter and using momentum to speed up convergence.
