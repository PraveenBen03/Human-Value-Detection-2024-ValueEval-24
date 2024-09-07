# SCaLAR NITK at Touché: Comparative Analysis of Machine Learning Models for Human Value Identification

## Overview

Welcome to the repository for the SCaLAR NITK project, presented at Touché Lab during CLEF 2024. This project involves the comparative analysis of machine learning models for identifying human values in textual data. Our research explores various NLP techniques and models to improve the detection of core human values from text.

## Project Structure

- **notebooks/**: Jupyter notebooks containing experiments and results.
- **src/**: Source code for data preprocessing, model training, and evaluation.
- **data/**: Data files including training datasets, test datasets, and preprocessed data.
- **models/**: Saved models and configuration files for different machine learning algorithms.
- **scripts/**: Utility scripts for running experiments and evaluations.
- **results/**: Output files including performance metrics and evaluation results.
- **docs/**: Documentation and references related to the project.

## Installation

To run the code, you need to have Python 3.8 or later installed. You also need to install the required Python packages. You can set up the environment by running:

```bash
pip install -r requirements.txt
```

## Requirements

To run the SCaLAR NITK project, you will need the following:

- **Python 3.8+**
- **PyTorch**: A deep learning framework for training and evaluating models.
- **Transformers**: A library for working with transformer models like BERT and RoBERTa.
- **Scikit-learn**: A machine learning library for classical models and metrics.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: A library for numerical operations.

## Data

The dataset used in this project consists of text data in multiple languages, annotated with human values. The dataset is split into training, validation, and test sets. Additionally, text data is processed into BERT embeddings to serve as input for the models.

### Data Files

- **data/train.csv**: Contains training data with annotations for human values.
- **data/validation.csv**: Used for tuning model hyperparameters.
- **data/test.csv**: Contains data for final model evaluation.
- **data/embeddings/**: Directory containing precomputed BERT embeddings for the dataset.

## Models

The project utilizes various machine learning models and techniques to analyze and identify human values in textual data.

### Classical ML Models

- **Support Vector Machines (SVM)**: A supervised learning model used for classification tasks. SVMs find the hyperplane that best separates the classes in the feature space.
- **K-Nearest Neighbors (KNN)**: A simple, instance-based learning algorithm where the class of a sample is determined by the majority class among its nearest neighbors.
- **Decision Trees**: A model that uses a tree-like graph of decisions and their possible consequences. It splits data into subsets based on the value of input features.

### Transformer Models

- **BERT (Bidirectional Encoder Representations from Transformers)**: A transformer-based model that processes text bidirectionally to understand the context of words in relation to other words in a sentence.
- **RoBERTa (Robustly optimized BERT approach)**: An improved version of BERT that enhances performance by optimizing the training process and removing the Next Sentence Prediction objective.

### Large Language Models

- **Mistral-7B**: A large-scale language model featuring QLoRA (Quantized Low-Rank Adaptation) quantization and Supervised Fine-Tuning (SFT) for efficient and accurate text understanding.

## Training and Evaluation

Models are evaluated using performance metrics such as precision, recall, and F1-score. Training and evaluation scripts are available in the **scripts/** and **notebooks/** directories.

### Training a Model

To train a model, navigate to the **src/** directory and execute:

```bash
python train_model.py --model <model_name> --data_path <data_path>
```

