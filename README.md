Credit Card Fraud Detection: Machine Learning vs Deep Learning

Overview

This project explores credit card fraud detection using traditional Machine Learning and Deep Learning techniques. The main objective is to compare different algorithms and neural network architectures in identifying fraudulent transactions within a highly imbalanced dataset.
The project evaluates traditional Machine Learning models alongside two Deep Learning architectures: a Multi-Layer Perceptron (MLP) and a Convolutional Neural Network (CNN), both implemented using PyTorch.

Dataset:

Source: Kaggle – Credit Card Fraud Detection

Total transactions: 284,807

Fraudulent transactions: 492

Legitimate transactions: 284,315

Features: 30 numerical features

Target: Class (0 = Legitimate, 1 = Fraudulent)

Technologies & Tools:

Python

Pandas & NumPy

Scikit-learn

XGBoost

PyTorch

Matplotlib & Seaborn

Models Implemented

Traditional Machine Learning:

Logistic Regression

Random Forest

XGBoost

Deep Learning:

Multi-Layer Perceptron (MLP) – implemented using PyTorch.

Convolutional Neural Network (CNN) – implemented using PyTorch.

Methodology:

The project follows several stages:
Data Preprocessing: Data preparation, feature handling and preprocessing.

Model Development: Implementation of traditional Machine Learning algorithms and two Deep Learning architectures (MLP and CNN).

Imbalance Handling: Use of class weighting and weighted Binary Cross-Entropy loss to address the imbalanced dataset.

Model Optimization: Neural network training using the Adam optimizer, early stopping and validation-based threshold selection.

Model Evaluation: Comparison of model performance using metrics suitable for imbalanced classification.


Evaluation Metrics:

The models are evaluated using:

Precision

Recall

F1-Score

Precision-Recall AUC (PR-AUC)

Confusion Matrix

These metrics provide a more meaningful evaluation of fraud detection performance than accuracy alone.

Results:

The comparison evaluates the effectiveness of traditional Machine Learning and Deep Learning approaches.

XGBoost F1-Score: 86.13%

PyTorch MLP F1-Score: 80.00%

XGBoost achieved an F1-score of 86.13%, while the PyTorch MLP achieved 80.00%. The CNN results are included in the overall model comparison and should be reported using the corresponding experimental results.


Installation

Clone the repository:

git clone YOUR_REPOSITORY_URL
cd Credit-Card-Fraud-Detection

Install dependencies:

pip install -r requirements.txt

Download the dataset from Kaggle and place it in the appropriate data directory before running the project.

Key Takeaways:

Compared traditional Machine Learning algorithms with Deep Learning architectures.

Implemented MLP and CNN models using PyTorch.

Explored strategies for handling highly imbalanced datasets.

Applied model optimization and validation-based threshold selection.

Evaluated model performance using metrics appropriate for fraud detection.

Analyzed the strengths and limitations of different classification approaches.
