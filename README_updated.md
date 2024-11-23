
# Bitcoin Price Forecasting using Machine Learning

This repository contains the code and datasets for the project "Bitcoin Price Forecasting using Machine Learning." The project focuses on leveraging high-dimensional features and machine learning techniques to classify Bitcoin price movements. The primary model used is an Artificial Neural Network (ANN) trained on PCA-reduced data.

## Objective

The main objective of this project is to develop an optimized ANN-based classification model for Bitcoin price forecasting by reducing feature dimensionality using Principal Component Analysis (PCA). This approach addresses the challenges posed by high-dimensional datasets and enhances model efficiency and accuracy.

## Dataset

We used the PCA-transformed dataset for classification:
- **Dataset Name**: `pca_95_cls.csv`
- **Features**: Principal components capturing 95% variance of the original dataset.
- **Source**: Extracted from raw Bitcoin trading data, preprocessed using PCA.

## Methodology

1. **Feature Reduction**:
   - PCA was applied to the high-dimensional dataset to reduce noise and computational complexity while retaining key information.
   
2. **Model Training**:
   - Two versions of the ANN model were implemented:
     - **Original ANN Model**: Simpler architecture without advanced optimization techniques.
     - **Updated ANN Model**: Optimized with PCA-transformed data and training enhancements, including `EarlyStopping` and `ModelCheckpoint`.
   - **Model Architecture**:
     - Input layer: Accepts PCA-transformed features.
     - Hidden layers: Includes Batch Normalization and Dropout for regularization.
     - Output layer: Classification of Bitcoin price movements.

3. **Performance Evaluation**:
   - Metrics: Accuracy, F1-Score, Precision, Recall, and Classification Report.

## Results

### Original ANN Model (Training_ANN_cls.ipynb)
- **Precision**: 0.24 (macro avg)
- **Recall**: 0.50 (macro avg)
- **F1-Score**: 0.33 (macro avg)
- **Accuracy**: 48%

The original model showed limited performance due to the lack of feature reduction and optimization techniques.

### Updated ANN Model (Training_ANN_cls_updated.ipynb)
- **Precision**: 0.55 (macro avg)
- **Recall**: 0.54 (macro avg)
- **F1-Score**: 0.54 (macro avg)
- **Accuracy**: 55%

The updated model, trained with PCA-reduced features and robust optimization, showed improved performance metrics compared to the original model.

## Project Files

- `Training_ANN_cls_updated.ipynb`: Updated notebook with PCA and optimized ANN training.
- `Training_ANN_cls.ipynb`: Original notebook with a simpler ANN training pipeline.
- `pca_95_cls.csv`: PCA-reduced dataset used for classification.
- `Results_ANN.ipynb`: Analysis and visualization of model performance metrics.
- `README.md`: Project documentation (this file).

## Usage Instructions

### Prerequisites
1. Python 3.8 or higher
2. Required libraries:
   - TensorFlow/Keras
   - Scikit-learn
   - Pandas
   - NumPy
3. Install dependencies using:
   ```bash
   pip install -r requirements.txt
   ```

### Steps to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/satish-matani/AIDI_1002_01_Final_Project.git
   ```
2. Navigate to the project directory:
   ```bash
   cd AIDI_1002_01_Final_Project
   ```
3. Open the updated notebook:
   ```bash
   jupyter notebook Training_ANN_cls_updated.ipynb
   ```
4. Execute the cells sequentially to train and evaluate the ANN model.

## Contributors

- Abhay Pal
- Vaibhav Patel
- Satish Matani

## Conclusion

This project demonstrates the effectiveness of PCA in reducing dataset dimensionality and optimizing ANN performance for Bitcoin price classification. The updated ANN model outperformed the original version by incorporating PCA-transformed data and training optimizations. Future work could explore integrating alternative deep learning models such as LSTM or hybrid architectures for enhanced prediction accuracy.
