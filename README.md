# Bitcoin Price Prediction using ANN: Analysis and Enhancements

## Project Description
This project is based on the research paper titled ["An Efficient Machine Learning Approach for Bitcoin Price Prediction"](https://link.springer.com/article/10.1007/s00521-020-05129-6). The goal is to replicate the methodology described in the paper, enhance its implementation with significant contributions, and evaluate its effectiveness on new datasets. The research focuses on Artificial Neural Networks (ANN) for Bitcoin price prediction, leveraging advanced machine learning techniques to analyze volatile financial data.

---

## Research Paper Overview
### Title: [An Efficient Machine Learning Approach for Bitcoin Price Prediction](https://link.springer.com/article/10.1007/s00521-020-05129-6)

The research paper proposes the following methodology:
1. **Data Preprocessing**: Cleaning and normalizing Bitcoin price datasets.
2. **Feature Selection**: Selecting relevant features affecting Bitcoin prices.
3. **Model Implementation**: Training an ANN model with the selected features.
4. **Evaluation**: Assessing model performance using Mean Squared Error (MSE).

---

## Replication of Original Code

### Original Model Highlights
The original code implemented a basic ANN architecture:
- **Architecture**: Three Dense layers with 400, 500, and 100 neurons, respectively, followed by a final output layer.
- **Activation Function**: ReLU for hidden layers, Sigmoid for the output layer.
- **Learning Rate**: Scheduled reduction based on epoch milestones.
- **Metrics**: Binary Crossentropy loss and Accuracy.
- **Results**: The model achieved a validation accuracy plateau of ~45.76% after 50 epochs.

### Issues with Original Implementation
- **Overfitting**: Lack of dropout layers and batch normalization.
- **Performance**: Low accuracy, indicating potential data imbalance or architectural limitations.
- **Evaluation Metrics**: Limited to accuracy, without additional performance indicators.

---

## Enhancements in Updated Code

### Updates Introduced
The updated implementation improves on the original methodology with the following changes in Training_ANN_cls_Group_Work_Updated.ipynb file:
1. **Architectural Enhancements**:
   - Introduced **Batch Normalization** to stabilize training.
   - Added **Dropout layers** to reduce overfitting.
   - Optimized **hidden layers** with 512, 256, and 128 neurons.
2. **Hyperparameter Tuning**:
   - Experimented with learning rates, batch sizes, and epochs.
   - Used dynamic learning rate scheduling.
3. **Evaluation Metrics**:
   - Added Mean Absolute Error (MAE) and R-Squared.
   - Visualized metrics for better interpretability.
4. **Visualization**:
   - Graphs for training loss, accuracy trends, and prediction errors.

### Updated Model Results
- **Validation Accuracy**: Improved to ~72.88% after 50 epochs.
- **Metrics Comparison**:
   - MAE: Significantly reduced compared to the original model.
   - R-Squared: Positive values indicating better fit.

### Contribution Significance
These enhancements align with the assignment requirements by:
- Introducing methodological improvements to address overfitting and stabilization issues.
- Providing a comprehensive evaluation framework with multiple performance metrics.

---

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

---

## Project Members
1. **Satish Kumar**
   - **Student ID**: 200574904

2. **Vandan Ashok Patel**
   - **Student ID**: 200623684

---

## Conclusion

This project demonstrates the effectiveness of PCA in reducing dataset dimensionality and optimizing ANN performance for Bitcoin price classification. The updated ANN model outperformed the original version by incorporating PCA-transformed data and training optimizations. Future work could explore integrating alternative deep learning models such as LSTM or hybrid architectures for enhanced prediction accuracy.

---
