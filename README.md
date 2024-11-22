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

## How to Run the Project

### Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/heliphix/btc_data.git
   cd btc_data
   ```
2. Switch to the appropriate branch:
   ```bash
   git checkout main
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the updated Jupyter Notebook:
   ```bash
   jupyter notebook Training_ANN_cls_Group_Work_Updated.ipynb
   ```

### Dataset
- Original dataset and additional datasets are available in the [GitHub Dataset Link](https://github.com/heliphix/btc_data/tree/paper_datasets).
- Preprocessing is automated in the code.

---

## Comparison Table

| Feature                 | Original Code                          | Updated Code                          |
|-------------------------|----------------------------------------|----------------------------------------|
| Architecture            | Dense(400-500-100)                    | Dense(512-256-128) + Dropout + BatchNorm |
| Learning Rate           | Static with step reductions            | Dynamic scheduling                    |
| Evaluation Metrics      | Accuracy                              | Accuracy, MAE, R-Squared               |
| Validation Accuracy     | ~45.76%                               | ~72.88%                                |
| Overfitting Prevention  | None                                  | Dropout + Batch Normalization          |
| Dataset Robustness      | Single Dataset                        | Multiple Datasets                      |

---

## Project Members
1. **Satish Kumar**
   - **Student ID**: 200574904

2. **Vandan Ashok Patel**
   - **Student ID**: 200623684

---
