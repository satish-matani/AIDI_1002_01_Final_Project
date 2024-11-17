# BTC Data Analysis and ANN Enhancement Project

## Project Description
This project is based on the research paper titled ["An Efficient Machine Learning Approach for Bitcoin Price Prediction"](https://link.springer.com/article/10.1007/s00521-020-05129-6). The goal of this project is to replicate the methodology presented in the paper, enhance its implementation by making significant contributions, and evaluate its effectiveness on new datasets. The research focuses on using Artificial Neural Networks (ANN) for Bitcoin price prediction, providing a robust framework for analyzing volatile financial data.

The project aligns with the assignment requirements by:
1. Reproducing the results of the selected research paper.
2. Introducing significant updates to the model methodology.
3. Evaluating the updated approach on new datasets.

---

## Research Paper Overview
### Title: [An Efficient Machine Learning Approach for Bitcoin Price Prediction](https://link.springer.com/article/10.1007/s00521-020-05129-6)
The research paper proposes a methodology for Bitcoin price prediction using ANN. The key steps involved are:
1. **Data Preprocessing**: Cleaning and normalizing Bitcoin price datasets.
2. **Feature Selection**: Choosing relevant features affecting Bitcoin prices.
3. **Model Implementation**: Training an ANN model with selected features.
4. **Evaluation**: Assessing the model's performance using metrics like Mean Squared Error (MSE).

---

## Project Enhancements
### Updates in `Training_ANN_cls_updated.ipynb`
The following enhancements were made to the original methodology:
1. **Additional Dataset Integration**: Introduced new datasets from [GitHub Dataset Link](https://github.com/heliphix/btc_data/tree/paper_datasets) to test the model's robustness in different contexts.
2. **Hyperparameter Tuning**: Experimented with different values for learning rate, batch size, and number of epochs to optimize the ANN performance.
3. **Enhanced Architecture**: Added dropout layers to reduce overfitting and adjusted the number of neurons in hidden layers for better performance.
4. **Evaluation Metrics**: Incorporated additional metrics like Mean Absolute Error (MAE) and R-Squared for comprehensive performance evaluation.
5. **Visualization**: Added detailed visualizations of training loss, accuracy, and prediction trends for better interpretability.

### How the Updates Satisfy Assignment Instructions
- **Replication**: The original methodology was implemented as described in the research paper, ensuring reproducibility of results.
- **Significant Contribution**: The updates introduce methodological improvements, exploring the impact of advanced techniques on the model’s accuracy and generalizability.
- **Testing on New Datasets**: The ANN model was evaluated on datasets different from the original study, fulfilling the requirement to test effectiveness in new contexts.

---

## How to Run the Project
### Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/heliphix/btc_data.git
   cd btc_data
   ```
2. Switch to the appropriate branch containing the project code:
   ```bash
   git checkout main
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Jupyter Notebook:
   ```bash
   jupyter notebook Training_ANN_cls_updated.ipynb
   ```

### Dataset
The datasets are available in the [GitHub Dataset Link](https://github.com/heliphix/btc_data/tree/paper_datasets). The code automatically downloads and preprocesses the datasets.

### Results
- The updated ANN model achieved improved performance metrics compared to the baseline model described in the research paper.
- Visualizations of training and testing phases demonstrate the stability and accuracy of the model.

---

## Project Members
1. **Satish Kumar**
   - **Student ID**: 200574904

2. **Vandan Ashok Patel**
   - **Student ID**: 200623684

---

## Adding Contributors
To add a contributor to your GitHub repository, follow these steps:
1. Go to your repository on GitHub.
2. Navigate to **Settings** > **Manage Access**.
3. Click "Invite a Collaborator."
4. Enter the GitHub username of the contributor (e.g., `Vandan Patel`).
5. Assign the appropriate permissions (e.g., Write access).

---

## Additional Notes
1. **Documentation**: Comprehensive documentation is provided in the notebook for all changes made.
2. **Significance of Updates**: The project not only improves the existing methodology but also explores its adaptability to different datasets, adding significant academic and practical value.
3. **Professor Instructions**: The GitHub repository is public and can be accessed at [BTC Data Project Repository](https://github.com/heliphix/btc_data/tree/main).

---

Feel free to explore the code and let us know if there are any questions!
