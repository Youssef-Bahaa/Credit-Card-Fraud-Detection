# Credit Card Fraud Detection

Detect fraudulent credit card transactions using machine learning.

## Project Structure

```
Credit-Card-Fraud-Detection/
├── Credit-Card-Fraud-Detection-EDA.ipynb    # Data exploration and visualization
├── credit_fraud_train.py                    # Train and test ML models
├── credit_fraud_utils_data.py               # Data preprocessing functions
├── credit_fraud_utils_eval.py               # Evaluation metrics and plots
├── credit_fraud_utils_utilities.py          # Helper functions
├── data/                                   # Dataset files
├── models/                                 # Trained models
├── results/                                # Metrics, logs, and figures
├── config/                                 # Config files
├── requirements.txt                        # Python dependencies
```

## Setup

1. Clone the repo:
    ```bash
    git clone https://github.com/Youssef-Bahaa/Credit-Card-Fraud-Detection.git
    cd Credit-Card-Fraud-Detection
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Data

- Download the [Kaggle dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud).
- Put `creditcard.csv` in the `data/` folder.

## Usage

- Explore data:
    ```bash
    jupyter notebook Credit-Card-Fraud-Detection-EDA.ipynb
    ```
- Train models:
    ```bash
    python credit_fraud_train.py
    ```

Results and figures appear in `results/`.

## Acknowledgements

- Kaggle Credit Card Fraud Dataset
- scikit-learn, pandas, numpy, matplotlib
