# ML-Income-Classifier

A Machine Learning classification project that predicts whether an individual's annual income exceeds $50,000 based on demographic and employment features from the UCI Adult dataset. This project demonstrates end-to-end ML workflow, including data cleaning, encoding, feature scaling, model training, and evaluation.

## Features
- **Dataset**: Adult dataset (48,842 records) from UCI Machine Learning Repository.
- **Preprocessing**: Handles missing values, one-hot encoding for categorical features, and StandardScaler for numerical features.
- **Model**: Logistic Regression for binary classification.
- **Evaluation**: Metrics include accuracy, precision, recall, and F1-score (achieves ~85% accuracy).

## Requirements
- Python 3.8+
- Libraries: pandas, scikit-learn, numpy

## How to Run
1. Clone the repository: `git clone https://github.com/Mohd-Shaad/ML-Income-Classifier.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the notebook in Jupyter or [Google Colab](https://colab.research.google.com/your-notebook-link).
4. Dataset loads automatically from UCI URL.

## Sample Output
- Accuracy: 0.8512
- Precision: 0.7521
- Recall: 0.6034
- F1-Score: 0.6695

## Dataset Source
[UCI Adult Dataset](https://archive.ics.uci.edu/dataset/2/adult)

## License
This project is open-source under the MIT License.
