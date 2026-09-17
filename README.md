# Loan Default Prediction Using Machine Learning

## Project Overview

This project focuses on predicting loan default outcomes using borrower and loan-related financial information. The workflow covers data preprocessing, exploratory data analysis (EDA), feature engineering, feature scaling, and the development and comparison of multiple classification models.

The objective is to explore how borrower characteristics such as income, loan amount, credit history, and loan term can be used to identify patterns associated with loan default.

This project demonstrates a practical machine-learning workflow relevant to **credit risk modeling and credit scoring**.

---

## Objectives

* Understand the structure and characteristics of the loan dataset.
* Identify and handle missing values.
* Perform exploratory data analysis on borrower and loan variables.
* Investigate relationships between borrower characteristics and loan defaults.
* Engineer meaningful features for machine-learning models.
* Train and compare multiple classification algorithms.
* Evaluate model performance using classification metrics.
* Identify limitations and potential improvements for credit default prediction.

---

## Dataset

The project uses a loan dataset containing borrower and loan-related information.

### Key Variables

| Feature          | Description                                               |
| ---------------- | --------------------------------------------------------- |
| `income`         | Borrower's income                                         |
| `loan_amount`    | Amount of the loan                                        |
| `credit_history` | Borrower's credit history indicator                       |
| `term`           | Loan repayment term                                       |
| `defaulted`      | Target variable indicating whether the borrower defaulted |

The dataset contains missing values that are handled during the preprocessing stage.

---

## Project Workflow

### 1. Data Loading

The dataset is loaded using Pandas from a CSV file.

### 2. Data Understanding

The following checks are performed:

* Dataset shape
* Column names
* Data types
* Initial records
* Summary statistics

### 3. Data Preprocessing

Missing values are handled using:

* **Median imputation** for numerical variables:

  * `income`
  * `loan_amount`

* **Mode imputation** for the binary categorical variable:

  * `credit_history`

Median imputation was selected for numerical variables because it is less sensitive to extreme values than the mean.

### 4. Exploratory Data Analysis

The project investigates:

* Income distribution
* Loan amount distribution
* Loan term distribution
* Credit history versus default status
* Correlation between numerical variables

### Key EDA Observations

* Income shows slight right skewness, with values concentrated around the middle-income range.
* Loan amounts are concentrated around the central range, with slight right skewness.
* Loan terms are primarily distributed across 36-month and 60-month loans.
* Borrowers with good credit history generally show fewer defaults compared with borrowers with poor credit history.
* Credit history appears to have the strongest linear association with the default variable among the examined numerical features.

### 5. Feature Engineering

The following transformations are performed:

#### Binary Encoding

The loan term is converted into a binary feature:

* `36 months` → `0`
* `60 months` → `1`

New feature:

```python
term_binary
```

#### Log Transformation

Log transformations are applied to:

```python
income
loan_amount
```

New features:

```python
log_income
log_loan_amount
```

These transformations help reduce skewness and improve feature representation.

### 6. Feature Selection

The final features used for modeling are:

```python
features = [
    'log_income',
    'log_loan_amount',
    'credit_history'
]
```

Target variable:

```python
defaulted
```

### 7. Feature Scaling

`StandardScaler` from Scikit-learn is used to standardize the numerical features.

The transformation places the variables on a comparable scale with approximately:

* Mean = 0
* Standard deviation = 1

### 8. Train-Test Split

The dataset is divided into:

* **80% training data**
* **20% testing data**

A fixed `random_state=42` is used to ensure reproducibility.

### 9. Model Development

Three classification algorithms are trained and compared:

#### Logistic Regression

Used as an interpretable baseline classification model.

#### Decision Tree Classifier

Used to capture potentially non-linear relationships between borrower characteristics and default outcomes.

#### Random Forest Classifier

Used as an ensemble learning approach designed to improve robustness and generalization.

---

## Machine Learning Models

```python
models = {
    'LogisticRegression': LogisticRegression(),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'RandomForestClassifier': RandomForestClassifier()
}
```

Each model is trained using a Scikit-learn pipeline and evaluated on the unseen test dataset.

---

## Model Evaluation

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Classification report

In credit risk modeling, **recall for the default class is particularly important**, because failing to identify a borrower who will default can lead to financial losses.

Therefore, accuracy alone may not be sufficient to determine whether a model is suitable for credit risk applications.

---

## Key Findings

Based on the analysis documented in the notebook:

* Credit history appears to be an important variable associated with loan default.
* Logistic Regression provides a relatively interpretable baseline for default prediction.
* Decision Trees and Random Forests provide alternative approaches for capturing more complex relationships.
* The models demonstrate the importance of evaluating classification performance beyond accuracy.
* Recall for the default class remains an important area for improvement.

---

## Limitations

The current project has several limitations:

* The dataset contains a limited number of explanatory variables.
* Class imbalance may affect the model's ability to identify defaulters.
* The current workflow does not include extensive hyperparameter tuning.
* No cross-validation framework has been implemented.
* No probability calibration or credit scorecard development has been performed.
* The current feature scaling is performed before the train-test split, which may introduce data leakage in a production modeling workflow.

---

## Potential Improvements

Future improvements could include:

* Implementing proper preprocessing pipelines fitted only on training data.
* Applying stratified train-test splitting.
* Using cross-validation.
* Performing hyperparameter tuning.
* Addressing class imbalance using:

  * Class weights
  * SMOTE
  * Other resampling techniques
* Adjusting classification thresholds to improve default recall.
* Exploring additional algorithms such as:

  * XGBoost
  * LightGBM
  * Gradient Boosting
* Evaluating ROC-AUC and Precision-Recall AUC.
* Performing feature importance analysis.
* Applying model interpretability techniques such as SHAP.
* Developing a more complete credit risk scorecard.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* SciPy
* Statsmodels
* Scikit-learn
* Jupyter Notebook

---

## Project Structure

```text
loan-default-prediction/
│
├── loan_default_prediction.ipynb
├── loan_data_1248_with_missing.csv
└── README.md
```

---

## Conclusion

This project demonstrates an end-to-end introductory machine-learning workflow for loan default prediction, covering data preprocessing, exploratory analysis, feature engineering, model development, and classification evaluation.

The project provides a foundation for further development toward more advanced **credit risk modeling, borrower default prediction, and credit scoring applications**.

---

## Future Direction

The next stage of development would focus on improving model reliability, handling class imbalance, preventing data leakage, tuning model performance, and incorporating credit risk-specific evaluation techniques.

