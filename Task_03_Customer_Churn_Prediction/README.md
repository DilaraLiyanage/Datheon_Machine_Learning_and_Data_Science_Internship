# Task 03 — Predictive Customer Churn Classification

## 1. Project Overview

This project focuses on predicting customer churn using machine learning techniques and telecommunications subscription data.

Customer churn occurs when an existing customer stops using a company's services. Predicting customers who are likely to leave can help businesses identify potential retention risks and take appropriate action.

The main objective of this task is to develop and compare Random Forest and XGBoost classification models using the IBM Telco Customer Churn dataset.

The project includes data preprocessing, categorical feature encoding, class imbalance handling, model training, cross-validation, performance evaluation, and feature importance analysis.

## 2. Dataset Description

**Dataset:** IBM Telco Customer Churn

**Dataset source:** [IBM Telco Customer Churn CSV](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)

The dataset contains customer information related to telecommunications services, including account details, subscription services, contract types, and payment information.

The target variable is `Churn`, which indicates whether a customer has discontinued the service.

### Main Dataset Attributes

| Attribute | Description |
|---|---|
| customerID | Unique identifier for each customer |
| gender | Customer's recorded gender |
| SeniorCitizen | Indicates whether the customer is classified as a senior citizen |
| Partner | Indicates whether the customer has a partner |
| Dependents | Indicates whether the customer has dependents |
| tenure | Number of months the customer has remained with the company |
| PhoneService | Indicates whether the customer subscribes to phone services |
| InternetService | Type of internet service used |
| OnlineSecurity | Indicates whether online security service is subscribed to |
| TechSupport | Indicates whether technical support is subscribed to |
| Contract | Customer's contract type |
| PaymentMethod | Customer's payment method |
| MonthlyCharges | Monthly subscription charges |
| TotalCharges | Total charges accumulated by the customer |
| Churn | Target variable indicating whether the customer left |

The dataset contains both numerical and categorical features. Therefore, preprocessing is required before training the classification models.

## 3. Project Objectives

The main objectives of this task are to:

1. Load and inspect the IBM Telco Customer Churn dataset.
2. Identify missing values, inconsistent data types, and class distribution.
3. Perform exploratory data analysis to understand customer characteristics and churn patterns.
4. Convert the `TotalCharges` attribute into a numerical data type and handle invalid or missing values.
5. Encode categorical variables using OneHotEncoder.
6. Analyze and address class imbalance using SMOTE or appropriate class weighting.
7. Train Random Forest and XGBoost classification models.
8. Apply cross-validation to compare model performance.
9. Evaluate the models using ROC-AUC, precision, recall, F1-score, and confusion matrices.
10. Visualize ROC curves and identify the ten most influential features.
11. Compare the models and discuss their strengths, limitations, and possible business applications.

## 4. Tools and Technologies

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- Imbalanced-Learn
- Matplotlib
- Seaborn
- Jupyter Notebook
- Visual Studio Code

## 5. Planned Methodology

### 5.1 Data Loading and Initial Inspection

The dataset will be loaded using Pandas.

Initial inspection will include:

- Dataset dimensions and column names
- Data types and missing values
- Duplicate record checks
- Distribution of the target variable
- Numerical and categorical feature summaries

The `customerID` column will be excluded from model training because it serves as a unique identifier rather than a meaningful customer characteristic.

### 5.2 Exploratory Data Analysis

Exploratory data analysis will be performed to understand relationships between customer attributes and churn.

The planned visualizations include:

- Churn class distribution
- Churn distribution by contract type
- Churn distribution by internet service
- Churn distribution by payment method
- Tenure distribution by churn status
- Monthly charges distribution by churn status
- Correlation heatmap for suitable numerical variables

These visualizations will be used to identify patterns that may be useful for churn prediction.

Any relationships observed will be interpreted as associations rather than evidence of causation.

### 5.3 Data Preprocessing

The preprocessing stage will prepare the dataset for classification.

**Handling Missing Values**

The `TotalCharges` column will be converted into a numeric format.

Values that cannot be converted will be treated as missing and handled using an appropriate imputation strategy.

**Categorical Feature Encoding**

Categorical features will be transformed into numerical representations using OneHotEncoder.

**Target Encoding**

The target variable will be encoded as:

- `0` — Customer did not churn
- `1` — Customer churned

**Train-Test Splitting**

The dataset will be split into training and testing subsets using an 80:20 ratio.

A stratified split will be used to preserve the target class proportions.

Preprocessing transformations will be fitted using the training data to avoid data leakage.

### 5.4 Handling Class Imbalance

Customer churn datasets may contain unequal numbers of customers who churned and customers who remained.

The class distribution will be examined before deciding how to address imbalance.

The following approaches will be investigated:

**SMOTE (Synthetic Minority Over-sampling Technique)**

SMOTE generates synthetic observations for the minority class based on neighboring samples.

If SMOTE is selected, it will be applied only within training folds, after the appropriate preprocessing steps, using an Imbalanced-Learn pipeline. The test data will not be oversampled.

**Class Weighting**

Class weighting assigns different importance to observations depending on their class.

For Random Forest, `class_weight` may be used. For XGBoost, `scale_pos_weight` may be considered.

These methods will be compared using validation performance rather than assuming one technique is always better.

### 5.5 Model Training

Two ensemble classification algorithms will be implemented.

**Random Forest Classifier**

Random Forest combines predictions from multiple decision trees to produce a final classification result.

It uses bootstrap sampling and randomized feature selection to reduce the instability of individual decision trees.

Scikit-Learn's RandomForestClassifier will be used.

**XGBoost Classifier**

XGBoost is a gradient-boosting algorithm that builds decision trees sequentially.

Each new tree is trained to improve the overall model by addressing errors represented in the current objective.

XGBClassifier will be used to implement the XGBoost model.

### 5.6 Cross-Validation and Model Optimization

Stratified cross-validation will be used to evaluate the classifiers.

Model training and imbalance-handling methods will be evaluated within the training data.

Selected hyperparameters may include:

**Random Forest**

- `n_estimators`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`

**XGBoost**

- `n_estimators`
- `max_depth`
- `learning_rate`
- `subsample`
- `scale_pos_weight`, where applicable

Hyperparameter selection will be based on cross-validation results.

The test set will be reserved for final model evaluation.

### 5.7 Model Evaluation

The classifiers will be evaluated using the following performance metrics:

| Metric | Purpose |
|---|---|
| Accuracy | Measures the proportion of correctly classified customers |
| Precision | Measures how many customers predicted to churn actually churned |
| Recall | Measures how many actual churn cases were correctly identified |
| F1-score | Balances precision and recall |
| ROC-AUC | Evaluates the model's ability to rank churn cases above non-churn cases across classification thresholds |
| Confusion Matrix | Shows true positives, true negatives, false positives, and false negatives |

The internship task specifies a target ROC-AUC score above 80%.

This value will be treated as a performance objective rather than an assumed result.

If either model does not achieve the target, the actual results and possible reasons will be documented.

### 5.8 ROC Curve Analysis

Receiver Operating Characteristic (ROC) curves will be generated for both classifiers.

The ROC curve illustrates the relationship between:

- True Positive Rate (TPR)
- False Positive Rate (FPR)

ROC-AUC will be calculated using predicted probabilities for the positive churn class.

The curves will be used to compare the discrimination ability of Random Forest and XGBoost.

### 5.9 Feature Importance Analysis

Feature importance will be examined to identify which input variables contribute most strongly to model predictions.

The top ten influential encoded features will be displayed using a horizontal bar chart.

Model-specific feature importance values will be interpreted carefully because they do not necessarily represent causal relationships.

Where appropriate, permutation importance may also be considered to provide an additional interpretation of feature contributions.

## 6. Expected Deliverables

The completed task will include:

- Jupyter Notebook containing the complete churn prediction pipeline
- Exploratory data analysis visualizations
- Data preprocessing and categorical encoding
- Class imbalance handling
- Random Forest and XGBoost classifiers
- Cross-validation and model comparison results
- Classification reports and confusion matrices
- ROC curves and ROC-AUC scores
- Top ten feature importance visualization
- Discussion of findings and model limitations

## 7. Results and Discussion

This section will be updated after completing model training and evaluation.

The final comparison will include:

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest | Pending | Pending | Pending | Pending | Pending |
| XGBoost | Pending | Pending | Pending | Pending | Pending |

The discussion will focus on model performance, the effects of class imbalance handling, and the trade-off between identifying churn cases and producing false alarms.

Possible business applications and limitations of the predictions will also be considered.

## 8. How to Run

After the development environment and required dependencies have been configured:

1. Open `notebooks/churn_prediction.ipynb` in VS Code or Jupyter Notebook.
2. Select the configured Python environment as the notebook kernel.
3. Ensure the Telco Customer Churn dataset is available in the `data/` directory.
4. Run the notebook cells in order.
5. Review the classification metrics, ROC curves, and feature importance plots.

## 9. Project Status

**Status:** Documentation and environment setup in progress.

Model implementation, imbalance handling, optimization, and evaluation have not yet been completed.

## 10. Author

**Dilara Liyanage**

BSc (Hons) Computer Science Undergraduate  
Sri Lanka Institute of Information Technology (SLIIT)

**Internship:** Datheon — Machine Learning and Data Science