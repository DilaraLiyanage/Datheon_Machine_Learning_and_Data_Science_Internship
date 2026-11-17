# Task 02 — Real Estate Housing Price Prediction

## 1. Project Overview

This project focuses on predicting housing prices using regression techniques and the California Housing dataset.

The main objective is to implement and compare Ordinary Least Squares (OLS), Ridge Regression (L2 regularization), and Lasso Regression (L1 regularization) to understand how regularization affects model performance.

The task includes data preprocessing, missing-value handling, feature engineering, multicollinearity analysis, hyperparameter tuning, model evaluation, and residual error analysis.

## 2. Dataset Description

**Dataset:** California Housing Dataset

**Dataset Source:** [Housing Dataset — Hands-On Machine Learning](https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv)

The dataset contains housing information from California districts, including geographical, demographic, and household-related attributes.

The target variable is `median_house_value`, representing the median house value for a district in US dollars.

### Dataset Features

| Feature | Description |
|---|---|
| longitude | Longitude of the district |
| latitude | Latitude of the district |
| housing_median_age | Median age of houses in the district |
| total_rooms | Total number of rooms |
| total_bedrooms | Total number of bedrooms |
| population | Population of the district |
| households | Number of households |
| median_income | Median income in the district, expressed in units of $10,000 |
| ocean_proximity | Categorical attribute describing the district's proximity to the ocean |
| median_house_value | Target variable — median house value in USD |

The dataset includes numerical and categorical attributes, making it suitable for experimenting with preprocessing pipelines and regularized regression methods.

## 3. Project Objectives

The main objectives of this task are to:

1. Load and inspect the California Housing dataset using Pandas.
2. Identify missing values, data types, and feature distributions.
3. Perform exploratory data analysis using statistical summaries and visualizations.
4. Examine correlations and potential multicollinearity between numerical features.
5. Handle missing numerical values using appropriate imputation techniques.
6. Encode categorical variables and standardize numerical features.
7. Engineer additional features that may improve predictive performance.
8. Implement Ordinary Least Squares, Ridge, and Lasso regression models.
9. Tune regularization parameters using GridSearchCV.
10. Compare models using RMSE, MAE, and R² scores.
11. Analyze residual errors and compare predicted values against actual house prices.

## 4. Tools and Technologies

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn
- Jupyter Notebook
- Visual Studio Code

## 5. Planned Methodology

### 5.1 Data Loading and Initial Inspection

The dataset will be loaded using Pandas.

Initial analysis will include:

- Dataset dimensions and feature data types
- Descriptive statistics
- Missing-value identification
- Duplicate-record checks
- Distribution of the target variable

### 5.2 Exploratory Data Analysis

Exploratory data analysis will be used to understand how housing characteristics relate to median house values.

The planned visualizations include:

- Histograms of numerical attributes
- Correlation heatmap
- Scatter plots comparing selected features and housing prices
- Geographical distribution of housing values
- Target variable distribution

### 5.3 Data Preprocessing

The dataset will be divided into training and testing subsets before fitting any preprocessing transformations.

The preprocessing pipeline will include:

**Missing-Value Imputation**

Missing numerical values will be handled using SimpleImputer. Median imputation will be considered because it is less affected by extreme observations than mean imputation.

**Feature Scaling**

StandardScaler will be applied to numerical features, particularly because Ridge and Lasso regularization are sensitive to differences in feature scales.

**Categorical Encoding**

The `ocean_proximity` feature will be transformed into numerical form using OneHotEncoder.

**Data Leakage Prevention**

Imputation, scaling, and encoding operations will be fitted only on the training data. Scikit-Learn pipelines and ColumnTransformer will be used to apply the same learned transformations consistently to validation and testing data.

### 5.4 Feature Engineering

Additional features will be investigated to represent relationships between existing housing attributes.

Potential features include:

- `rooms_per_household = total_rooms / households`
- `bedrooms_per_room = total_bedrooms / total_rooms`
- `population_per_household = population / households`

These features may capture housing density and household characteristics more effectively than the original attributes alone.

Division-by-zero cases and missing values will be handled appropriately.

Feature engineering will be performed within the training workflow to avoid introducing data leakage.

### 5.5 Multicollinearity Analysis

Correlation analysis will be used to identify strongly related numerical variables.

Highly correlated predictors may make coefficient estimates unstable in ordinary linear regression.

Ridge Regression will be investigated for its ability to reduce coefficient magnitudes, while Lasso Regression will be examined for its ability to shrink some coefficients to zero.

### 5.6 Regression Model Training

Three regression models will be implemented and compared.

**Ordinary Least Squares (OLS)**

LinearRegression will be used as the baseline model.

It estimates regression coefficients by minimizing the sum of squared prediction errors.

**Ridge Regression (L2 Regularization)**

Ridge Regression adds an L2 penalty to the ordinary least squares objective.

It reduces large coefficient values and can help stabilize models when predictors are highly correlated.

**Lasso Regression (L1 Regularization)**

Lasso Regression adds an L1 penalty to the regression objective.

It can reduce some feature coefficients to zero, allowing it to perform a form of feature selection.

### 5.7 Hyperparameter Optimization

GridSearchCV will be used to evaluate multiple values of the regularization parameter `alpha` for Ridge and Lasso.

Cross-validation will be performed using the training data.

The best hyperparameters will be selected based on cross-validation performance, without using the test set for model selection.

### 5.8 Model Evaluation

The models will be evaluated using three regression metrics.

| Metric | Description |
|---|---|
| RMSE | Root Mean Squared Error; gives greater weight to larger prediction errors |
| MAE | Mean Absolute Error; measures average absolute prediction error |
| R² | Coefficient of determination; measures how well the model explains variation in the target |

Lower RMSE and MAE values indicate better predictive accuracy.

Higher R² values generally indicate better explanatory performance, with 1 representing a perfect fit.

The test set will be reserved for final evaluation after preprocessing and hyperparameter tuning decisions have been completed.

### 5.9 Residual Error Analysis

Residuals will be calculated as:

`Residual = Actual Value - Predicted Value`

The following visualizations will be generated:

- Actual vs. predicted house prices
- Residuals vs. predicted values
- Residual distribution histogram
- Regression model comparison chart

Residual analysis will help identify systematic prediction errors, unusual observations, and limitations of the regression models.

## 6. Expected Deliverables

The completed task will include:

- Jupyter Notebook containing the complete regression pipeline
- Exploratory data analysis and correlation visualizations
- Feature engineering and preprocessing steps
- Ordinary Least Squares, Ridge, and Lasso models
- Hyperparameter tuning results
- A comparison of RMSE, MAE, and R² scores
- Predicted vs. actual housing price plots
- Residual error distribution plots
- Discussion of model performance and limitations

## 7. Results and Discussion

This section will be updated after training and evaluating the regression models.

The final results will include a comparison table containing the evaluation metrics for all three models.

The discussion will examine the effects of regularization, feature engineering, and hyperparameter tuning on predictive performance.

Any limitations observed during the experiments will also be documented.

## 8. How to Run

After the Python environment and project dependencies are configured:

1. Open `notebooks/housing_regression.ipynb` in VS Code or Jupyter Notebook.
2. Select the configured Python environment as the notebook kernel.
3. Ensure the housing dataset is available in the `data/` directory.
4. Run the notebook cells in order.
5. Review the model comparison metrics and generated plots.

## 9. Project Status

**Status:** Documentation and environment setup in progress.

Model implementation, hyperparameter optimization, and evaluation have not yet been completed.

## 10. Author

**Dilara Liyanage**

BSc (Hons) Computer Science Undergraduate  
Sri Lanka Institute of Information Technology (SLIIT)

**Internship:** Datheon — Machine Learning and Data Science