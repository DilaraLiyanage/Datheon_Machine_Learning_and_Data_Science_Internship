# Task 01 — Multi-Class Benchmark Classification

## 1. Project Overview

This project focuses on developing a supervised machine learning pipeline for multi-class classification using the Iris dataset.

The main objective is to implement and compare Logistic Regression and Decision Tree classifiers to understand how different classification algorithms perform on the same dataset.

The task covers the complete machine learning workflow, including exploratory data analysis, feature scaling, dataset splitting, model training, performance evaluation, and visualization.

## 2. Dataset Description

**Dataset:** Iris Flower Dataset

**Dataset source:** [Seaborn Iris Dataset](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv)

The Iris dataset contains 150 flower samples belonging to three species:

- Iris setosa
- Iris versicolor
- Iris virginica

Each sample contains four numerical features:

| Feature | Description |
|---|---|
| sepal_length | Sepal length in centimetres |
| sepal_width | Sepal width in centimetres |
| petal_length | Petal length in centimetres |
| petal_width | Petal width in centimetres |

The target variable is `species`, which represents the flower category.

The dataset is suitable for this task because it has multiple target classes and numerical input features that can be used to compare classification models.

## 3. Project Objectives

The main objectives of this task are to:

1. Load and explore the Iris dataset using Pandas.
2. Examine feature distributions and relationships through exploratory data analysis.
3. Split the dataset into training and testing subsets using an 80:20 ratio.
4. Apply feature standardization using StandardScaler.
5. Train Logistic Regression and Decision Tree classification models.
6. Evaluate both models using accuracy, precision, recall, and F1-score.
7. Visualize model predictions using confusion matrix heatmaps.
8. Compare model performance and discuss the results.
9. Visualize decision boundaries to understand how the models separate different flower classes.

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

### 5.1 Data Loading and Inspection

The dataset will be loaded using Pandas.

Initial inspection will include checking the dataset dimensions, feature types, missing values, duplicate records, and target class distribution.

### 5.2 Exploratory Data Analysis

Exploratory data analysis will be carried out to understand patterns in the dataset.

The planned visualizations include:

- Feature distribution plots
- Pair plots showing relationships between flower measurements
- Correlation heatmap
- Target class distribution

### 5.3 Data Preprocessing

The dataset will be separated into input features and the target variable.

An 80:20 stratified train-test split will be used to preserve the class proportions in both subsets.

StandardScaler will be fitted on the training data and then applied to both training and testing data to prevent data leakage.

### 5.4 Model Training

Two supervised classification algorithms will be implemented.

**Logistic Regression**

Logistic Regression will be used as a linear classification baseline. For this multi-class problem, the classifier will predict the flower species using the input measurements.

**Decision Tree Classifier**

Decision Tree will be used as a non-linear classification model that learns decision rules from the input features.

The models will be trained using the same training and testing split to support a fair comparison.

### 5.5 Model Evaluation

The performance of both classifiers will be evaluated using the following metrics:

| Metric | Purpose |
|---|---|
| Accuracy | Measures the overall proportion of correct predictions |
| Precision | Measures how many predictions assigned to a class are correct |
| Recall | Measures how many actual samples of a class are correctly identified |
| F1-score | Provides a balance between precision and recall |
| Confusion Matrix | Shows correct and incorrect predictions for each class |

Macro-averaged precision, recall, and F1-score will be reported to compare performance across all three flower species.

### 5.6 Data Visualization

The final notebook will include confusion matrix heatmaps and model comparison charts.

Decision boundary plots will also be created using two selected features. These visualizations will help illustrate differences in how the classifiers separate the three flower classes.

## 6. Expected Deliverables

The completed task will include:

- A Jupyter Notebook containing the classification pipeline
- Exploratory data analysis visualizations
- Trained Logistic Regression and Decision Tree models
- Classification reports containing evaluation metrics
- Confusion matrix heatmaps
- Decision boundary visualizations
- A comparison of the two classification algorithms

## 7. Results and Discussion

This section will be updated after the models have been trained and evaluated.

It will include the actual classification results, performance comparisons, observations from the confusion matrices, and a discussion of the strengths and limitations of each model.

## 8. How to Run

After the development environment and required dependencies are configured:

1. Open `notebooks/iris_classification.ipynb` in Jupyter Notebook or VS Code.
2. Select the configured Python environment as the notebook kernel.
3. Run the notebook cells in sequence.
4. Review the evaluation metrics and generated visualizations.

## 9. Project Status

**Status:** Documentation and environment setup in progress.

Model implementation, testing, and evaluation have not yet been completed.

## 10. Author

**Dilara Liyanage**

BSc (Hons) Computer Science Undergraduate  
Sri Lanka Institute of Information Technology (SLIIT)

**Internship:** Datheon — Machine Learning and Data Science