# Task 04 — NLP Sentiment Classification on Customer Reviews

## 1. Project Overview

This project focuses on developing a Natural Language Processing (NLP) pipeline to classify customer reviews based on their sentiment.

The main objective is to convert unstructured text into numerical features using Term Frequency-Inverse Document Frequency (TF-IDF) and train supervised machine learning models to identify positive and negative reviews.

The task covers text preprocessing, feature extraction, model training, evaluation, and the development of a function that can classify new user-entered reviews.

Two classification algorithms, Multinomial Naive Bayes and Logistic Regression, will be compared to understand their performance on text classification.

## 2. Dataset Description

**Initial Dataset:** Customer Sentiment Reviews

**File:** `sentiment_reviews.csv`

The supplied dataset contains short customer reviews with manually assigned sentiment labels.

### Dataset Attributes

| Attribute | Description |
|---|---|
| review_id | Unique identifier assigned to each review |
| review_text | Textual customer review |
| sentiment | Sentiment label associated with the review |

### Initial Dataset Summary

The uploaded dataset contains 10 records with three sentiment categories.

| Sentiment | Number of Reviews |
|---|---:|
| Positive | 5 |
| Negative | 4 |
| Neutral | 1 |
| **Total** | **10** |

The internship task focuses on binary sentiment classification using positive and negative labels.

The supplied dataset includes one neutral review and is too small for reliable model benchmarking. Therefore, the implementation will explicitly document how neutral reviews are handled.

A larger, appropriately labelled review dataset may be used for model development and evaluation, subject to the internship requirements. Any additional dataset will be identified and cited in the completed project.

The original supplied dataset will be preserved.

## 3. Project Objectives

The main objectives of this task are to:

1. Load and inspect the customer review dataset using Pandas.
2. Examine sentiment labels, class distributions, missing values, and duplicate reviews.
3. Clean raw review text by removing unnecessary HTML tags, punctuation, and other unwanted characters where appropriate.
4. Convert text to lowercase and perform tokenization.
5. Apply stopword removal and lemmatization where suitable.
6. Convert processed text into numerical features using TF-IDF.
7. Train a Multinomial Naive Bayes classifier.
8. Train a Logistic Regression classifier for comparison.
9. Evaluate both models using classification metrics.
10. Compare the strengths and limitations of the classification algorithms.
11. Implement a `predict_sentiment()` function that accepts new user-entered review text and returns a sentiment prediction.

## 4. Tools and Technologies

- Python
- Pandas
- NumPy
- Scikit-Learn
- NLTK
- Regular Expressions (Regex)
- Matplotlib
- Seaborn
- Jupyter Notebook
- Visual Studio Code

## 5. Planned Methodology

### 5.1 Data Loading and Inspection

The review dataset will be loaded using Pandas.

Initial inspection will include:

- Dataset dimensions
- Column names and data types
- Missing review text or labels
- Duplicate records
- Sentiment class distribution
- Review text length distribution

The `review_id` attribute will be excluded from model training because it is an identifier rather than a feature representing review sentiment.

### 5.2 Sentiment Label Preparation

The task requires classification into positive and negative sentiment categories.

The existing dataset also includes a neutral label.

For binary classification, the neutral record will be excluded from binary model training and evaluation rather than automatically assigning it to either positive or negative sentiment.

The original dataset will remain unchanged.

This decision and its effect on the available training data will be documented.

A larger labelled dataset will be considered for a more reliable experiment if it is permitted within the internship task.

### 5.3 Text Preprocessing

Customer reviews are unstructured text, so preprocessing will be performed before feature extraction.

The planned preprocessing stages include:

**Lowercase Conversion**

Text will be converted to lowercase to reduce unnecessary differences between words with different capitalization.

**HTML Tag Removal**

HTML tags will be removed from reviews when present.

**Punctuation and Unwanted Character Handling**

Unnecessary punctuation and special characters will be removed or normalized where appropriate.

Care will be taken not to remove useful sentiment information such as negation.

**Tokenization**

Text will be divided into smaller units, typically words or tokens.

**Stopword Removal**

Common words that contribute limited information may be removed.

Negation words such as "not" and "never" will be retained because they can change the meaning of a review.

**Lemmatization**

Lemmatization will be considered to reduce words to their dictionary base forms.

The preprocessing choices will be evaluated rather than assuming every cleaning operation improves classification accuracy.

### 5.4 TF-IDF Feature Extraction

Machine learning classifiers require numerical input features.

TF-IDF (Term Frequency-Inverse Document Frequency) will be used to represent review text as numerical vectors.

TF-IDF assigns weights to words based on their occurrence in a document and their distribution across the document collection.

Scikit-Learn's TfidfVectorizer will be used.

The initial configuration will include:

`TfidfVectorizer(max_features=5000)`

Additional settings such as n-gram ranges and minimum document frequency may be investigated if the dataset size supports meaningful tuning.

The vectorizer will be fitted only on training data to prevent data leakage.

### 5.5 Train-Test Splitting

The dataset will be divided into training and testing subsets when sufficient labelled observations are available.

An 80:20 split with stratification will be considered to preserve class proportions.

The original 10-row dataset is insufficient for reliable train-test evaluation, especially after excluding the neutral review.

Any evaluation performed using this small dataset will be identified as a demonstration rather than a reliable benchmark.

### 5.6 Model Training

Two supervised machine learning classifiers will be implemented.

**Multinomial Naive Bayes**

Multinomial Naive Bayes is a probabilistic classification algorithm based on Bayes' theorem.

It is commonly applied to text classification and can work with non-negative word-count or TF-IDF features.

The model assumes conditional independence between features given the class.

Scikit-Learn's MultinomialNB will be used.

**Logistic Regression**

Logistic Regression will be implemented as a second classification model.

It estimates probabilities for the sentiment classes using a linear decision function.

This model will provide a comparison with the probabilistic Naive Bayes approach.

Both models will use comparable training and evaluation data.

### 5.7 Model Evaluation

The classifiers will be evaluated using the following metrics:

| Metric | Purpose |
|---|---|
| Accuracy | Measures the proportion of correctly classified reviews |
| Precision | Measures the proportion of predicted positive reviews that are actually positive |
| Recall | Measures the proportion of actual positive reviews correctly identified |
| F1-score | Provides a balance between precision and recall |
| Confusion Matrix | Shows correct and incorrect predictions for each sentiment class |

Where the dataset permits, precision, recall, and F1-score for both sentiment classes will also be reported.

Evaluation results will be interpreted in relation to the number of available test observations and the class distribution.

### 5.8 Interactive Sentiment Prediction

A function named `predict_sentiment()` will be implemented to classify new user-provided text.

The function will:

1. Accept a review as a string.
2. Apply the same preprocessing used during model development.
3. Transform the text using the trained TF-IDF vectorizer.
4. Predict the sentiment using the selected classification model.
5. Return the predicted sentiment label.

The notebook will include examples demonstrating how the function behaves with different review texts.

These demonstrations will not be treated as independent evidence of classification accuracy.

### 5.9 Model Comparison

The results of Multinomial Naive Bayes and Logistic Regression will be compared using consistent evaluation data.

The analysis will consider:

- Overall predictive performance
- Precision and recall for each class
- Common misclassification patterns
- The effect of text preprocessing
- Limitations arising from dataset size and vocabulary coverage

## 6. Expected Deliverables

The completed task will include:

- A Jupyter Notebook containing the NLP classification pipeline
- Dataset inspection and sentiment distribution analysis
- Text preprocessing functions
- TF-IDF feature extraction
- Multinomial Naive Bayes and Logistic Regression classifiers
- Classification reports and confusion matrices
- Model performance comparison
- An interactive `predict_sentiment()` function
- Example predictions using custom text inputs
- Documentation of findings and limitations

## 7. Results and Discussion

This section will be updated after implementing and evaluating the sentiment classification models.

The final comparison will include:

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Multinomial Naive Bayes | Pending | Pending | Pending | Pending |
| Logistic Regression | Pending | Pending | Pending | Pending |

Actual performance values will be reported only after the experiments have been completed.

The discussion will explain the observed results and any limitations associated with dataset size, class distribution, or text preprocessing.

## 8. How to Run

After setting up the Python environment and installing the required dependencies:

1. Open `notebooks/sentiment_classification.ipynb` in VS Code or Jupyter Notebook.
2. Select the configured Python environment as the notebook kernel.
3. Ensure the required review dataset is available in the `data/` directory.
4. Run the notebook cells in sequence.
5. Review the classification reports and visualizations.
6. Test the `predict_sentiment()` function with sample review text.

## 9. Project Status

**Status:** Documentation and environment setup in progress.

Text preprocessing, model training, evaluation, and interactive prediction have not yet been completed.

The suitability of the supplied dataset for reliable benchmarking will be addressed during implementation.

## 10. Author

**Dilara Liyanage**

BSc (Hons) Computer Science Undergraduate  
Sri Lanka Institute of Information Technology (SLIIT)

**Internship:** Datheon — Machine Learning and Data Science