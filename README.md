## Credit Card Fraud Detection using Python and Scikit Learn

This Jupyter Notebook demonstrates the process of building a machine learning model to detect credit card fraud using logistic regression and evaluation metrics in Python.

### Overview

This project involves the following steps:

1. **Importing Libraries:** Importing necessary Python libraries such as NumPy, Pandas, and scikit-learn.
2. **Loading the Dataset:** Reading the credit card dataset into a Pandas DataFrame.
3. **Data Preprocessing:** Normalizing the 'Amount' column using StandardScaler and dropping irrelevant columns like 'Time' and original 'Amount'.
4. **Handling Imbalanced Data:** Creating a balanced dataset by combining fraud and legitimate samples.
5. **Splitting the Data:** Dividing the dataset into features (X) and target variable (y) for training and testing purposes using the train_test_split function.
6. **Model Building:** Initializing a Logistic Regression model for credit card fraud detection.
7. **Hyperparameter Tuning:** Using GridSearchCV to perform hyperparameter tuning (regularization parameter 'C') using 5-fold cross-validation and selecting the best model.
8. **Model Evaluation:** Training the best model on the training set and evaluating its performance on the test set using various evaluation metrics:
   - Accuracy Score: Measures the overall accuracy of the model's predictions.
   - Precision Score: Indicates the model's ability to avoid false positives.
   - Recall Score: Measures the model's capability to identify actual positives.
   - F1 Score: Represents the harmonic mean of precision and recall.
   - Confusion Matrix: Visualizes the model's performance showing true positives, true negatives, false positives, and false negatives.
9. **Cross-Validation:** Utilizing 5-fold cross-validation to assess the model's performance on different subsets of the data and obtaining cross-validation scores.
10. **Conclusion:** Summarizing the model's performance and mean cross-validation score.

### Result Summary

- **Accuracy:** 91.23%
- **Precision:** 92.59%
- **Recall:** 89.29%
- **F1 Score:** 90.91%
- **Confusion Matrix:**
```
[[27 2]
[ 3 25]]
```
- **Cross-Validation Scores:** [98.25%, 92.59%, 94.55%, 92.31%, 98.25%]
- **Mean CV Score:** 95.19%

## The results indicate a well-performing model with high accuracy, precision, recall, and F1 score. The confusion matrix illustrates low false positives and false negatives, signifying the model's accuracy in predicting both fraudulent and legitimate transactions.

![image](https://github.com/ParmeetChanne/credit-fraud-detection/assets/67189839/3719a3ef-194a-4e52-b222-75eff641582c)


---
