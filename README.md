🧠 Naive Bayes Classifier on Iris Dataset

This project demonstrates how to use Naive Bayes (GaussianNB) from scikit-learn to classify the famous Iris dataset.

📌 Overview

Loads the Iris dataset from sklearn.datasets.

Splits the data into training (70%) and testing (30%) sets.

Applies Gaussian Naive Bayes classifier.

Evaluates model performance with:

Accuracy Score

Classification Report (precision, recall, F1-score)

Confusion Matrix

⚙️ Requirements

Install dependencies with:

pip install scikit-learn numpy

▶️ How to Run

Save the code in a file, e.g., naive_bayes_iris.py, then run:

python naive_bayes_iris.py

📊 Expected Output

The program prints the Naive Bayes results:

=== Naive Bayes ===
Accuracy: 0.96
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        19
           1       0.95      0.95      0.95        19
           2       0.95      0.95      0.95        17

    accuracy                           0.96        55
   macro avg       0.96      0.96      0.96        55
weighted avg       0.96      0.96      0.96        55

[[19  0  0]
 [ 0 18  1]
 [ 0  1 16]]


(numbers may vary slightly due to random splits)

📚 Notes

The model uses Gaussian Naive Bayes, suitable for continuous features.

You can easily replace it with other classifiers (like SVM, Decision Trees) to compare performance.

Try experimenting with different test_size or add feature scaling for other models.
