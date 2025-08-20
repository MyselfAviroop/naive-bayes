🧠 Naive Bayes Classifiers on Iris Dataset

This project demonstrates how to apply three types of Naive Bayes classifiers on the classic Iris dataset using scikit-learn:

GaussianNB → Best suited for continuous features (natural fit for Iris).

BernoulliNB → Works on binary features (applied here with binarization).

MultinomialNB → Designed for count/discrete features (adapted here with scaling).

📌 Overview

Load the Iris dataset.

Split into train (70%) and test (30%) sets.

Train three Naive Bayes models: Gaussian, Bernoulli, and Multinomial.

Evaluate each model with:

Accuracy Score

Classification Report (Precision, Recall, F1-score)

Confusion Matrix

⚙️ Requirements

Install dependencies:

pip install scikit-learn numpy

▶️ How to Run

Save the code in naive_bayes_variants.py and run:

python naive_bayes_variants.py

📊 Example Output
=== Gaussian Naive Bayes ===
Accuracy: 0.96
...
Confusion Matrix:
[[19  0  0]
 [ 0 18  1]
 [ 0  1 16]]

=== Bernoulli Naive Bayes ===
Accuracy: 0.67
...
Confusion Matrix:
[[19  0  0]
 [ 0 11  8]
 [ 0  9  8]]

=== Multinomial Naive Bayes ===
Accuracy: 0.91
...
Confusion Matrix:
[[19  0  0]
 [ 0 17  2]
 [ 0  3 14]]


(Numbers may vary depending on random split and scaling)

📚 Notes

GaussianNB performs best here (since features are continuous).

BernoulliNB performs worst — binarization destroys too much information.

MultinomialNB does reasonably well after scaling features into integer counts.

The experiment shows how different distribution assumptions in Naive Bayes affect results.
