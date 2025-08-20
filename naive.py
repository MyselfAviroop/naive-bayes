from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.preprocessing import MinMaxScaler

# Load Iris dataset
X, y = load_iris(return_X_y=True)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- 1. Gaussian Naive Bayes (works naturally with continuous data) ---
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_g = gnb.predict(X_test)

print("\n=== Gaussian Naive Bayes ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_g):.2f}")
print(classification_report(y_test, y_pred_g))
print(confusion_matrix(y_test, y_pred_g))

# --- 2. Bernoulli Naive Bayes (needs binary features, so we binarize with threshold) ---
bnb = BernoulliNB()
# scale features to [0,1] then binarize
scaler = MinMaxScaler()
X_train_b = scaler.fit_transform(X_train) > 0.5
X_test_b = scaler.transform(X_test) > 0.5

bnb.fit(X_train_b, y_train)
y_pred_b = bnb.predict(X_test_b)

print("\n=== Bernoulli Naive Bayes ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_b):.2f}")
print(classification_report(y_test, y_pred_b))
print(confusion_matrix(y_test, y_pred_b))

# --- 3. Multinomial Naive Bayes (expects counts / non-negative features) ---
mnb = MultinomialNB()
# Scale to positive values (0–1), then multiply to simulate counts
X_train_m = (scaler.fit_transform(X_train) * 10).astype(int)
X_test_m = (scaler.transform(X_test) * 10).astype(int)

mnb.fit(X_train_m, y_train)
y_pred_m = mnb.predict(X_test_m)

print("\n=== Multinomial Naive Bayes ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_m):.2f}")
print(classification_report(y_test, y_pred_m))
print(confusion_matrix(y_test, y_pred_m))
