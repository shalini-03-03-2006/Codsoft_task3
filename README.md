# Codsoft_task3
Great 🌸! The Iris dataset is a classic beginner-friendly dataset for classification tasks. It contains 150 samples of iris flowers with 4 features:

Sepal Length

Sepal Width

Petal Length

Petal Width


Target (label) = Species (Setosa, Versicolor, Virginica).

We’ll build a classification model using Python + Scikit-learn.


---

📝 Step-by-Step: Iris Flower Classification

1. Import Libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


---

2. Load the Iris Dataset

# Load dataset from sklearn
iris = load_iris()

# Create DataFrame
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data["species"] = iris.target

# Map numeric target to species names
data["species"] = data["species"].map({0:"setosa", 1:"versicolor", 2:"virginica"})

print(data.head())
print(data["species"].value_counts())


---

3. Data Visualization

# Pairplot to see relationships
sns.pairplot(data, hue="species", palette="Set2")
plt.show()

# Correlation heatmap
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.show()


---

4. Split Features & Target

X = data.drop("species", axis=1)
y = data["species"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


---

5. Train Model (Logistic Regression)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)


---

6. Evaluate Model

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


---

7. Visualization of Predictions

# Compare actual vs predicted
comparison = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
print(comparison.head())

# Plot confusion matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Blues", fmt="d",
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Iris Classification")
plt.show()


---

✅ End Result:

You’ll get a Logistic Regression model that classifies iris flowers with ~95–98% accuracy.

This project helps you understand classification basics.



---

👉 Do you want me to also show you how to improve this model with a Decision Tree & Random Forest (just like an extension to make Task 3 more powerful)?

