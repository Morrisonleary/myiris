import pandas as pd
import numpy as np
import sklearn
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split



print("Hello, World!")

# Load the Iris dataset
iris_data = load_iris()
iris_df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
iris_df['species'] = pd.Categorical.from_codes(iris_data.target, iris_data.target_names)

# Data Exploration
sns.pairplot(iris_df, hue='species')
plt.show()

iris_df.hist(edgecolor='black', linewidth=1.2)
fig = plt.gcf()
fig.set_size_inches(12, 6)
plt.show()

iris_df.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
plt.show()

plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.violinplot(x='species', y='sepal length (cm)', data=iris_df)
plt.subplot(2,2,2)
sns.violinplot(x='species', y='sepal width (cm)', data=iris_df)
plt.subplot(2,2,3)
sns.violinplot(x='species', y='petal length (cm)', data=iris_df)
plt.subplot(2,2,4)
sns.violinplot(x='species', y='petal width (cm)', data=iris_df)
plt.show()

numeric_data = iris_df.select_dtypes(include=[np.number])
correlation_matrix = numeric_data.corr()
plt.figure(figsize=(10,7))
sns.heatmap(correlation_matrix, annot=True, cmap='cubehelix_r')
plt.show()

# Data Preprocessing
X = iris_df.drop('species', axis=1)
y = iris_df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature Selection and Model Training
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
importances = model.feature_importances_
for feature, importance in zip(iris_data.feature_names, importances):
    print(f"{feature}: {importance}")

# Cross-Validation
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Cross-Validation Accuracy: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")

# Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Set Accuracy: {accuracy:.2f}")

class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()