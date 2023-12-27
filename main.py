import pandas as pd
import numpy as np
import sklearn
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split



print("Hello, World!")

# Step 3

# Load the Iris dataset
iris_data = load_iris()
iris_df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)

# Add a column for the species, which is our target
iris_df['species'] = pd.Categorical.from_codes(iris_data.target, iris_data.target_names)

# iris_data = load_iris() loads the Iris dataset.
# pd.DataFrame(iris_data.data, columns=iris_data.feature_names) converts the dataset into a DataFrame. The columns are named after the features of the Iris dataset.
# iris_df['species'] = ... adds a new column to the DataFrame for the species of each iris flower. The species is the target variable you will predict.

# Display the first 5 rows of the DataFrame
# print(iris_df.head())

# Display basic information about the dataset
# print(iris_df.info())

# iris_df.head() displays the first five rows of the DataFrame, giving you a glimpse of the data.
# iris_df.info() provides a concise summary of the DataFrame, including the number of non-null entries in each column and the data types.

# Step 4
# Descriptivie statistics
# print(iris_df.describe())
# Class distribution 
# print(iris_df['species'].value_counts())
# Check for Null Values
# print(iris_df.isnull().sum())

# Pair Plot
sns.pairplot(iris_df, hue='species')
plt.show()

# Histograms for Each Feature
iris_df.hist(edgecolor='black', linewidth=1.2)
fig = plt.gcf()
fig.set_size_inches(12, 6)
plt.show()

# Box Plots
iris_df.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
plt.show()

# Violin Plots
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

# Correlation Matrix Heatmap
numeric_data = iris_df.select_dtypes(include=[np.number])
correlation_matrix = numeric_data.corr()
plt.figure(figsize=(10,7))
sns.heatmap(correlation_matrix, annot=True, cmap='cubehelix_r')
plt.show()


# Data Preprocessing

# Splitting Data into Features and Target
X = iris_df.drop('species', axis=1)
y = iris_df['species']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5 Feature Selection Feature Importance with a Decision Tree Model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
importances = model.feature_importances_

# Print the feature importances
for feature, importance in zip(iris_data.feature_names, importances): print(f"{feature}: {importance}")

# Initialize the model
dt_model = DecisionTreeClassifier(random_state=42)
rf_model = RandomForestClassifier(random_state=42)
knn_model = KNeighborsClassifier()

# Train the model 
dt_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)

# Make predictions
dt_predictions = dt_model.predict(X_test)
rf_predictions = rf_model.predict(X_test)
knn_predictions = knn_model.predict(X_test)

# Evaluate accuracy
dt_accuracy = accuracy_score(y_test, dt_predictions)
rf_accuracy = accuracy_score(y_test, rf_predictions)
knn_accuracy = accuracy_score(y_test, knn_predictions)

print(f"Decision Tree Accuracy: {dt_accuracy}")
print(f"Random Forest Accuracy: {rf_accuracy}")
print(f"KNN Accuracy: {knn_accuracy}")

# Feature Scaling (Optional)
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)