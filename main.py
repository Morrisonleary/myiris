import pandas as pd
import numpy as np
import sklearn
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns



print("Hello, World!")

# Load the Iris dataset
iris_data = load_iris()
iris_df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)

# Add a column for the species, which is our target
iris_df['species'] = pd.Categorical.from_codes(iris_data.target, iris_data.target_names)

# iris_data = load_iris() loads the Iris dataset.
# pd.DataFrame(iris_data.data, columns=iris_data.feature_names) converts the dataset into a DataFrame. The columns are named after the features of the Iris dataset.
# iris_df['species'] = ... adds a new column to the DataFrame for the species of each iris flower. The species is the target variable you will predict.

# Display the first 5 rows of the DataFrame
print(iris_df.head())

# Display basic information about the dataset
print(iris_df.info())

# iris_df.head() displays the first five rows of the DataFrame, giving you a glimpse of the data.
# iris_df.info() provides a concise summary of the DataFrame, including the number of non-null entries in each column and the data types.