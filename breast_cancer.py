import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
plt.style.use('ggplot')
# Breast cancer dataset for classification
data = load_breast_cancer()
print (data.feature_names)
print (data.target_names)
# Convert to DataFrame for easier handling
df = pd.read_csv('data.csv')
print(df.head())
print(df.info())
# removev less useful columns
df = df.drop(columns=['id', 'Unnamed: 32'])
# Split the dataset into features and target variable
X = df.drop(columns=['diagnosis'])
y = df['diagnosis'].map({'M': 1, 'B': 0})  # Map 'M' to 1 and 'B' to 0
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f'Training set size: {X_train.shape}, Testing set size: {X_test.shape}')
# Display first few rows of the training set
print(X_train.head())
print(y_train.head())
# Visualize the distribution of the target variable
plt.figure(figsize=(6,4))
y.value_counts().plot(kind='bar', color=['red', 'black'])
plt.title('Distribution of Diagnosis')
plt.xlabel('Diagnosis (0 = Benign, 1 = Malignant)')
plt.ylabel('Count')
plt.show()
# Visualize some feature distributions
feature_cols = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean']
for col in feature_cols:
    plt.figure(figsize=(6,4))
    plt.hist(df[col], bins=30, color='brown', alpha=0.7)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

