# Breast Cancer Classification – Data Exploration & Preprocessing
This project performs exploratory data analysis (EDA), preprocessing, and dataset preparation for a breast cancer classification model using the Wisconsin Breast Cancer dataset.
## Project Overview
The objective of this project is to explore the dataset, clean it, visualize important patterns, and prepare the data for machine learning classification.
We convert raw CSV data into a structured format suitable for model training.
## Technologies Used
Python
NumPy
Pandas
Matplotlib
Scikit-learn
## Dataset
### The project uses:
load_breast_cancer() from scikit-learn (for reference)
data.csv (Breast Cancer Wisconsin dataset)
The dataset contains diagnostic measurements of breast cell nuclei and a target variable:
M = Malignant (mapped to 1)
B = Benign (mapped to 0)
## Steps Performed
1. Import Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
plt.style.use('ggplot')
2. Load Breast Cancer Dataset (Sklearn Reference)
data = load_breast_cancer()
print(data.feature_names)
print(data.target_names)
3. Load the CSV Dataset
df = pd.read_csv('data.csv')
print(df.head())
print(df.info())
4. Data Cleaning
Remove columns not needed:
df = df.drop(columns=['id', 'Unnamed: 32'])
5. Feature/Target Split
X = df.drop(columns=['diagnosis'])
y = df['diagnosis'].map({'M': 1, 'B': 0})
6. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
7. Visualizations
Target Distribution
A bar chart showing counts of benign vs malignant cases.
### Feature Distributions
#### Histograms for:
- radius_mean
- texture_mean
- perimeter_mean
- area_mean
These help understand feature spread and detect skewness.
## Key Insights
- The dataset is imbalanced, with benign cases being more common.
- Distribution plots provide insight into which features may separate the classes well.
