
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, plot_precision_recall_curve, f1_score

# Load the datasets
train_data = pd.read_csv('hr_train.csv')
test_data = pd.read_csv('hr_test.csv')

# Data Cleaning - Handling missing values
train_data.fillna(method='ffill', inplace=True)
test_data.fillna(method='ffill', inplace=True)

# EDA - Distribution of 'satisfaction_level'
plt.figure(figsize=(10, 6))
sns.histplot(train_data['satisfaction_level'], kde=True, color='blue')
plt.title('Distribution of Satisfaction Level')
plt.xlabel('Satisfaction Level')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig('satisfaction_level_distribution.png')

# EDA - Heatmap for correlation
plt.figure(figsize=(12, 8))
sns.heatmap(train_data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.png')

# Preprocessing Pipeline
categorical_features = ['sales', 'salary']
numeric_features = train_data.drop(['left', 'sales', 'salary'], axis=1).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Prepare features and target
X = train_data.drop('left', axis=1)
y = train_data['left']

# SVM Model Pipeline
svm_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC(kernel='rbf', probability=True))
])

# Parameter grid for GridSearch
param_grid = {'classifier__C': [1], 'classifier__gamma': ['scale']}
grid_search = GridSearchCV(svm_pipeline, param_grid, cv=3, scoring='accuracy')
grid_search.fit(X, y)

# Best SVM model from Grid Search
best_svm_pipeline = grid_search.best_estimator_

# Evaluation using the test set
X_test = test_data.drop('left', axis=1)
y_test = test_data['left']
y_pred = best_svm_pipeline.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Visualization - Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix for SVM Model')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('svm_confusion_matrix.png')

# Advanced Evaluation - Precision-Recall Curve and F1 Score
plt.figure(figsize=(10, 8))
plot_precision_recall_curve(best_svm_pipeline, X_test, y_test)
plt.title('Precision-Recall Curve')
plt.savefig('precision_recall_curve.png')

f1 = f1_score(y_test, y_pred, average='weighted')
print('F1 Score:', f1)
