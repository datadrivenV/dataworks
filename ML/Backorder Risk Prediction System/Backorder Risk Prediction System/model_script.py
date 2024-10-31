import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
product_train_df = pd.read_csv('product_train.csv')

# Data Preprocessing
# Identifying missing values and filling them
product_train_df.fillna(product_train_df.median(), inplace=True)

# Advanced preprocessing with Pipeline
numeric_features = product_train_df.select_dtypes(include=['int64', 'float64']).columns
categorical_features = product_train_df.select_dtypes(include=['object']).columns

# Creating a column transformer to apply different preprocessing to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Exploratory Data Analysis
# Boxplots for numerical features to detect outliers
plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_features, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(y=product_train_df[col])
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()

# Correlation matrix heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(product_train_df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Feature and target preparation
X = product_train_df.drop('went_on_backorder', axis=1)
y = product_train_df['went_on_backorder'].apply(lambda x: 1 if x == 'Yes' else 0)

# Integrating preprocessor with a modeling pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42))])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model
pipeline.fit(X_train, y_train)

# Predictions and evaluation
y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]  # Probability estimates

print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("ROC AUC Score:")
print(roc_auc_score(y_test, y_pred_proba))

# Feature Importance Visualization (if applicable)
try:
    feature_importances = pd.DataFrame({'feature': X.columns, 'importance': pipeline.named_steps['classifier'].feature_importances_})
    feature_importances.sort_values('importance', ascending=False, inplace=True)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importances, x='importance', y='feature')
    plt.title('Feature Importance')
    plt.show()
except AttributeError:
    print("Feature importance not available for the pipeline configuration.")
