
# Bank Marketing Campaign - Python Project with Model Performance Visualizations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# Load datasets
train_data = pd.read_csv('bank-full_train.csv')
test_data = pd.read_csv('bank-full_test.csv')

# EDA - Exploratory Data Analysis
def eda_visualizations(data):
    plt.figure(figsize=(10, 6))
    sns.histplot(data['age'], kde=True)
    plt.title("Age Distribution")
    plt.savefig('eda_age_distribution.png')

    plt.figure(figsize=(10, 6))
    sns.countplot(data['job'], order=data['job'].value_counts().index)
    plt.xticks(rotation=45)
    plt.title("Job Type Distribution")
    plt.savefig('eda_job_distribution.png')

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='y', y='balance', data=data)
    plt.title("Balance by Subscription")
    plt.savefig('eda_balance_by_subscription.png')
    
    # Heatmap of correlations
    plt.figure(figsize=(10, 6))
    corr = data.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig('eda_correlation_heatmap.png')

# Preprocess data
def preprocess_data(data, is_train=True):
    label_encoder = LabelEncoder()
    categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    for col in categorical_cols:
        data[col] = label_encoder.fit_transform(data[col])
    
    if 'y' in data.columns:  # Target only in train
        data['y'] = label_encoder.fit_transform(data['y'])
    features = data.drop(columns=['y', 'ID']) if is_train else data.drop(columns=['ID'])
    
    scaler = StandardScaler()
    features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)
    return features, data['y'] if 'y' in data.columns else None

# Apply EDA and Preprocessing
eda_visualizations(train_data)
X_train, y_train = preprocess_data(train_data)
X_test, _ = preprocess_data(test_data, is_train=False)

# Neural Network Model
def create_nn_model(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Model Training
model = create_nn_model(X_train.shape[1])
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32, callbacks=[early_stopping])

# Save model and training history plots
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Model Loss Over Epochs')
plt.savefig('nn_model_loss.png')

# Predictions on Test Data
y_test_pred = model.predict(X_test)
y_test_pred_class = (y_test_pred > 0.5).astype(int)

# Model Performance Evaluation on Training Data
y_train_pred = model.predict(X_train)
y_train_pred_class = (y_train_pred > 0.5).astype(int)
fpr, tpr, thresholds = roc_curve(y_train, y_train_pred)
roc_auc = roc_auc_score(y_train, y_train_pred)

# Plot ROC Curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Training Data')
plt.legend(loc="lower right")
plt.savefig('nn_model_roc_curve.png')

# Confusion Matrix Visualization
conf_matrix = confusion_matrix(y_train, y_train_pred_class)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Training Data")
plt.savefig('nn_model_confusion_matrix.png')

# Output predictions to CSV for review
test_data['y_pred'] = y_test_pred_class
test_data[['ID', 'y_pred']].to_csv('bank_test_predictions.csv', index=False)

# End of script
