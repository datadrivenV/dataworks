import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb

# Load the data
data = pd.read_csv('Reduced_Phishing_URL_Dataset.csv')

# Drop non-numeric columns
non_numeric_columns = data.select_dtypes(exclude=['number']).columns
if len(non_numeric_columns) > 0:
    data = data.drop(columns=non_numeric_columns)

# Separate features and target
X = data.drop(columns='label').values
y = data['label'].values

# Add noise to features
X += np.random.normal(0, 0.01, X.shape)

# Flip 10% of labels
flip_indices = np.random.choice(len(y), size=int(0.1 * len(y)), replace=False)
y[flip_indices] = 1 - y[flip_indices]

# Split data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Convert data to DMatrix format for xgb.train
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)

# Define parameters for XGBoost
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 2,
    'learning_rate': 0.01,
    'reg_alpha': 50,
    'reg_lambda': 50,
    'min_child_weight': 15,
    'subsample': 0.4,
    'colsample_bytree': 0.4,
    'seed': 42
}

# Train the XGBoost model with early stopping
evals = [(dtrain, 'train'), (dval, 'eval')]
xgb_model = xgb.train(
    params,
    dtrain,
    num_boost_round=100,
    evals=evals,
    early_stopping_rounds=10,
    verbose_eval=True
)

# Predict on the test set using XGBoost
y_pred = (xgb_model.predict(dtest) > 0.5).astype(int)
y_prob = xgb_model.predict(dtest)

# Evaluate the XGBoost model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print(f"\nXGBoost Test Set Performance: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

# Benchmarking other models
models = {
    "Logistic Regression": LogisticRegression(max_iter=200, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=3, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42),
    "Naive Bayes": GaussianNB()
}

# Scale data for benchmarking models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

performance = {
    'Model': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1 Score': [],
    'AUC': []
}

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    probabilities = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    auc = roc_auc_score(y_test, probabilities) if probabilities is not None else "N/A"

    # Append metrics to dictionary
    performance['Model'].append(model_name)
    performance['Accuracy'].append(accuracy)
    performance['Precision'].append(precision)
    performance['Recall'].append(recall)
    performance['F1 Score'].append(f1)
    performance['AUC'].append(auc)

    # Confusion Matrix
    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

# Display performance metrics for all models
performance_df = pd.DataFrame(performance)
print("\nComparison of Performance Metrics Across Models:")
print(performance_df)

# Plot comparison of metrics
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
for i, metric in enumerate(metrics):
    sns.barplot(x='Model', y=metric, data=performance_df, ax=axes[i], palette='Blues')
    axes[i].set_title(f'Model {metric} Comparison')
    axes[i].set_ylim(0, 1)
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.show()

# Plot ROC curves for all models including XGBoost
plt.figure(figsize=(10, 8))
if hasattr(xgb_model, "predict"):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"XGBoost (AUC = {auc:.2f})")

for model_name, model in models.items():
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        auc_score = roc_auc_score(y_test, probabilities)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_score:.2f})")

# Plot the diagonal for random guessing
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")

plt.title("ROC Curves for All Models", fontsize=14)
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.legend(loc="lower right")
plt.show()
