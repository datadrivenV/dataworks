
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam

# Custom Transformer for lagged features
class LagFeatureGenerator(BaseEstimator, TransformerMixin):
    def __init__(self, lags=3):
        self.lags = lags

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for lag in range(1, self.lags + 1):
            for col in X.columns:
                X[f'{col}_lag{lag}'] = X[col].shift(lag)
        X.dropna(inplace=True)
        return X

# Load the data (Extract)
def load_data(filepath):
    return pd.read_csv(filepath)

# Data cleaning and preprocessing (Transform)
def preprocess_data(data):
    data.fillna(method='bfill', inplace=True)  # Back fill to handle missing values
    
    # Feature Engineering with Lag Features
    feature_cols = ['sales0', 'population']
    lag_processor = Pipeline([
        ('lag_gen', LagFeatureGenerator(lags=3)),
        ('imputer', SimpleImputer(strategy='median'))  # Filling any residuals from lagging
    ])
    data[feature_cols] = lag_processor.fit_transform(data[feature_cols])

    # Normalize features
    scaler = MinMaxScaler()
    data[feature_cols] = scaler.fit_transform(data[feature_cols])

    return data

# Prepare the data for LSTM (Load)
def prepare_data_for_lstm(data, target_column, feature_columns):
    X = data[feature_columns].values
    y = data[target_column].values
    X = X.reshape((X.shape[0], 1, X.shape[1]))  # Reshape for LSTM [samples, time steps, features]
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Main execution function
def main():
    # Load data
    data = load_data('store_train.csv')
    
    # Preprocess data
    data = preprocess_data(data)
    
    # Prepare data for model
    features = ['sales0', 'population'] + [f'{x}_lag{y}' for x in ['sales0', 'population'] for y in range(1, 4)]
    X_train, X_test, y_train, y_test = prepare_data_for_lstm(data, 'store', features)
    
    # LSTM Model
    model = Sequential()
    model.add(LSTM(50, input_shape=(1, len(features))))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the model
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)
    
    # Evaluate the model
    predictions = model.predict(X_test)
    predictions = (predictions > 0.5).astype(int).flatten()
    print('Accuracy:', accuracy_score(y_test, predictions))
    print('F1 Score:', f1_score(y_test, predictions))
    print(classification_report(y_test, predictions))

if __name__ == '__main__':
    main()
