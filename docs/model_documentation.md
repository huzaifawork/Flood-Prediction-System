# Flood Prediction Model Documentation

## Overview

This document provides technical information about the flood prediction model implemented in this project. The model predicts river discharge (in cumecs - cubic meters per second) based on meteorological data and classifies the risk level of potential flooding.

## Model Architecture: Stacking Ensemble

### Why We Used a Stacking Ensemble

We implemented a stacking ensemble model instead of a single algorithm for several reasons:

1. **Superior Performance**: Ensemble methods typically outperform individual models by combining their strengths and reducing weaknesses. Our stacking ensemble demonstrated an R² score of 0.688, which is better than the previous Random Forest model's R² of 0.68.

2. **Better Generalization**: By combining different model types (tree-based and linear), the ensemble generalizes better to unseen data, reducing overfitting risk.

3. **Handles Non-linearity and Complexity**: Hydrological systems are complex with non-linear relationships between weather variables and discharge. The ensemble captures these complex relationships better than any individual model.

4. **More Robust Predictions**: Multiple models working together are less likely to make large errors in predictions since errors from one model can be compensated for by other models.

5. **Feature Importance Analysis**: Different models highlight different aspects of feature importance, providing richer insights about what drives flood prediction.

### Component Models

Our stacking ensemble consists of the following models:

1. **Random Forest Regressor**
   - Hyperparameters: n_estimators=200, max_depth=20, min_samples_split=10, min_samples_leaf=4
   - Strengths: Handles non-linear relationships, robust to outliers, good with high-dimensional data

2. **Gradient Boosting Regressor**
   - Hyperparameters: n_estimators=200, learning_rate=0.1, max_depth=5
   - Strengths: High accuracy, handles mixed data types well, captures complex patterns

3. **XGBoost Regressor**
   - Hyperparameters: n_estimators=200, learning_rate=0.1, max_depth=6
   - Strengths: Regularization capabilities, handles missing values, speed and performance

4. **LightGBM Regressor**
   - Hyperparameters: n_estimators=200, learning_rate=0.1, max_depth=6
   - Strengths: Faster training, handles large datasets efficiently, good with categorical features

5. **Meta-learner**: Ridge Regression
   - Hyperparameters: alpha=1.0
   - Strengths: Prevents overfitting through regularization, stable predictions

### How Stacking Works

1. The base models (RF, GB, XGBoost, LightGBM) are trained on the original training data
2. Using cross-validation, predictions from these models create a new feature set
3. The meta-learner (Ridge) is trained on this new feature set to make the final prediction
4. For new data, base models make predictions which are then fed to the meta-learner

## Performance Metrics

### Regression Metrics

| Metric | Training Set | Test Set |
|--------|--------------|----------|
| RMSE   | 74.24        | 86.94    |
| MAE    | 53.97        | 63.54    |
| R²     | 0.783        | 0.688    |

### Classification Metrics

| Metric    | Value |
|-----------|-------|
| Accuracy  | 73.5% |
| Precision | 73.9% |
| Recall    | 73.5% |
| F1 Score  | 73.3% |

### Confusion Matrix

|           | Predicted Low | Predicted Medium | Predicted High | Predicted Extreme |
|-----------|---------------|------------------|----------------|-------------------|
| Actual Low    | 803           | 126              | 3              | 0                 |
| Actual Medium | 92            | 337              | 80             | 1                 |
| Actual High   | 4             | 108              | 94             | 1                 |
| Actual Extreme| 0             | 13               | 17             | 2                 |

## Risk Classification

The model classifies flood risk into four categories based on the predicted discharge:

| Risk Level | Discharge Range (cumecs) |
|------------|---------------------------|
| Low        | < 200                     |
| Medium     | 200-400                   |
| High       | 400-600                   |
| Extreme    | > 600                     |

## Features

The model uses the following features:

1. Minimum Temperature (°C)
2. Maximum Temperature (°C)
3. Precipitation (mm)
4. Month (derived from date)
5. Day (derived from date) 
6. Day of Year (derived from date)

### Feature Importance

From the Random Forest base model, the features ranked by importance are:

1. Max Temperature (highest importance)
2. Min Temperature
3. Precipitation
4. Month (if available)
5. Day of Year (if available)
6. Day (if available)

## Implementation Details

### Data Preprocessing

1. Missing value imputation using mean values
2. Date conversion and extraction of time-based features
3. Feature scaling using StandardScaler
4. Train-test split (80-20)

### Model Training

1. Individual models trained on scaled features
2. Cross-validation (5-fold) used during stacking
3. Final model trained on full training set

### Model Evaluation

1. Regression metrics (RMSE, MAE, R²) calculated on test set
2. Predicted discharges classified into risk categories
3. Classification metrics (accuracy, precision, recall, F1) calculated
4. Confusion matrix generated to understand error patterns

## Files and Structure

- **Models**: Located in the `models/` directory
  - `stacking_model.joblib` - The trained stacking ensemble model
  - `scaler.joblib` - The fitted StandardScaler for preprocessing

- **Source Code**: Located in the `src/` directory
  - Main scripts for model training and prediction

- **Data**: Located in the `data/` directory
  - `Merged_Weather_Flow_Final_1995_2017.xlsx` - Training dataset

- **Results**: Located in the `results/` directory
  - Visualizations, metrics, and prediction samples

## Usage

To make predictions with the model:

1. Load the model and scaler
2. Preprocess input features (same as training)
3. Transform features using the scaler
4. Generate predictions using the model
5. Classify risk based on predicted discharge values

## Future Improvements

1. Incorporate additional meteorological features if available
2. Explore more complex ensemble architectures
3. Implement time series cross-validation for better temporal generalization
4. Consider deep learning approaches for sequence modeling
5. Explore uncertainty quantification in predictions 