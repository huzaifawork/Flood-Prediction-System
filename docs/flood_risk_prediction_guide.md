# Flood Risk Prediction System: User Guide

## Introduction

This guide explains how to use the flood risk prediction system to forecast river discharge levels and assess potential flood risks based on meteorological data. The system uses a sophisticated machine learning model that takes weather inputs and provides predictions about expected discharge in cubic meters per second (cumecs).

## Risk Levels Explained

Our system classifies flood risk into four categories:

| Risk Level | Discharge Range (cumecs) | Description |
|------------|--------------------------|-------------|
| **Low**    | < 200                    | Normal river conditions, no significant flooding expected |
| **Medium** | 200-400                  | Elevated water levels, possible minor flooding in flood-prone areas |
| **High**   | 400-600                  | Significant flooding possible, evacuation preparations may be needed |
| **Extreme**| > 600                    | Severe flooding expected, immediate action required including possible evacuations |

## Required Inputs

To generate a flood prediction, you need the following weather data:

1. **Minimum Temperature** (°C)
2. **Maximum Temperature** (°C)
3. **Precipitation** (mm)
4. **Date** (to derive month, day, and day-of-year features)

## How to Use the Model

### Option 1: Using the Python Script

1. Ensure you have Python installed with the required packages (pandas, numpy, joblib, scikit-learn)

2. Load the model and scaler:
   ```python
   import joblib
   import pandas as pd
   import numpy as np
   
   # Load the model and scaler
   model = joblib.load('models/stacking_model.joblib')
   scaler = joblib.load('models/scaler.joblib')
   ```

3. Prepare your input data:
   ```python
   # For a single prediction
   data = {
       'Min Temp': 20.0,
       'Max Temp': 32.0,
       'Prcp': 8.0,
       'Month': 7,  # Derived from date
       'Day': 20,    # Derived from date
       'DayOfYear': 201  # Derived from date
   }
   
   # Convert to DataFrame
   features = pd.DataFrame([data])
   ```

4. Make predictions:
   ```python
   # Scale the features
   features_scaled = scaler.transform(features)
   
   # Predict discharge
   predicted_discharge = model.predict(features_scaled)[0]
   
   print(f"Predicted Discharge: {predicted_discharge:.2f} cumecs")
   ```

5. Classify risk level:
   ```python
   def classify_risk(discharge):
       if discharge < 200:
           return "Low Risk"
       elif discharge < 400:
           return "Medium Risk"
       elif discharge < 600:
           return "High Risk"
       else:
           return "Extreme Risk"
   
   risk_level = classify_risk(predicted_discharge)
   print(f"Flood Risk: {risk_level}")
   ```

### Option 2: For Batch Predictions

To process multiple weather scenarios:

1. Prepare a CSV or Excel file with columns: 'Min Temp', 'Max Temp', 'Prcp', and date information

2. Use this code:
   ```python
   # Load batch data
   batch_data = pd.read_csv('your_scenarios.csv')
   
   # Extract date features if needed
   if 'Date' in batch_data.columns:
       batch_data['Date'] = pd.to_datetime(batch_data['Date'])
       batch_data['Month'] = batch_data['Date'].dt.month
       batch_data['Day'] = batch_data['Date'].dt.day
       batch_data['DayOfYear'] = batch_data['Date'].dt.dayofyear
   
   # Select relevant features
   features = batch_data[['Min Temp', 'Max Temp', 'Prcp', 'Month', 'Day', 'DayOfYear']]
   
   # Scale features
   features_scaled = scaler.transform(features)
   
   # Generate predictions
   predictions = model.predict(features_scaled)
   
   # Classify risk levels
   batch_data['Predicted_Discharge'] = predictions
   batch_data['Risk_Level'] = batch_data['Predicted_Discharge'].apply(classify_risk)
   
   # Save results
   batch_data.to_csv('prediction_results.csv', index=False)
   ```

## Interpreting Results

When interpreting the predictions, consider:

1. **Prediction Confidence**: The model performs best for low and medium risk categories, with 86% and 66% accuracy respectively.

2. **Uncertainty**: Higher discharge predictions (especially in the extreme range) have greater uncertainty.

3. **Local Factors**: The model predictions should be adjusted based on local knowledge of terrain, infrastructure, and previous flood patterns.

4. **Time Horizon**: The model predicts based on current conditions and does not account for future weather changes.

## Example Scenarios

| Weather Conditions | Predicted Discharge | Risk Level |
|--------------------|---------------------|------------|
| Min=15°C, Max=25°C, Prcp=2.5mm | 194.35 cumecs | Low |
| Min=20°C, Max=32°C, Prcp=8.0mm | 276.16 cumecs | Medium |
| Min=22°C, Max=35°C, Prcp=12.0mm | 432.43 cumecs | High |
| Min=24°C, Max=37°C, Prcp=20.0mm | 561.69 cumecs | High |

## Best Practices and Recommendations

1. **Regular Updates**: Use the most recent weather data available for accurate predictions.

2. **Combined Approach**: Use this model alongside traditional forecasting methods for critical decisions.

3. **Local Calibration**: Consider calibrating risk thresholds based on historical flooding in your specific region.

4. **Multiple Scenarios**: Run predictions with different weather scenarios to understand potential range of outcomes.

5. **Emergency Planning**: For Medium risk and above, review emergency flood response plans.

## Technical Support

For technical issues or questions about the model:

- Check documentation in the `docs/` directory
- Review model metrics in the `results/` directory
- Contact the model development team for additional support

## Limitations

- The model's accuracy depends on the quality of the input data
- Local geographical features may affect actual flooding in ways not captured by the model
- Extreme events outside the range of training data may have reduced prediction accuracy 