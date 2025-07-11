"""
SUPARCO Enhanced Flood Prediction System
========================================
Integrates current ML models with 200-year climate projections
Based on Sattar et al., 2020 research findings:
- Temperature rise: +1.3°C to +3.7°C (RCP4.5 & RCP8.5)
- Precipitation change: -20% to +23%
- Streamflow impact: Increased Nov-May, Reduced Jun-Dec
- 5-GCM ensemble average (SUPARCO recommended)
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys
import json
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore', category=UserWarning)

# Get the absolute path to the models directory
MODELS_DIR = os.path.dirname(os.path.abspath(__file__))

def classify_risk(discharge):
    """
    Enhanced 4-level risk classification for Swat Basin at Chakdara
    Based on regional flood thresholds and Sattar et al., 2020 findings
    """
    if discharge < 350:
        return "Low Risk"
    elif discharge < 600:
        return "Medium Risk"
    elif discharge < 900:
        return "High Risk"
    else:
        return "Extreme Risk"

def load_suparco_forecast_data():
    """Load SUPARCO 200-year forecast data if available"""
    forecast_file = os.path.join(os.path.dirname(MODELS_DIR), '..', 'suparco_200_year_forecast.csv')
    
    if os.path.exists(forecast_file):
        try:
            forecast_df = pd.read_csv(forecast_file)
            return forecast_df
        except Exception as e:
            print(f"Warning: Could not load SUPARCO forecast data: {e}")
    
    return None

def get_climate_projection(target_date=None):
    """Get climate projection for a specific date from SUPARCO data"""
    forecast_data = load_suparco_forecast_data()
    
    if forecast_data is None:
        return None
    
    if target_date is None:
        target_date = datetime.now()
    
    # Convert target_date to pandas datetime
    target_date = pd.to_datetime(target_date)
    
    # Find closest date in forecast data
    forecast_data['Date'] = pd.to_datetime(forecast_data['Date'])
    closest_idx = (forecast_data['Date'] - target_date).abs().idxmin()
    
    projection = forecast_data.iloc[closest_idx]
    
    return {
        'date': projection['Date'].strftime('%Y-%m-%d'),
        'year': projection['Year'],
        'precipitation': projection['Precipitation'],
        'min_temp': projection['Min_Temp'],
        'max_temp': projection['Max_Temp'],
        'predicted_discharge': projection['Predicted_Discharge'],
        'flood_risk': projection['Flood_Risk'],
        'source': 'SUPARCO 200-year forecast (SSP 585)'
    }

def enhanced_flood_prediction(min_temp, max_temp, precipitation, target_date=None):
    """
    Enhanced flood prediction using both ML model and SUPARCO projections
    """
    results = {
        'current_prediction': None,
        'climate_projection': None,
        'combined_assessment': None
    }
    
    # Current ML model prediction
    try:
        model = joblib.load(os.path.join(MODELS_DIR, 'stacking_model.joblib'))
        scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.joblib'))
        
        # Prepare input data
        features = pd.DataFrame([{
            'Min Temp': min_temp,
            'Max Temp': max_temp,
            'Prcp': precipitation
        }])
        
        # Scale and predict
        features_scaled = scaler.transform(features)
        predicted_discharge = model.predict(features_scaled)[0]
        risk_level = classify_risk(predicted_discharge)
        
        results['current_prediction'] = {
            'discharge': predicted_discharge,
            'risk_level': risk_level,
            'source': 'ML Model (Historical data 1995-2017)'
        }
        
    except Exception as e:
        print(f"Error with ML model prediction: {e}")
    
    # SUPARCO climate projection
    if target_date:
        projection = get_climate_projection(target_date)
        if projection:
            results['climate_projection'] = projection
    
    # Combined assessment
    if results['current_prediction'] and results['climate_projection']:
        # Weight the predictions based on temporal relevance
        current_discharge = results['current_prediction']['discharge']
        projected_discharge = results['climate_projection']['predicted_discharge']
        
        # Simple weighted average (can be enhanced with more sophisticated methods)
        combined_discharge = (current_discharge * 0.6) + (projected_discharge * 0.4)
        combined_risk = classify_risk(combined_discharge)
        
        results['combined_assessment'] = {
            'discharge': combined_discharge,
            'risk_level': combined_risk,
            'confidence': 'High' if abs(current_discharge - projected_discharge) < 100 else 'Medium',
            'source': 'Combined ML + SUPARCO projection'
        }
    
    return results

def main():
    """Main prediction function"""
    try:
        # Read input data from stdin
        input_data = sys.stdin.read()
        data = json.loads(input_data)
        
        min_temp = data.get('Min Temp')
        max_temp = data.get('Max Temp')
        precipitation = data.get('Prcp')
        target_date = data.get('target_date')  # Optional future date
        
        print("🌊 SUPARCO Enhanced Flood Prediction System")
        print("=" * 50)
        
        # Get enhanced predictions
        results = enhanced_flood_prediction(min_temp, max_temp, precipitation, target_date)
        
        # Display current ML prediction
        if results['current_prediction']:
            pred = results['current_prediction']
            print(f"🤖 ML Model Prediction:")
            print(f"   Predicted Discharge: {pred['discharge']:.2f} cumecs")
            print(f"   Flood Risk: {pred['risk_level']}")
            print(f"   Source: {pred['source']}")
        
        # Display SUPARCO projection if available
        if results['climate_projection']:
            proj = results['climate_projection']
            print(f"\n🌍 SUPARCO Climate Projection:")
            print(f"   Date: {proj['date']}")
            print(f"   Projected Discharge: {proj['predicted_discharge']:.2f} cumecs")
            print(f"   Flood Risk: {proj['flood_risk']}")
            print(f"   Source: {proj['source']}")
        
        # Display combined assessment
        if results['combined_assessment']:
            combined = results['combined_assessment']
            print(f"\n🎯 Combined Assessment:")
            print(f"   Final Discharge: {combined['discharge']:.2f} cumecs")
            print(f"   Final Risk Level: {combined['risk_level']}")
            print(f"   Confidence: {combined['confidence']}")
            print(f"   Source: {combined['source']}")
        
        # Use the best available prediction for output
        if results['combined_assessment']:
            final_prediction = results['combined_assessment']
        elif results['current_prediction']:
            final_prediction = results['current_prediction']
        else:
            raise Exception("No prediction available")
        
        # Standard output format for backend compatibility
        print(f"\nPredicted Discharge: {final_prediction['discharge']:.2f} cumecs")
        print(f"Flood Risk: {final_prediction['risk_level']}")
        
        # Additional information
        print(f"\nAdditional Information:")
        print(f"- Min Temperature: {min_temp}°C")
        print(f"- Max Temperature: {max_temp}°C")
        print(f"- Precipitation: {precipitation} mm")
        
        if precipitation > 100:
            print(f"- High precipitation significantly increases flood risk")
        elif precipitation > 50:
            print(f"- Moderate precipitation may contribute to flood risk")
        else:
            print(f"- Low precipitation reduces flood risk")
        
        if target_date:
            print(f"- Forecast Date: {target_date}")
            print(f"- Climate Scenario: SSP 585 (High emissions)")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
