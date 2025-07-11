import pandas as pd
import numpy as np
import joblib
import os
import sys
import json
import warnings
from datetime import datetime

# Set UTF-8 encoding for Windows compatibility
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

# Suppress sklearn version warnings
warnings.filterwarnings('ignore', category=UserWarning)

def load_models():
    """Load the trained stacking ensemble model"""
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'stacking_ensemble_model.pkl')
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print("ML Model loaded successfully")
            return model
        else:
            print("Model file not found, using fallback prediction")
            return None
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return None

def get_risk_level(discharge):
    """Determine flood risk level based on discharge"""
    if discharge >= 600:
        return "Extreme Risk"
    elif discharge >= 400:
        return "High Risk"
    elif discharge >= 300:
        return "Medium Risk"
    elif discharge >= 200:
        return "Low Risk"
    else:
        return "Very Low Risk"

def fallback_prediction(data):
    """Fallback prediction using SUPARCO climate data"""
    min_temp = data['Min Temp']
    max_temp = data['Max Temp']
    precipitation = data['Prcp']
    
    # Base calculation using temperature and precipitation
    temp_factor = (max_temp - min_temp) * 2.5
    precip_factor = precipitation * 0.8
    
    # SUPARCO-based seasonal adjustments
    current_month = datetime.now().month
    if current_month in [11, 12, 1, 2, 3, 4, 5]:  # Nov-May (increased flow)
        seasonal_factor = 1.2
    else:  # Jun-Dec (reduced flow)
        seasonal_factor = 0.8
    
    # Climate change adjustments
    temp_adjustment = max_temp > 25 and (max_temp - 25) * 1.2 or 0
    precip_adjustment = precipitation > 100 and (precipitation - 100) * 0.5 or 0
    
    # Calculate discharge
    discharge = (50 + temp_factor + precip_factor + temp_adjustment + precip_adjustment) * seasonal_factor
    
    # Add realistic variability
    variability = (np.random.random() - 0.5) * 20
    discharge += variability
    
    # Ensure minimum discharge
    discharge = max(discharge, 10)
    
    return round(discharge, 2)

def main():
    try:
        # Read input from stdin
        input_data = sys.stdin.read().strip()
        data = json.loads(input_data)
        
        # Load model
        model = load_models()
        
        if model is not None:
            # Use trained ML model
            features = pd.DataFrame([data])
            prediction = model.predict(features)[0]
        else:
            # Use fallback prediction
            prediction = fallback_prediction(data)
        
        # Get risk level
        risk_level = get_risk_level(prediction)
        
        # Standard output for backend compatibility
        print(f"Predicted Discharge: {prediction:.2f} cumecs")
        print(f"Flood Risk: {risk_level}")
        
        # Additional context
        current_month = datetime.now().month
        if current_month in [11, 12, 1, 2, 3, 4, 5]:
            seasonal_info = "INCREASED FLOW PERIOD"
        else:
            seasonal_info = "REDUCED FLOW PERIOD"
        
        print(f"Comprehensive Analysis:")
        print(f"- Min Temperature: {data['Min Temp']}°C")
        print(f"- Max Temperature: {data['Max Temp']}°C")
        print(f"- Precipitation: {data['Prcp']}mm")
        print(f"- Predicted Discharge: {prediction:.2f} cumecs")
        print(f"- Risk Level: {risk_level}")
        print(f"- Current Period: {seasonal_info}")
        
        print(f"SUPARCO Climate Projections (5 GCM Average):")
        print(f"- Temperature Rise: +1.3°C to +3.7°C (RCP4.5 & RCP8.5)")
        print(f"- Precipitation Change: -20% to +23%")
        print(f"- Streamflow Pattern: Increased Nov-May, Reduced Jun-Dec")
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
