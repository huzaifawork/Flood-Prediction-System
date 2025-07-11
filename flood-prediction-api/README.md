# Flood Prediction API

A simple Flask API for the flood prediction machine learning model.

## Features

- **Prediction Endpoint**: Make flood discharge predictions based on weather parameters
- **Health Check**: Verify API is running properly
- **CORS Support**: Enable cross-origin requests from the web application

## Getting Started

### Prerequisites

- Python 3.7+
- pip

### Installation

1. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the API
```bash
python app.py
```

The API will be available at `http://localhost:5000`.

## API Endpoints

### Make Prediction

**URL**: `/api/predict`
**Method**: `POST`
**Content-Type**: `application/json`

**Request Body**:
```json
{
  "minTemp": 15.0,
  "maxTemp": 30.0,
  "precipitation": 50.0
}
```

**Response**:
```json
{
  "discharge": 125.5,
  "riskLevel": "Low Risk",
  "confidence": 0.85,
  "input": {
    "Min Temp": 15.0,
    "Max Temp": 30.0,
    "Prcp": 50.0
  }
}
```

### Health Check

**URL**: `/api/health`
**Method**: `GET`

**Response**:
```json
{
  "status": "ok"
}
```

## Model Integration

The API loads the stacking ensemble model from the `../models/` directory. Ensure that the model files (`stacking_model.joblib` and `scaler.joblib`) are available at this location.

## Error Handling

The API returns appropriate HTTP status codes and error messages:

- `400 Bad Request`: Missing or invalid input data
- `500 Internal Server Error`: Issues with model loading or prediction

## Deployment

For production deployment:

1. Use a production WSGI server like Gunicorn
2. Set `debug=False` in the Flask application
3. Consider containerizing the application with Docker
4. Set appropriate security headers and rate limiting 