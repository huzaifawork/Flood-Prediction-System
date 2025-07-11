from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
from datetime import datetime

app = Flask(__name__)
CORS(app)

WEATHER_API_KEY = '411cfe190e7248a48de113909250107'
WEATHER_API_BASE_URL = 'https://api.weatherapi.com/v1'

@app.route('/api/health')
def health():
    return jsonify({"status": "healthy", "message": "API is running"})

@app.route('/api/weather/current')
def get_current_weather():
    try:
        lat = float(request.args.get('lat', 34.773647))
        lon = float(request.args.get('lon', 72.359901))
        location = request.args.get('location', 'Mingora, Pakistan')
        
        print(f"🌤️ Fetching weather from WeatherAPI.com: {lat}, {lon}")
        
        # Fetch current weather from WeatherAPI.com
        url = f"{WEATHER_API_BASE_URL}/current.json"
        params = {
            'key': WEATHER_API_KEY,
            'q': f"{lat},{lon}",
            'aqi': 'no'
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        print(f"✅ Weather data loaded: {data['location']['name']}, {data['current']['temp_c']}°C")
        
        # Transform to our format
        weather_data = {
            'location': {
                'name': data['location']['name'],
                'country': data['location']['country'],
                'lat': data['location']['lat'],
                'lon': data['location']['lon']
            },
            'current': {
                'temp': round(data['current']['temp_c']),
                'temp_min': round(data['current']['temp_c']),
                'temp_max': round(data['current']['temp_c']),
                'humidity': data['current']['humidity'],
                'pressure': data['current']['pressure_mb'],
                'visibility': data['current']['vis_km'],
                'wind_speed': round(data['current']['wind_kph'] / 3.6, 1),
                'wind_deg': data['current']['wind_degree'],
                'precipitation': data['current']['precip_mm'],
                'uv': data['current']['uv'],
                'cloud': data['current']['cloud'],
                'weather': {
                    'main': data['current']['condition']['text'],
                    'description': data['current']['condition']['text'].lower(),
                    'icon': '01d'
                }
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify({
            'success': True,
            'data': weather_data
        })
        
    except Exception as e:
        print(f"❌ Error fetching weather: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Flask app...")
    app.run(debug=True, port=5000, host='0.0.0.0')
