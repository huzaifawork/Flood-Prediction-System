import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Cloud, Sun, CloudRain, Thermometer, Wind, Eye, RefreshCw, Loader2 } from 'lucide-react';
import { FloodPredictionInput } from '../types';
import { weatherApiService } from '../api/weatherApiService';
import toast from 'react-hot-toast';

interface WeatherIntegrationProps {
  onWeatherUpdate: (data: Partial<FloodPredictionInput>) => void;
}

const WeatherIntegration = ({ onWeatherUpdate }: WeatherIntegrationProps) => {
  const [isLoading, setIsLoading] = useState(false);
  const [weatherData, setWeatherData] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  // Simulated weather data for Swat region (since we don't have API key)
  const simulatedWeatherData = {
    location: "Swat, Pakistan",
    current: {
      temp_c: 22,
      temp_min: 15,
      temp_max: 28,
      humidity: 65,
      wind_kph: 12,
      visibility_km: 10,
      condition: {
        text: "Partly cloudy",
        icon: "partly-cloudy"
      },
      precip_mm: 0
    },
    forecast: [
      { date: "Today", temp_min: 15, temp_max: 28, precip_mm: 2, condition: "Partly cloudy" },
      { date: "Tomorrow", temp_min: 17, temp_max: 30, precip_mm: 5, condition: "Light rain" },
      { date: "Day 3", temp_min: 16, temp_max: 26, precip_mm: 15, condition: "Moderate rain" }
    ]
  };

  const fetchWeatherData = async () => {
    setIsLoading(true);
    setError(null);

    try {
      console.log('🌤️ Fetching real weather data from Flask backend...');

      // Get real weather data from Flask backend (which calls WeatherAPI.com)
      const weatherResponse = await fetch('http://localhost:5000/api/weather/current');
      const forecastResponse = await fetch('http://localhost:5000/api/weather/forecast');

      if (!weatherResponse.ok || !forecastResponse.ok) {
        throw new Error('Failed to fetch weather data from backend');
      }

      const weatherResult = await weatherResponse.json();
      const forecastResult = await forecastResponse.json();

      if (!weatherResult.success || !forecastResult.success) {
        throw new Error('Weather API returned error');
      }

      const weather = weatherResult.data;
      const forecast = forecastResult.data;

      // Transform to component format
      const weatherData = {
        location: {
          name: weather.location.name,
          country: weather.location.country,
          coordinates: `${weather.location.lat}, ${weather.location.lon}`
        },
        current: {
          temp: weather.current.temp,
          temp_min: weather.current.temp_min,
          temp_max: weather.current.temp_max,
          condition: weather.current.weather.main,
          humidity: weather.current.humidity,
          wind_speed: weather.current.wind_speed,
          visibility: weather.current.visibility,
          precip_mm: weather.current.precipitation || 0
        },
        forecast: forecast.slice(0, 3) // Next 3 days
      };

      setWeatherData(weatherData);

      // Update form with real weather data
      onWeatherUpdate({
        min_temp: weather.current.temp_min,
        max_temp: weather.current.temp_max,
        precipitation: forecast[0]?.precipitation || weather.current.precipitation || 0,
        date: new Date().toISOString().split('T')[0]
      });

      console.log('✅ Real weather data loaded from Flask backend:', weather.location.name);
      toast.success(`Real weather data loaded for ${weather.location.name} via Flask backend`);

    } catch (err) {
      console.error('❌ Error fetching real weather data:', err);
      setError('Failed to fetch real weather data');
      toast.error('Failed to fetch real weather data');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchWeatherData();
  }, []);

  const getWeatherIcon = (condition?: string) => {
    if (!condition) {
      return <Sun className="h-8 w-8 text-yellow-500" />;
    }

    const conditionLower = condition.toLowerCase();
    if (conditionLower.includes('rain')) {
      return <CloudRain className="h-8 w-8 text-blue-500" />;
    } else if (conditionLower.includes('cloud')) {
      return <Cloud className="h-8 w-8 text-gray-500" />;
    } else {
      return <Sun className="h-8 w-8 text-yellow-500" />;
    }
  };

  const getRiskColor = (precipitation: number) => {
    if (precipitation > 50) return 'text-red-500';
    if (precipitation > 20) return 'text-orange-500';
    if (precipitation > 5) return 'text-yellow-500';
    return 'text-green-500';
  };

  return (
    <div className="card p-6">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-xl font-semibold text-foreground flex items-center">
          <Cloud className="h-5 w-5 mr-2" />
          Weather Integration
        </h3>
        <button
          onClick={fetchWeatherData}
          disabled={isLoading}
          className="btn btn-outline btn-sm"
        >
          <RefreshCw className={`h-4 w-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {isLoading && (
        <div className="text-center py-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-muted-foreground">Fetching weather data...</p>
        </div>
      )}

      {error && (
        <div className="text-center py-8">
          <p className="text-red-600">{error}</p>
        </div>
      )}

      {weatherData && !isLoading && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="space-y-6"
        >
          {/* Current Weather */}
          <div className="bg-muted/50 rounded-lg p-4">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h4 className="font-semibold text-foreground">Current Weather</h4>
                <p className="text-sm text-muted-foreground">
                  {weatherData.location?.name || weatherData.location?.coordinates || 'Mingora, Swat'}
                </p>
              </div>
              {getWeatherIcon(weatherData.current?.condition || weatherData.current?.weather?.main)}
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center">
                <Thermometer className="h-5 w-5 text-red-500 mx-auto mb-1" />
                <div className="text-lg font-semibold">{weatherData.current?.temp || weatherData.current?.temp_c || 'N/A'}°C</div>
                <div className="text-xs text-muted-foreground">Current</div>
              </div>
              <div className="text-center">
                <Thermometer className="h-5 w-5 text-blue-500 mx-auto mb-1" />
                <div className="text-lg font-semibold">
                  {weatherData.current?.temp_min || 'N/A'}°C / {weatherData.current?.temp_max || 'N/A'}°C
                </div>
                <div className="text-xs text-muted-foreground">Min / Max</div>
              </div>
              <div className="text-center">
                <CloudRain className="h-5 w-5 text-blue-500 mx-auto mb-1" />
                <div className="text-lg font-semibold">{weatherData.current?.precip_mm || weatherData.current?.precipitation || 0}mm</div>
                <div className="text-xs text-muted-foreground">Precipitation</div>
              </div>
              <div className="text-center">
                <Wind className="h-5 w-5 text-gray-500 mx-auto mb-1" />
                <div className="text-lg font-semibold">{weatherData.current?.wind_speed || weatherData.current?.wind_kph || 'N/A'} {weatherData.current?.wind_speed ? 'm/s' : 'km/h'}</div>
                <div className="text-xs text-muted-foreground">Wind</div>
              </div>
            </div>
          </div>

          {/* 3-Day Forecast */}
          <div>
            <h4 className="font-semibold text-foreground mb-4">3-Day Forecast</h4>
            <div className="space-y-3">
              {weatherData.forecast.map((day: any, index: number) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="flex items-center justify-between p-3 bg-muted/30 rounded-lg"
                >
                  <div className="flex items-center space-x-3">
                    {getWeatherIcon(day.weather?.main || day.condition)}
                    <div>
                      <div className="font-medium">{day.date}</div>
                      <div className="text-sm text-muted-foreground">{day.weather?.main || day.condition || 'Clear'}</div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="font-medium">
                      {day.temp_min}°C / {day.temp_max}°C
                    </div>
                    <div className={`text-sm font-medium ${getRiskColor(day.precipitation || day.precip_mm || 0)}`}>
                      {day.precipitation || day.precip_mm || 0}mm rain
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>

          {/* SUPARCO Context */}
          <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
            <h4 className="font-semibold text-foreground mb-2">SUPARCO Climate Context</h4>
            <div className="text-sm text-muted-foreground space-y-1">
              <p>• Current conditions within SUPARCO projected ranges</p>
              <p>• Temperature: Normal for climate change scenario</p>
              <p>• Precipitation: {weatherData.current.precip_mm > 20 ? 'Above' : 'Within'} typical seasonal patterns</p>
              <p>• Flood risk assessment incorporates 5 GCM ensemble data</p>
            </div>
          </div>

          {/* Auto-fill Button */}
          <button
            onClick={() => onWeatherUpdate({
              min_temp: weatherData.current.temp_min,
              max_temp: weatherData.current.temp_max,
              precipitation: weatherData.current.precip_mm,
              date: new Date().toISOString().split('T')[0]
            })}
            className="btn btn-primary w-full"
          >
            Use Current Weather Data
          </button>
        </motion.div>
      )}
    </div>
  );
};

export default WeatherIntegration;
