import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { MapPin, Loader2, RefreshCw, AlertCircle, Mountain } from 'lucide-react';
import { weatherService } from '../api/weatherService';
import { openMeteoService } from '../api/openMeteoService';
import { weatherApiService } from '../api/weatherApiService';
import { WeatherData, FloodPredictionInput } from '../types';
import WeatherCard from './WeatherCard';
import WeatherForecast from './WeatherForecast';
import LoadingSkeleton from './LoadingSkeleton';
import toast from 'react-hot-toast';

interface WeatherIntegrationPanelProps {
  onWeatherDataSelect: (data: Partial<FloodPredictionInput>) => void;
  className?: string;
}

const WeatherIntegrationPanel: React.FC<WeatherIntegrationPanelProps> = ({
  onWeatherDataSelect,
  className = '',
}) => {
  const [weatherData, setWeatherData] = useState<WeatherData | null>(null);
  const [forecast, setForecast] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [locationInput, setLocationInput] = useState('');
  const [isExpanded, setIsExpanded] = useState(false);

  // Auto-fetch weather data on component mount
  useEffect(() => {
    // Default to Swat weather using WeatherAPI.com with real API key
    fetchSwatWeather();
  }, []);

  const fetchCurrentLocationWeather = async () => {
    setIsLoading(true);
    setError(null);

    try {
      // Use Flask backend for real weather data (which calls WeatherAPI.com)
      const weatherResponse = await fetch('http://localhost:5000/api/weather/current');
      const forecastResponse = await fetch('http://localhost:5000/api/weather/forecast');

      if (!weatherResponse.ok || !forecastResponse.ok) {
        throw new Error('Failed to fetch weather data from Flask backend');
      }

      const weatherResult = await weatherResponse.json();
      const forecastResult = await forecastResponse.json();

      if (!weatherResult.success || !forecastResult.success) {
        throw new Error('Weather API returned error');
      }

      // Transform Flask backend response to component format
      const weather = {
        location: weatherResult.data.location,
        current: weatherResult.data.current
      };

      setWeatherData(weather);
      setForecast(forecastResult.data);
      toast.success(`Real weather data loaded for ${weather.location.name} via Flask backend`);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to fetch weather data';
      setError(errorMessage);
      toast.error(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  const fetchWeatherByCity = async () => {
    if (!locationInput.trim()) {
      toast.error('Please enter a city name');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      // Use Flask backend for city weather lookup
      const weatherResponse = await fetch(`http://localhost:5000/api/weather/current?q=${encodeURIComponent(locationInput)}`);

      if (!weatherResponse.ok) {
        throw new Error('Failed to fetch city weather data');
      }

      const weatherResult = await weatherResponse.json();

      if (!weatherResult.success) {
        throw new Error('Weather API returned error for city');
      }

      const weather = {
        location: weatherResult.data.location,
        current: weatherResult.data.current
      };

      // Get forecast for this location
      const forecastResponse = await fetch(`http://localhost:5000/api/weather/forecast?lat=${weather.location.lat}&lon=${weather.location.lon}`);
      const forecastResult = await forecastResponse.json();

      setWeatherData(weather);
      setForecast(forecastResult.success ? forecastResult.data : []);
      toast.success(`Real weather data loaded for ${weather.location.name} via Flask backend`);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to fetch weather data';
      setError(errorMessage);
      toast.error(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  const fetchSwatWeather = async () => {
    setIsLoading(true);
    setError(null);

    try {
      // Use Flask backend for Swat weather (same as current location)
      const weatherResponse = await fetch('http://localhost:5000/api/weather/current');
      const forecastResponse = await fetch('http://localhost:5000/api/weather/forecast');

      if (!weatherResponse.ok || !forecastResponse.ok) {
        throw new Error('Failed to fetch Swat weather data from Flask backend');
      }

      const weatherResult = await weatherResponse.json();
      const forecastResult = await forecastResponse.json();

      if (!weatherResult.success || !forecastResult.success) {
        throw new Error('Weather API returned error for Swat');
      }

      const weather = {
        location: weatherResult.data.location,
        current: weatherResult.data.current
      };

      setWeatherData(weather);
      setForecast(forecastResult.data);
      toast.success(`Latest real weather data loaded for ${weather.location.name} via Flask backend`);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to fetch Swat weather data';
      setError(errorMessage);
      toast.error(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  const handleUseWeatherData = () => {
    if (!weatherData) return;

    const predictionData: Partial<FloodPredictionInput> = {
      min_temp: weatherData.current.temp_min,
      max_temp: weatherData.current.temp_max,
      precipitation: 0, // Current weather doesn't include precipitation, use forecast
      location: {
        lat: weatherData.location.lat,
        lon: weatherData.location.lon,
        name: weatherData.location.name,
      },
    };

    // If we have forecast data, use the first day's precipitation
    if (forecast.length > 0) {
      predictionData.precipitation = forecast[0].precipitation;
    }

    onWeatherDataSelect(predictionData);
    toast.success('Weather data applied to prediction form');
  };

  const handleRefresh = () => {
    if (weatherData) {
      fetchCurrentLocationWeather();
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className={`bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 ${className}`}
    >
      {/* Header */}
      <div className="p-4 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold text-gray-800 dark:text-white">
            Weather Integration
          </h3>
          <div className="flex items-center space-x-2">
            {weatherData && (
              <motion.button
                onClick={handleRefresh}
                disabled={isLoading}
                className="p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 transition-colors duration-200"
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
              >
                <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
              </motion.button>
            )}
            <button
              onClick={() => setIsExpanded(!isExpanded)}
              className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 transition-colors duration-200"
            >
              <motion.div
                animate={{ rotate: isExpanded ? 180 : 0 }}
                transition={{ duration: 0.2 }}
              >
                ▼
              </motion.div>
            </button>
          </div>
        </div>

        {/* Location Input */}
        <div className="mt-4 space-y-3">
          <div className="flex space-x-2">
            <div className="flex-1">
              <input
                type="text"
                value={locationInput}
                onChange={(e) => setLocationInput(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && fetchWeatherByCity()}
                placeholder="Enter city name..."
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-800 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-colors duration-200"
              />
            </div>
            <motion.button
              onClick={fetchWeatherByCity}
              disabled={isLoading}
              className="px-4 py-2 bg-blue-500 hover:bg-blue-600 disabled:bg-gray-400 text-white rounded-lg transition-colors duration-200 flex items-center space-x-2"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              {isLoading ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <MapPin className="w-4 h-4" />
              )}
              <span>Search</span>
            </motion.button>
          </div>

          {/* Quick Action Buttons */}
          <div className="flex space-x-2">
            <motion.button
              onClick={fetchSwatWeather}
              disabled={isLoading}
              className="flex-1 px-4 py-2 bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-600 hover:to-emerald-700 disabled:bg-gray-400 text-white rounded-lg transition-all duration-200 flex items-center justify-center space-x-2"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <Mountain className="w-4 h-4" />
              <span>Swat Weather (Free API)</span>
            </motion.button>
            <motion.button
              onClick={fetchCurrentLocationWeather}
              disabled={isLoading}
              className="px-4 py-2 bg-blue-500 hover:bg-blue-600 disabled:bg-gray-400 text-white rounded-lg transition-colors duration-200 flex items-center space-x-2"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <MapPin className="w-4 h-4" />
              <span>Current Location</span>
            </motion.button>
          </div>
        </div>
      </div>

      {/* Content */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.3 }}
            className="p-4 overflow-hidden"
          >
            {isLoading && (
              <div className="space-y-4">
                <LoadingSkeleton variant="weather" />
              </div>
            )}

            {error && (
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                className="flex items-center space-x-2 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg"
              >
                <AlertCircle className="w-5 h-5 text-red-500" />
                <p className="text-red-700 dark:text-red-300">{error}</p>
              </motion.div>
            )}

            {weatherData && !isLoading && (
              <div className="space-y-6">
                <WeatherCard
                  weather={weatherData}
                  onUseWeatherData={handleUseWeatherData}
                />

                {forecast.length > 0 && (
                  <WeatherForecast forecast={forecast} />
                )}
              </div>
            )}

            {!weatherData && !isLoading && !error && (
              <div className="text-center py-8">
                <MapPin className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                <p className="text-gray-500 dark:text-gray-400">
                  Search for a city or use your current location to get weather data
                </p>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

export default WeatherIntegrationPanel;
