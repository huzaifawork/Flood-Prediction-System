import { motion } from 'framer-motion';
import { Cloud, CloudRain, Sun, CloudSnow, Zap, Droplets } from 'lucide-react';
import { WeatherForecast as WeatherForecastType } from '../types';

interface WeatherForecastProps {
  forecast: WeatherForecastType[];
  className?: string;
}

const WeatherForecast: React.FC<WeatherForecastProps> = ({
  forecast,
  className = ''
}) => {
  const getWeatherIcon = (weatherMain: string, size: number = 20) => {
    const iconProps = { size };

    switch (weatherMain.toLowerCase()) {
      case 'clear':
        return <Sun {...iconProps} className="text-yellow-500" />;
      case 'clouds':
        return <Cloud {...iconProps} className="text-gray-500" />;
      case 'rain':
        return <CloudRain {...iconProps} className="text-blue-500" />;
      case 'snow':
        return <CloudSnow {...iconProps} className="text-blue-200" />;
      case 'thunderstorm':
        return <Zap {...iconProps} className="text-purple-500" />;
      default:
        return <Cloud {...iconProps} />;
    }
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    const today = new Date();
    const tomorrow = new Date(today);
    tomorrow.setDate(today.getDate() + 1);

    if (date.toDateString() === today.toDateString()) {
      return 'Today';
    } else if (date.toDateString() === tomorrow.toDateString()) {
      return 'Tomorrow';
    } else {
      return date.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' });
    }
  };

  const getTemperatureColor = (temp: number) => {
    if (temp < 0) return 'text-blue-600';
    if (temp < 10) return 'text-blue-500';
    if (temp < 20) return 'text-green-500';
    if (temp < 30) return 'text-yellow-500';
    if (temp < 40) return 'text-orange-500';
    return 'text-red-500';
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className={`bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700 ${className}`}
    >
      <h3 className="text-lg font-semibold text-gray-800 dark:text-white mb-4">
        5-Day Forecast
      </h3>

      <div className="space-y-3">
        {forecast.map((day, index) => (
          <motion.div
            key={day.date}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.1 }}
            className="flex items-center justify-between p-3 rounded-lg bg-gray-50 dark:bg-gray-700 hover:bg-gray-100 dark:hover:bg-gray-600 transition-colors duration-200"
          >
            {/* Date */}
            <div className="flex-1">
              <p className="text-sm font-medium text-gray-800 dark:text-white">
                {formatDate(day.date)}
              </p>
              <p className="text-xs text-gray-500 dark:text-gray-400 capitalize">
                {day.weather.description}
              </p>
            </div>

            {/* Weather Icon */}
            <div className="flex-shrink-0 mx-4">
              <motion.div
                whileHover={{ scale: 1.1 }}
                transition={{ type: "spring", stiffness: 400 }}
              >
                {getWeatherIcon(day.weather.main)}
              </motion.div>
            </div>

            {/* Precipitation */}
            <div className="flex items-center space-x-1 flex-shrink-0 mx-2">
              <Droplets className="w-3 h-3 text-blue-500" />
              <span className="text-xs text-gray-600 dark:text-gray-400">
                {day.precipitation.toFixed(1)}mm
              </span>
            </div>

            {/* Temperature Range */}
            <div className="flex items-center space-x-2 flex-shrink-0">
              <span className={`text-sm font-medium ${getTemperatureColor(day.temp_max)}`}>
                {Math.round(day.temp_max)}°
              </span>
              <span className="text-gray-400">/</span>
              <span className={`text-sm ${getTemperatureColor(day.temp_min)}`}>
                {Math.round(day.temp_min)}°
              </span>
            </div>

            {/* Humidity */}
            <div className="flex-shrink-0 ml-4">
              <span className="text-xs text-gray-500 dark:text-gray-400">
                {Math.round(day.humidity)}%
              </span>
            </div>
          </motion.div>
        ))}
      </div>

      {/* Summary */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.6 }}
        className="mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800"
      >
        <p className="text-sm text-blue-800 dark:text-blue-200">
          <strong>Forecast Summary:</strong> Average precipitation of{' '}
          {(forecast.reduce((sum, day) => sum + day.precipitation, 0) / forecast.length).toFixed(1)}mm
          over the next 5 days. Temperature range from{' '}
          {Math.min(...forecast.map(day => day.temp_min))}° to{' '}
          {Math.max(...forecast.map(day => day.temp_max))}°C.
        </p>
      </motion.div>
    </motion.div>
  );
};

export default WeatherForecast;
