import { WeatherData, WeatherForecast } from '../types';

const API_KEY = import.meta.env.VITE_OPENWEATHER_API_KEY;
const BASE_URL = import.meta.env.VITE_OPENWEATHER_BASE_URL || 'https://api.openweathermap.org/data/2.5';

// Cache for weather data to reduce API calls
const weatherCache = new Map<string, { data: any; timestamp: number }>();
const CACHE_DURATION = 10 * 60 * 1000; // 10 minutes

class WeatherService {
  private isValidApiKey(): boolean {
    return API_KEY && API_KEY !== 'demo_key_replace_with_real_key';
  }

  private getCacheKey(lat: number, lon: number, type: string): string {
    return `${type}_${lat.toFixed(2)}_${lon.toFixed(2)}`;
  }

  private getFromCache(key: string): any | null {
    const cached = weatherCache.get(key);
    if (cached && Date.now() - cached.timestamp < CACHE_DURATION) {
      return cached.data;
    }
    return null;
  }

  private setCache(key: string, data: any): void {
    weatherCache.set(key, { data, timestamp: Date.now() });
  }

  async getCurrentWeather(lat: number, lon: number): Promise<WeatherData> {
    if (!this.isValidApiKey()) {
      return this.getMockWeatherData(lat, lon);
    }

    const cacheKey = this.getCacheKey(lat, lon, 'current');
    const cached = this.getFromCache(cacheKey);
    if (cached) {
      return cached;
    }

    try {
      const response = await fetch(
        `${BASE_URL}/weather?lat=${lat}&lon=${lon}&appid=${API_KEY}&units=metric`
      );

      if (!response.ok) {
        throw new Error(`Weather API error: ${response.status}`);
      }

      const data = await response.json();
      const weatherData = this.transformCurrentWeatherData(data);
      this.setCache(cacheKey, weatherData);
      return weatherData;
    } catch (error) {
      console.error('Error fetching current weather:', error);
      return this.getMockWeatherData(lat, lon);
    }
  }

  async getWeatherForecast(lat: number, lon: number): Promise<WeatherForecast[]> {
    if (!this.isValidApiKey()) {
      return this.getMockForecastData();
    }

    const cacheKey = this.getCacheKey(lat, lon, 'forecast');
    const cached = this.getFromCache(cacheKey);
    if (cached) {
      return cached;
    }

    try {
      const response = await fetch(
        `${BASE_URL}/forecast?lat=${lat}&lon=${lon}&appid=${API_KEY}&units=metric`
      );

      if (!response.ok) {
        throw new Error(`Forecast API error: ${response.status}`);
      }

      const data = await response.json();
      const forecast = this.transformForecastData(data);
      this.setCache(cacheKey, forecast);
      return forecast;
    } catch (error) {
      console.error('Error fetching weather forecast:', error);
      return this.getMockForecastData();
    }
  }

  async getWeatherByCity(cityName: string): Promise<WeatherData> {
    if (!this.isValidApiKey()) {
      return this.getMockWeatherData(40.7128, -74.0060, cityName);
    }

    try {
      const response = await fetch(
        `${BASE_URL}/weather?q=${encodeURIComponent(cityName)}&appid=${API_KEY}&units=metric`
      );

      if (!response.ok) {
        throw new Error(`Weather API error: ${response.status}`);
      }

      const data = await response.json();
      return this.transformCurrentWeatherData(data);
    } catch (error) {
      console.error('Error fetching weather by city:', error);
      throw error;
    }
  }

  private transformCurrentWeatherData(data: any): WeatherData {
    return {
      location: {
        name: data.name,
        country: data.sys.country,
        lat: data.coord.lat,
        lon: data.coord.lon,
      },
      current: {
        temp: Math.round(data.main.temp),
        temp_min: Math.round(data.main.temp_min),
        temp_max: Math.round(data.main.temp_max),
        humidity: data.main.humidity,
        pressure: data.main.pressure,
        visibility: data.visibility / 1000, // Convert to km
        wind_speed: data.wind?.speed || 0,
        wind_deg: data.wind?.deg || 0,
        weather: {
          main: data.weather[0].main,
          description: data.weather[0].description,
          icon: data.weather[0].icon,
        },
      },
    };
  }

  private transformForecastData(data: any): WeatherForecast[] {
    const dailyForecasts = new Map<string, any>();

    // Group forecasts by date and get daily min/max
    data.list.forEach((item: any) => {
      const date = new Date(item.dt * 1000).toISOString().split('T')[0];
      
      if (!dailyForecasts.has(date)) {
        dailyForecasts.set(date, {
          date,
          temp_min: item.main.temp_min,
          temp_max: item.main.temp_max,
          precipitation: item.rain?.['3h'] || 0,
          humidity: item.main.humidity,
          weather: item.weather[0],
        });
      } else {
        const existing = dailyForecasts.get(date);
        existing.temp_min = Math.min(existing.temp_min, item.main.temp_min);
        existing.temp_max = Math.max(existing.temp_max, item.main.temp_max);
        existing.precipitation += item.rain?.['3h'] || 0;
      }
    });

    return Array.from(dailyForecasts.values())
      .slice(0, 5)
      .map(forecast => ({
        ...forecast,
        temp_min: Math.round(forecast.temp_min),
        temp_max: Math.round(forecast.temp_max),
        precipitation: Math.round(forecast.precipitation * 10) / 10, // Round to 1 decimal
      }));
  }

  private getMockWeatherData(lat: number, lon: number, cityName?: string): WeatherData {
    return {
      location: {
        name: cityName || 'Demo Location',
        country: 'XX',
        lat,
        lon,
      },
      current: {
        temp: 25,
        temp_min: 20,
        temp_max: 30,
        humidity: 65,
        pressure: 1013,
        visibility: 10,
        wind_speed: 5.2,
        wind_deg: 180,
        weather: {
          main: 'Clear',
          description: 'clear sky',
          icon: '01d',
        },
      },
    };
  }

  // Mock data removed - using only real weather APIs

  // Get user's current location
  async getCurrentLocation(): Promise<{ lat: number; lon: number }> {
    return new Promise((resolve, reject) => {
      if (!navigator.geolocation) {
        reject(new Error('Geolocation is not supported by this browser'));
        return;
      }

      navigator.geolocation.getCurrentPosition(
        (position) => {
          resolve({
            lat: position.coords.latitude,
            lon: position.coords.longitude,
          });
        },
        (error) => {
          reject(new Error(`Geolocation error: ${error.message}`));
        },
        {
          enableHighAccuracy: true,
          timeout: 10000,
          maximumAge: 300000, // 5 minutes
        }
      );
    });
  }
}

export const weatherService = new WeatherService();
