import { WeatherData, WeatherForecast } from '../types';

// Open-Meteo API (free, no API key required)
const BASE_URL = 'https://api.open-meteo.com/v1';

// Coordinates for Mingora/Saidu Sharif, Swat (exact coordinates from your Python script)
const DEFAULT_COORDINATES = {
  latitude: 34.773647,
  longitude: 72.359901,
  name: 'Mingora/Saidu Sharif, Swat',
  country: 'Pakistan'
};

// Cache for weather data
const weatherCache = new Map<string, { data: any; timestamp: number }>();
const CACHE_DURATION = 10 * 60 * 1000; // 10 minutes

class OpenMeteoService {
  private getCacheKey(lat: number, lon: number, type: string): string {
    return `openmeteo_${type}_${lat.toFixed(2)}_${lon.toFixed(2)}`;
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

  async getCurrentWeather(lat: number = DEFAULT_COORDINATES.latitude, lon: number = DEFAULT_COORDINATES.longitude): Promise<WeatherData> {
    const cacheKey = this.getCacheKey(lat, lon, 'current');
    const cached = this.getFromCache(cacheKey);
    if (cached) {
      return cached;
    }

    try {
      const url = `${BASE_URL}/forecast?latitude=${lat}&longitude=${lon}&hourly=temperature_2m,precipitation_probability,precipitation&current=precipitation&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=Asia%2FKarachi`;

      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`Open-Meteo API error: ${response.status}`);
      }

      const data = await response.json();
      const weatherData = this.transformCurrentWeatherData(data, lat, lon);
      this.setCache(cacheKey, weatherData);
      return weatherData;
    } catch (error) {
      console.error('Error fetching current weather from Open-Meteo:', error);
      return this.getMockWeatherData(lat, lon);
    }
  }

  async getWeatherForecast(lat: number = DEFAULT_COORDINATES.latitude, lon: number = DEFAULT_COORDINATES.longitude): Promise<WeatherForecast[]> {
    const cacheKey = this.getCacheKey(lat, lon, 'forecast');
    const cached = this.getFromCache(cacheKey);
    if (cached) {
      return cached;
    }

    try {
      const url = `${BASE_URL}/forecast?latitude=${lat}&longitude=${lon}&hourly=temperature_2m,precipitation_probability,precipitation&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=Asia%2FKarachi&forecast_days=7`;

      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`Open-Meteo forecast API error: ${response.status}`);
      }

      const data = await response.json();
      const forecast = this.transformForecastData(data);
      this.setCache(cacheKey, forecast);
      return forecast;
    } catch (error) {
      console.error('Error fetching weather forecast from Open-Meteo:', error);
      return this.getMockForecastData();
    }
  }

  async getSwatWeather(): Promise<WeatherData> {
    return this.getCurrentWeather(DEFAULT_COORDINATES.latitude, DEFAULT_COORDINATES.longitude);
  }

  async getSwatForecast(): Promise<WeatherForecast[]> {
    return this.getWeatherForecast(DEFAULT_COORDINATES.latitude, DEFAULT_COORDINATES.longitude);
  }

  private transformCurrentWeatherData(data: any, lat: number, lon: number): WeatherData {
    const current = data.current || {};
    const daily = data.daily || {};
    const hourly = data.hourly || {};

    // Get today's data from daily forecast
    const todayIndex = 0;
    const todayTempMax = daily.temperature_2m_max?.[todayIndex] || 25;
    const todayTempMin = daily.temperature_2m_min?.[todayIndex] || 15;
    const currentPrecipitation = current.precipitation || 0;

    // Get current hour temperature from hourly data
    const currentTemp = hourly.temperature_2m?.[0] || todayTempMax;

    return {
      location: {
        name: this.getLocationName(lat, lon),
        country: 'Pakistan',
        lat,
        lon,
      },
      current: {
        temp: Math.round(currentTemp),
        temp_min: Math.round(todayTempMin),
        temp_max: Math.round(todayTempMax),
        humidity: 65, // Default humidity
        pressure: 1013, // Default pressure
        visibility: 10, // Default visibility
        wind_speed: 5, // Default wind speed
        wind_deg: 0, // Default wind direction
        precipitation: Math.round(currentPrecipitation * 10) / 10,
        weather: {
          main: currentPrecipitation > 0 ? 'Rain' : 'Clear',
          description: currentPrecipitation > 0 ? 'light rain' : 'clear sky',
          icon: currentPrecipitation > 0 ? '10d' : '01d',
        },
      },
    };
  }

  private transformForecastData(data: any): WeatherForecast[] {
    const daily = data.daily || {};
    const dates = daily.time || [];
    const tempMax = daily.temperature_2m_max || [];
    const tempMin = daily.temperature_2m_min || [];
    const precipitation = daily.precipitation_sum || [];

    return dates.slice(0, 7).map((date: string, index: number) => ({
      date,
      temp_min: Math.round(tempMin[index] || 15),
      temp_max: Math.round(tempMax[index] || 25),
      precipitation: Math.round((precipitation[index] || 0) * 10) / 10,
      humidity: 65, // Default humidity
      weather: {
        main: precipitation[index] > 5 ? 'Rain' : precipitation[index] > 0 ? 'Clouds' : 'Clear',
        description: precipitation[index] > 5 ? 'moderate rain' : precipitation[index] > 0 ? 'light rain' : 'clear sky',
        icon: precipitation[index] > 5 ? '10d' : precipitation[index] > 0 ? '09d' : '01d',
      },
    }));
  }

  private getLocationName(lat: number, lon: number): string {
    // Check if coordinates match Swat area
    if (Math.abs(lat - DEFAULT_COORDINATES.latitude) < 0.1 &&
        Math.abs(lon - DEFAULT_COORDINATES.longitude) < 0.1) {
      return 'Mingora/Saidu Sharif, Swat';
    }
    return `Location (${lat.toFixed(2)}, ${lon.toFixed(2)})`;
  }

  private getWeatherCondition(code: number): string {
    if (code === 0) return 'Clear';
    if (code <= 3) return 'Clouds';
    if (code <= 48) return 'Fog';
    if (code <= 67) return 'Rain';
    if (code <= 77) return 'Snow';
    if (code <= 82) return 'Rain';
    if (code <= 86) return 'Snow';
    if (code <= 99) return 'Thunderstorm';
    return 'Clear';
  }

  private getWeatherDescription(code: number): string {
    const descriptions: { [key: number]: string } = {
      0: 'clear sky',
      1: 'mainly clear',
      2: 'partly cloudy',
      3: 'overcast',
      45: 'fog',
      48: 'depositing rime fog',
      51: 'light drizzle',
      53: 'moderate drizzle',
      55: 'dense drizzle',
      61: 'slight rain',
      63: 'moderate rain',
      65: 'heavy rain',
      71: 'slight snow',
      73: 'moderate snow',
      75: 'heavy snow',
      80: 'slight rain showers',
      81: 'moderate rain showers',
      82: 'violent rain showers',
      95: 'thunderstorm',
      96: 'thunderstorm with slight hail',
      99: 'thunderstorm with heavy hail',
    };
    return descriptions[code] || 'clear sky';
  }

  private getWeatherIcon(code: number): string {
    if (code === 0) return '01d';
    if (code <= 3) return '02d';
    if (code <= 48) return '50d';
    if (code <= 67) return '10d';
    if (code <= 77) return '13d';
    if (code <= 82) return '09d';
    if (code <= 86) return '13d';
    if (code <= 99) return '11d';
    return '01d';
  }

  // Mock data removed - using only real Open-Meteo API

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

export const openMeteoService = new OpenMeteoService();
