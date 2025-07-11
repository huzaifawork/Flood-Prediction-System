import { WeatherData, WeatherForecast } from '../types';

// WeatherAPI.com configuration
const API_KEY = '411cfe190e7248a48de113909250107';
const BASE_URL = 'https://api.weatherapi.com/v1';

// Swat River Basin coordinates (Mingora/Saidu Sharif)
const SWAT_COORDINATES = {
  latitude: 34.773647,
  longitude: 72.359901,
  name: 'Mingora, Swat',
  country: 'Pakistan'
};

// Cache for weather data to reduce API calls
const weatherCache = new Map<string, { data: any; timestamp: number }>();
const CACHE_DURATION = 10 * 60 * 1000; // 10 minutes

class WeatherApiService {
  private getCacheKey(lat: number, lon: number, type: string): string {
    return `weatherapi_${type}_${lat.toFixed(2)}_${lon.toFixed(2)}`;
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

  async getCurrentWeather(lat: number = SWAT_COORDINATES.latitude, lon: number = SWAT_COORDINATES.longitude): Promise<WeatherData> {
    const cacheKey = this.getCacheKey(lat, lon, 'current');
    const cached = this.getFromCache(cacheKey);
    if (cached) {
      console.log('🔄 Using cached weather data');
      return cached;
    }

    try {
      const url = `${BASE_URL}/current.json?key=${API_KEY}&q=${lat},${lon}&aqi=no`;
      console.log('🌤️ Fetching real weather data from WeatherAPI.com...', url);

      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`WeatherAPI error: ${response.status}`);
      }

      const data = await response.json();
      console.log('📊 Raw WeatherAPI.com response:', data);
      const weatherData = this.transformCurrentWeatherData(data);
      this.setCache(cacheKey, weatherData);

      console.log('✅ Real weather data loaded:', weatherData.location.name, weatherData.current);
      return weatherData;
    } catch (error) {
      console.error('❌ Error fetching weather from WeatherAPI:', error);
      throw new Error(`Failed to fetch real weather data: ${error}`);
    }
  }

  async getWeatherForecast(lat: number = SWAT_COORDINATES.latitude, lon: number = SWAT_COORDINATES.longitude, days: number = 7): Promise<WeatherForecast[]> {
    const cacheKey = this.getCacheKey(lat, lon, `forecast_${days}`);
    const cached = this.getFromCache(cacheKey);
    if (cached) {
      console.log('🔄 Using cached forecast data');
      return cached;
    }

    try {
      const url = `${BASE_URL}/forecast.json?key=${API_KEY}&q=${lat},${lon}&days=${days}&aqi=no&alerts=no`;
      console.log('📅 Fetching real weather forecast from WeatherAPI.com...');

      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`WeatherAPI forecast error: ${response.status}`);
      }

      const data = await response.json();
      const forecast = this.transformForecastData(data);
      this.setCache(cacheKey, forecast);
      
      console.log('✅ Real forecast data loaded:', forecast.length, 'days');
      return forecast;
    } catch (error) {
      console.error('❌ Error fetching forecast from WeatherAPI:', error);
      throw new Error(`Failed to fetch real forecast data: ${error}`);
    }
  }

  async getSwatWeather(): Promise<WeatherData> {
    return this.getCurrentWeather(SWAT_COORDINATES.latitude, SWAT_COORDINATES.longitude);
  }

  async getSwatForecast(): Promise<WeatherForecast[]> {
    return this.getWeatherForecast(SWAT_COORDINATES.latitude, SWAT_COORDINATES.longitude, 7);
  }

  async getWeatherByCity(cityName: string): Promise<WeatherData> {
    try {
      const url = `${BASE_URL}/current.json?key=${API_KEY}&q=${encodeURIComponent(cityName)}&aqi=no`;
      console.log(`🏙️ Fetching weather for ${cityName} from WeatherAPI.com...`);

      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`WeatherAPI city error: ${response.status}`);
      }

      const data = await response.json();
      const weatherData = this.transformCurrentWeatherData(data);
      
      console.log('✅ City weather data loaded:', weatherData.location.name);
      return weatherData;
    } catch (error) {
      console.error(`❌ Error fetching weather for ${cityName}:`, error);
      throw error;
    }
  }

  private transformCurrentWeatherData(data: any): WeatherData {
    return {
      location: {
        name: data.location.name,
        country: data.location.country,
        lat: data.location.lat,
        lon: data.location.lon,
      },
      current: {
        temp: Math.round(data.current.temp_c),
        temp_min: Math.round(data.current.temp_c - 2), // Approximate min temp
        temp_max: Math.round(data.current.temp_c + 5), // Approximate max temp
        humidity: data.current.humidity,
        pressure: data.current.pressure_mb,
        visibility: data.current.vis_km,
        wind_speed: data.current.wind_kph / 3.6, // Convert to m/s
        wind_deg: data.current.wind_degree,
        precipitation: data.current.precip_mm,
        weather: {
          main: data.current.condition.text,
          description: data.current.condition.text.toLowerCase(),
          icon: this.getWeatherIcon(data.current.condition.code, data.current.is_day),
        },
      },
    };
  }

  private transformForecastData(data: any): WeatherForecast[] {
    return data.forecast.forecastday.map((day: any) => ({
      date: day.date,
      temp_min: Math.round(day.day.mintemp_c),
      temp_max: Math.round(day.day.maxtemp_c),
      precipitation: day.day.totalprecip_mm,
      humidity: day.day.avghumidity,
      weather: {
        main: day.day.condition.text,
        description: day.day.condition.text.toLowerCase(),
        icon: this.getWeatherIcon(day.day.condition.code, 1),
      },
    }));
  }

  private getWeatherIcon(code: number, isDay: number): string {
    // WeatherAPI.com condition codes to OpenWeather icons mapping
    const iconMap: { [key: number]: { day: string; night: string } } = {
      1000: { day: '01d', night: '01n' }, // Sunny/Clear
      1003: { day: '02d', night: '02n' }, // Partly cloudy
      1006: { day: '03d', night: '03n' }, // Cloudy
      1009: { day: '04d', night: '04n' }, // Overcast
      1030: { day: '50d', night: '50n' }, // Mist
      1063: { day: '10d', night: '10n' }, // Patchy rain possible
      1066: { day: '13d', night: '13n' }, // Patchy snow possible
      1069: { day: '13d', night: '13n' }, // Patchy sleet possible
      1072: { day: '09d', night: '09n' }, // Patchy freezing drizzle possible
      1087: { day: '11d', night: '11n' }, // Thundery outbreaks possible
      1114: { day: '13d', night: '13n' }, // Blowing snow
      1117: { day: '13d', night: '13n' }, // Blizzard
      1135: { day: '50d', night: '50n' }, // Fog
      1147: { day: '50d', night: '50n' }, // Freezing fog
      1150: { day: '09d', night: '09n' }, // Patchy light drizzle
      1153: { day: '09d', night: '09n' }, // Light drizzle
      1168: { day: '09d', night: '09n' }, // Freezing drizzle
      1171: { day: '09d', night: '09n' }, // Heavy freezing drizzle
      1180: { day: '10d', night: '10n' }, // Patchy light rain
      1183: { day: '10d', night: '10n' }, // Light rain
      1186: { day: '10d', night: '10n' }, // Moderate rain at times
      1189: { day: '10d', night: '10n' }, // Moderate rain
      1192: { day: '10d', night: '10n' }, // Heavy rain at times
      1195: { day: '10d', night: '10n' }, // Heavy rain
      1198: { day: '09d', night: '09n' }, // Light freezing rain
      1201: { day: '09d', night: '09n' }, // Moderate or heavy freezing rain
      1204: { day: '13d', night: '13n' }, // Light sleet
      1207: { day: '13d', night: '13n' }, // Moderate or heavy sleet
      1210: { day: '13d', night: '13n' }, // Patchy light snow
      1213: { day: '13d', night: '13n' }, // Light snow
      1216: { day: '13d', night: '13n' }, // Patchy moderate snow
      1219: { day: '13d', night: '13n' }, // Moderate snow
      1222: { day: '13d', night: '13n' }, // Patchy heavy snow
      1225: { day: '13d', night: '13n' }, // Heavy snow
      1237: { day: '13d', night: '13n' }, // Ice pellets
      1240: { day: '09d', night: '09n' }, // Light rain shower
      1243: { day: '09d', night: '09n' }, // Moderate or heavy rain shower
      1246: { day: '09d', night: '09n' }, // Torrential rain shower
      1249: { day: '13d', night: '13n' }, // Light sleet showers
      1252: { day: '13d', night: '13n' }, // Moderate or heavy sleet showers
      1255: { day: '13d', night: '13n' }, // Light snow showers
      1258: { day: '13d', night: '13n' }, // Moderate or heavy snow showers
      1261: { day: '13d', night: '13n' }, // Light showers of ice pellets
      1264: { day: '13d', night: '13n' }, // Moderate or heavy showers of ice pellets
      1273: { day: '11d', night: '11n' }, // Patchy light rain with thunder
      1276: { day: '11d', night: '11n' }, // Moderate or heavy rain with thunder
      1279: { day: '11d', night: '11n' }, // Patchy light snow with thunder
      1282: { day: '11d', night: '11n' }, // Moderate or heavy snow with thunder
    };

    const icons = iconMap[code] || { day: '01d', night: '01n' };
    return isDay ? icons.day : icons.night;
  }

  // Mock data methods removed - using only real WeatherAPI.com data

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

export const weatherApiService = new WeatherApiService();
