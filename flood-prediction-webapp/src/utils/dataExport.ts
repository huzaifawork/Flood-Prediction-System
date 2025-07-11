// Utility function to download files without external dependencies
const downloadFile = (content: string, filename: string, mimeType: string) => {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
};

export interface ExportOptions {
  filename?: string;
  format: 'csv' | 'json' | 'xlsx';
  includeHeaders?: boolean;
  dateFormat?: 'iso' | 'local' | 'timestamp';
}

export interface ExportData {
  [key: string]: any;
}

/**
 * Convert data to CSV format
 */
export const convertToCSV = (data: ExportData[], options: ExportOptions = { format: 'csv' }): string => {
  if (!data || data.length === 0) {
    return '';
  }

  const headers = Object.keys(data[0]);
  const csvContent = [];

  // Add headers if requested
  if (options.includeHeaders !== false) {
    csvContent.push(headers.join(','));
  }

  // Add data rows
  data.forEach(row => {
    const values = headers.map(header => {
      let value = row[header];
      
      // Handle different data types
      if (value === null || value === undefined) {
        return '';
      }
      
      // Format dates
      if (value instanceof Date) {
        switch (options.dateFormat) {
          case 'timestamp':
            value = value.getTime();
            break;
          case 'local':
            value = value.toLocaleString();
            break;
          default:
            value = value.toISOString();
        }
      }
      
      // Escape commas and quotes in strings
      if (typeof value === 'string') {
        if (value.includes(',') || value.includes('"') || value.includes('\n')) {
          value = `"${value.replace(/"/g, '""')}"`;
        }
      }
      
      return value;
    });
    
    csvContent.push(values.join(','));
  });

  return csvContent.join('\n');
};

/**
 * Convert data to JSON format
 */
export const convertToJSON = (data: ExportData[], options: ExportOptions = { format: 'json' }): string => {
  const processedData = data.map(row => {
    const processedRow: ExportData = {};
    
    Object.keys(row).forEach(key => {
      let value = row[key];
      
      // Format dates
      if (value instanceof Date) {
        switch (options.dateFormat) {
          case 'timestamp':
            value = value.getTime();
            break;
          case 'local':
            value = value.toLocaleString();
            break;
          default:
            value = value.toISOString();
        }
      }
      
      processedRow[key] = value;
    });
    
    return processedRow;
  });

  return JSON.stringify(processedData, null, 2);
};

/**
 * Export flood prediction data
 */
export const exportFloodData = async (
  data: ExportData[],
  options: ExportOptions = { format: 'csv' }
) => {
  if (!data || data.length === 0) {
    throw new Error('No data to export');
  }

  const timestamp = new Date().toISOString().split('T')[0];
  const defaultFilename = `flood_data_${timestamp}`;
  const filename = options.filename || defaultFilename;

  let content: string;
  let mimeType: string;
  let fileExtension: string;

  switch (options.format) {
    case 'json':
      content = convertToJSON(data, options);
      mimeType = 'application/json';
      fileExtension = 'json';
      break;
    case 'csv':
    default:
      content = convertToCSV(data, options);
      mimeType = 'text/csv;charset=utf-8';
      fileExtension = 'csv';
      break;
  }

  downloadFile(content, `${filename}.${fileExtension}`, mimeType);
};

/**
 * Export prediction results
 */
export const exportPredictionResults = (
  predictions: any[],
  format: 'csv' | 'json' = 'csv'
) => {
  const exportData = predictions.map(prediction => ({
    timestamp: prediction.timestamp || new Date().toISOString(),
    min_temperature: prediction.input_data?.min_temp || prediction.minTemp,
    max_temperature: prediction.input_data?.max_temp || prediction.maxTemp,
    precipitation: prediction.input_data?.precipitation || prediction.precipitation,
    predicted_discharge: prediction.discharge || prediction.prediction,
    risk_level: prediction.risk_level || prediction.riskLevel,
    confidence: prediction.confidence,
    model_type: prediction.model_info?.type || 'Stacking Ensemble'
  }));

  return exportFloodData(exportData, {
    format,
    filename: `flood_predictions_${new Date().toISOString().split('T')[0]}`,
    includeHeaders: true,
    dateFormat: 'iso'
  });
};

/**
 * Export historical data
 */
export const exportHistoricalData = (
  historicalData: any[],
  format: 'csv' | 'json' = 'csv'
) => {
  const exportData = historicalData.map(record => ({
    date: record.date,
    year: record.year,
    month: record.month,
    min_temperature: record.min_temp,
    max_temperature: record.max_temp,
    precipitation: record.precipitation,
    discharge: record.discharge,
    risk_level: record.risk_level,
    event_type: record.event_type,
    description: record.description
  }));

  return exportFloodData(exportData, {
    format,
    filename: `historical_flood_data_${new Date().toISOString().split('T')[0]}`,
    includeHeaders: true,
    dateFormat: 'iso'
  });
};

/**
 * Export forecast data
 */
export const exportForecastData = (
  forecastData: any[],
  scenario: string,
  format: 'csv' | 'json' = 'csv'
) => {
  const exportData = forecastData.map(forecast => ({
    year: forecast.year,
    temperature_increase: forecast.temperature_increase,
    precipitation_change: forecast.precipitation_change,
    predicted_discharge: forecast.discharge,
    risk_level: forecast.risk_level,
    climate_scenario: scenario,
    confidence: forecast.confidence || 0.85
  }));

  return exportFloodData(exportData, {
    format,
    filename: `climate_forecast_${scenario}_${new Date().toISOString().split('T')[0]}`,
    includeHeaders: true,
    dateFormat: 'iso'
  });
};

/**
 * Export weather data
 */
export const exportWeatherData = (
  weatherData: any[],
  location: string,
  format: 'csv' | 'json' = 'csv'
) => {
  const exportData = weatherData.map(weather => ({
    timestamp: weather.timestamp || weather.time,
    location: location,
    temperature: weather.temp_c || weather.temperature,
    humidity: weather.humidity,
    precipitation: weather.precip_mm || weather.precipitation,
    wind_speed: weather.wind_kph || weather.wind_speed,
    pressure: weather.pressure_mb || weather.pressure,
    condition: weather.condition?.text || weather.condition,
    visibility: weather.vis_km || weather.visibility
  }));

  return exportFloodData(exportData, {
    format,
    filename: `weather_data_${location}_${new Date().toISOString().split('T')[0]}`,
    includeHeaders: true,
    dateFormat: 'iso'
  });
};

/**
 * Export system analytics
 */
export const exportAnalytics = (
  analyticsData: any,
  format: 'csv' | 'json' = 'json'
) => {
  // For analytics, JSON is usually more appropriate due to nested structure
  const content = JSON.stringify(analyticsData, null, 2);
  const filename = `system_analytics_${new Date().toISOString().split('T')[0]}.json`;

  downloadFile(content, filename, 'application/json');
};
