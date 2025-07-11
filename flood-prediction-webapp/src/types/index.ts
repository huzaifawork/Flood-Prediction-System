// Define the flood risk levels
export enum FloodRiskLevel {
  LOW = "Low Risk",
  MEDIUM = "Medium Risk",
  HIGH = "High Risk",
  EXTREME = "Extreme Risk",
}

// Input data for flood prediction
export interface FloodPredictionInput {
  min_temp: number;
  max_temp: number;
  precipitation: number;
  date?: string;
  location?: {
    lat: number;
    lon: number;
    name?: string;
  };
}

// Result of flood prediction
export interface FloodPredictionResult {
  prediction?: number;
  discharge?: number;
  risk_level?: string;
  input_data?: {
    "Min Temp": number;
    "Max Temp": number;
    Prcp: number;
  };
  confidence: number;
  timestamp: string;
  model_info?: {
    type?: string;
    algorithms?: string[];
    training_period?: string;
    accuracy?: string;
    version?: string;
    note?: string;
  };
}

// Weather API types
export interface WeatherData {
  location: {
    name: string;
    country: string;
    lat: number;
    lon: number;
  };
  current: {
    temp: number;
    temp_min: number;
    temp_max: number;
    humidity: number;
    pressure: number;
    visibility: number;
    wind_speed: number;
    wind_deg: number;
    precipitation?: number;
    weather: {
      main: string;
      description: string;
      icon: string;
    };
  };
  forecast?: WeatherForecast[];
}

export interface WeatherForecast {
  date: string;
  temp_min: number;
  temp_max: number;
  precipitation: number;
  humidity: number;
  weather: {
    main: string;
    description: string;
    icon: string;
  };
}

// Health check types
export interface HealthStatus {
  status: 'healthy' | 'unhealthy' | 'checking';
  message: string;
  responseTime?: number;
  lastCheck?: string;
  uptime?: number;
}

// Historical data point
export interface HistoricalDataPoint {
  date: string;
  minTemp: number;
  maxTemp: number;
  precipitation: number;
  discharge: number;
  riskLevel: FloodRiskLevel;
}

// Chart data for visualization
export interface ChartData {
  name: string;
  value: number;
}

// Risk threshold values - Updated 4-level system
export const RISK_THRESHOLDS = {
  LOW: 300,      // 0-299 m³/s → Low Risk
  MEDIUM: 400,   // 300-399 m³/s → Medium Risk
  HIGH: 600,     // 400-599 m³/s → High Risk
  EXTREME: 600,  // 600+ m³/s → Extreme Risk
};

// Determine risk level based on discharge value - Updated 4-level system
export const getRiskLevel = (discharge: number): FloodRiskLevel => {
  if (discharge < 300) {
    return FloodRiskLevel.LOW;        // 0-299 m³/s → Low Risk
  } else if (discharge < 400) {
    return FloodRiskLevel.MEDIUM;     // 300-399 m³/s → Medium Risk
  } else if (discharge < 600) {
    return FloodRiskLevel.HIGH;       // 400-599 m³/s → High Risk
  } else {
    return FloodRiskLevel.EXTREME;    // 600+ m³/s → Extreme Risk
  }
};

// Get color for risk level
export const getRiskColor = (riskLevel: FloodRiskLevel): string => {
  switch (riskLevel) {
    case FloodRiskLevel.LOW:
      return "risk-low";
    case FloodRiskLevel.MEDIUM:
      return "risk-medium";
    case FloodRiskLevel.HIGH:
      return "risk-high";
    case FloodRiskLevel.EXTREME:
      return "risk-extreme";
    default:
      return "bg-gray-500";
  }
};
