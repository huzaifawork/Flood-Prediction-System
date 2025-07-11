import axios from "axios";
import { FloodPredictionInput, FloodPredictionResult, HealthStatus } from "../types";

interface PredictionRequest {
  "Min Temp": number;
  "Max Temp": number;
  Prcp: number;
}

interface PredictionResponse {
  prediction: number;
  risk_level: string;
  input_data: {
    "Min Temp": number;
    "Max Temp": number;
    Prcp: number;
  };
}

// Base URL for API - using environment variable with fallback
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:5000/api";

// Health monitoring state
let lastHealthCheck: HealthStatus = {
  status: 'checking',
  message: 'Initializing...',
  lastCheck: new Date().toISOString(),
};

// Health check listeners
const healthListeners = new Set<(status: HealthStatus) => void>();

/**
 * Make a flood prediction with the provided input data
 */
export const predictFlood = async (
  input: FloodPredictionInput
): Promise<FloodPredictionResult> => {
  try {
    // Prepare request data for original Flask backend
    const requestData = {
      minTemp: input.minTemp,
      maxTemp: input.maxTemp,
      precipitation: input.precipitation,
    };

    // Make API call to Flask backend
    const response = await axios.post(`${API_BASE_URL}/predict`, requestData);

    // Transform response from original Flask backend
    return {
      discharge: response.data.discharge,
      riskLevel: response.data.riskLevel,
      confidence: response.data.confidence || 0.85,
    };
  } catch (error) {
    console.error("Error making prediction:", error);
    throw error;
  }
};

export const predictionService = {
  async predict(data: FloodPredictionInput): Promise<FloodPredictionResult> {
    try {
      // Transform the data to match Flask backend format
      const requestData = {
        minTemp: data.min_temp,
        maxTemp: data.max_temp,
        precipitation: data.precipitation,
      };

      console.log("🔬 Sending data to Flask backend:", requestData);

      // Call Flask backend predict endpoint
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestData),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Failed to make prediction");
      }

      const result = await response.json();
      console.log("🌊 Flask Backend Response:", result);

      // Transform response to match FloodPredictionResult interface
      return {
        prediction: result.discharge,
        discharge: result.discharge,
        risk_level: result.riskLevel,
        input_data: result.input,
        confidence: result.confidence || 0.85,
        timestamp: new Date().toISOString(),
        model_info: {
          type: "Actual Stacking Ensemble Model",
          algorithms: ["Random Forest", "XGBoost", "LightGBM", "Gradient Boosting"],
          training_period: "1995-2017",
          note: "Trained on actual Swat River Basin data"
        },
      };
    } catch (error) {
      console.error("Prediction error:", error);
      throw error;
    }
  },

  async checkHealth(): Promise<{ status: string; message: string }> {
    const startTime = Date.now();
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      const responseTime = Date.now() - startTime;

      if (!response.ok) {
        throw new Error("Health check failed");
      }

      const data = await response.json();

      // Update health status
      lastHealthCheck = {
        status: 'healthy',
        message: data.message || 'Backend is healthy',
        responseTime,
        lastCheck: new Date().toISOString(),
      };

      // Notify listeners
      healthListeners.forEach(listener => listener(lastHealthCheck));

      return data;
    } catch (error) {
      const responseTime = Date.now() - startTime;
      console.error("Health check error:", error);

      // Update health status
      lastHealthCheck = {
        status: 'unhealthy',
        message: error instanceof Error ? error.message : 'Unknown error',
        responseTime,
        lastCheck: new Date().toISOString(),
      };

      // Notify listeners
      healthListeners.forEach(listener => listener(lastHealthCheck));

      throw error;
    }
  },

  // Health monitoring methods
  subscribeToHealthUpdates(callback: (status: HealthStatus) => void): () => void {
    healthListeners.add(callback);
    // Send current status immediately
    callback(lastHealthCheck);

    // Return unsubscribe function
    return () => {
      healthListeners.delete(callback);
    };
  },

  getCurrentHealthStatus(): HealthStatus {
    return lastHealthCheck;
  },

  // Start automatic health monitoring
  startHealthMonitoring(intervalMs: number = 30000): () => void {
    const interval = setInterval(async () => {
      try {
        await this.checkHealth();
      } catch (error) {
        // Error is already handled in checkHealth
      }
    }, intervalMs);

    // Initial health check
    this.checkHealth().catch(() => {
      // Error is already handled in checkHealth
    });

    // Return stop function
    return () => {
      clearInterval(interval);
    };
  },
};
