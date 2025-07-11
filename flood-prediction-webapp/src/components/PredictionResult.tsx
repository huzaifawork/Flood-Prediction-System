import { motion } from "framer-motion";
import { FloodPredictionResult } from "../types";
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from "recharts";
import { useTheme } from "../context/ThemeContext";

interface PredictionResultProps {
  result: FloodPredictionResult;
}

const PredictionResult: React.FC<PredictionResultProps> = ({ result }) => {
  const { prediction, risk_level, input_data } = result;
  const { theme } = useTheme();

  // Colors for the gauge chart based on confidence
  const getConfidenceColor = () => {
    if (prediction >= 0.8) return "#22c55e"; // green
    if (prediction >= 0.6) return "#eab308"; // yellow
    return "#ef4444"; // red
  };

  // Animation variants
  const containerVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        duration: 0.5,
        when: "beforeChildren",
        staggerChildren: 0.1,
      },
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 10 },
    visible: { opacity: 1, y: 0 },
  };

  // Risk level icon and color
  const getRiskIcon = () => {
    switch (risk_level) {
      case "Low Risk":
        return (
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-5 w-5 text-green-500"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"
            />
          </svg>
        );
      case "Medium Risk":
        return (
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-5 w-5 text-yellow-500"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
            />
          </svg>
        );
      case "High Risk":
        return (
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-5 w-5 text-orange-500"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
        );
      case "Extreme Risk":
        return (
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-5 w-5 text-red-500"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
            />
          </svg>
        );
      default:
        return null;
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-white dark:bg-dark-card rounded-lg shadow-lg p-6"
    >
      <h3 className="text-xl font-semibold mb-4 text-gray-800 dark:text-gray-100">
        Prediction Results
      </h3>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <h4 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-2">
            Input Parameters
          </h4>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-300">
                Min Temperature:
              </span>
              <span className="font-medium text-gray-800 dark:text-gray-100">
                {input_data["Min Temp"]}°C
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-300">
                Max Temperature:
              </span>
              <span className="font-medium text-gray-800 dark:text-gray-100">
                {input_data["Max Temp"]}°C
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-300">
                Precipitation:
              </span>
              <span className="font-medium text-gray-800 dark:text-gray-100">
                {input_data["Prcp"]} mm
              </span>
            </div>
          </div>
        </div>

        <div>
          <h4 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-2">
            Prediction Results
          </h4>
          <div className="space-y-4">
            <div>
              <div className="flex justify-between items-center">
                <span className="text-gray-600 dark:text-gray-300">
                  Predicted Discharge:
                </span>
                <span className="font-medium text-gray-800 dark:text-gray-100">
                  {prediction.toFixed(2)} cumecs
                </span>
              </div>
              <div className="mt-2">
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                  <motion.div
                    className="bg-blue-500 h-2 rounded-full"
                    initial={{ width: 0 }}
                    animate={{
                      width: `${Math.min((prediction / 1000) * 100, 100)}%`,
                    }}
                    transition={{ duration: 0.8, ease: "easeOut" }}
                  />
                </div>
              </div>
            </div>


          </div>
        </div>
      </div>

      <div className="mt-6 pt-4 border-t border-gray-200 dark:border-gray-700">
        <p className="text-sm text-gray-600 dark:text-gray-400">
          {input_data["Prcp"] > 30 ? (
            <>
              The precipitation of {input_data["Prcp"]} mm is significantly
              high, which increases the flood risk. Combined with the
              temperature conditions, this suggests elevated flooding potential.
            </>
          ) : (
            <>
              Based on the current weather parameters, the model has calculated
              the above discharge prediction and risk assessment. Monitor local
              weather updates for any changes.
            </>
          )}
        </p>
      </div>
    </motion.div>
  );
};

export default PredictionResult;
