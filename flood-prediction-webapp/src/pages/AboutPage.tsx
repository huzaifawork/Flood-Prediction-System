import { motion } from 'framer-motion';

const AboutPage = () => {
  return (
    <div className="max-w-4xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <h1 className="text-3xl font-bold mb-2">About the Flood Prediction System</h1>
        <p className="text-gray-600 mb-8">
          Learn about our advanced machine learning model for flood discharge prediction.
        </p>
      </motion.div>

      <div className="space-y-8">
        <motion.div
          className="card"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
        >
          <h2 className="text-xl font-semibold mb-4">Project Overview</h2>
          <p className="mb-4">
            This flood prediction system implements a machine learning model for predicting flood discharge based on 
            meteorological data. The model uses features like temperature and precipitation to predict river discharge 
            in cumecs (cubic meters per second).
          </p>
          <p>
            The system is designed to provide early warnings and risk assessments to help communities prepare for 
            potential flooding events. By analyzing weather parameters, our model can accurately predict discharge 
            levels and categorize them into different risk levels.
          </p>
        </motion.div>

        <motion.div
          className="card"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
        >
          <h2 className="text-xl font-semibold mb-4">The Dataset</h2>
          <p className="mb-4">
            The model was trained on the <code>Merged_Weather_Flow_Final_1995_2017.xlsx</code> dataset, which contains 
            historical weather and river discharge data from 1995 to 2017. The dataset includes the following key features:
          </p>
          <ul className="list-disc pl-6 mb-4 space-y-2">
            <li><strong>Date:</strong> Date of the observation</li>
            <li><strong>Min Temp:</strong> Minimum temperature for the day</li>
            <li><strong>Max Temp:</strong> Maximum temperature for the day</li>
            <li><strong>Prcp:</strong> Precipitation amount</li>
            <li><strong>Discharge (cumecs):</strong> River discharge in cubic meters per second (target variable)</li>
          </ul>
        </motion.div>

        <motion.div
          className="card"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4 }}
        >
          <h2 className="text-xl font-semibold mb-4">The Model</h2>
          <p className="mb-4">
            Our system uses a stacking ensemble model that combines multiple machine learning algorithms to achieve 
            higher accuracy than any single model. The ensemble includes:
          </p>
          <ul className="list-disc pl-6 mb-4 space-y-2">
            <li><strong>Random Forest:</strong> An ensemble of decision trees that excels at capturing non-linear relationships</li>
            <li><strong>Gradient Boosting:</strong> A powerful boosting algorithm that builds trees sequentially</li>
            <li><strong>XGBoost:</strong> An optimized implementation of gradient boosting known for its performance</li>
            <li><strong>LightGBM:</strong> A gradient boosting framework that uses tree-based learning algorithms</li>
          </ul>
          <p>
            These models are combined using a meta-learner (Ridge regression) that weights their predictions to produce 
            a final, more accurate prediction.
          </p>
        </motion.div>

        <motion.div
          className="card"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
        >
          <h2 className="text-xl font-semibold mb-4">Risk Assessment</h2>
          <p className="mb-4">
            The system categorizes flood risk into four levels based on the predicted discharge:
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-risk-low/10 p-4 rounded-lg border border-risk-low">
              <h3 className="font-semibold text-risk-low mb-2">Low Risk (&lt; 200 cumecs)</h3>
              <p className="text-sm">Normal river conditions with minimal flooding concerns. Regular monitoring is sufficient.</p>
            </div>
            <div className="bg-risk-medium/10 p-4 rounded-lg border border-risk-medium">
              <h3 className="font-semibold text-risk-medium mb-2">Medium Risk (200-400 cumecs)</h3>
              <p className="text-sm">Elevated river levels with potential for minor flooding. Increased vigilance recommended.</p>
            </div>
            <div className="bg-risk-high/10 p-4 rounded-lg border border-risk-high">
              <h3 className="font-semibold text-risk-high mb-2">High Risk (400-600 cumecs)</h3>
              <p className="text-sm">Significant flooding likely. Preparedness measures should be implemented.</p>
            </div>
            <div className="bg-risk-extreme/10 p-4 rounded-lg border border-risk-extreme">
              <h3 className="font-semibold text-risk-extreme mb-2">Extreme Risk (&gt; 600 cumecs)</h3>
              <p className="text-sm">Severe flooding expected. Immediate action and evacuation plans may be necessary.</p>
            </div>
          </div>
        </motion.div>

        <motion.div
          className="card"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.6 }}
        >
          <h2 className="text-xl font-semibold mb-4">Future Improvements</h2>
          <p className="mb-4">
            We are continuously working to improve the flood prediction system. Future enhancements include:
          </p>
          <ul className="list-disc pl-6 mb-4 space-y-2">
            <li>Incorporating additional weather features like humidity, wind speed, etc.</li>
            <li>Exploring time-series models to better capture temporal dependencies</li>
            <li>Implementing an early warning system based on prediction thresholds</li>
            <li>Creating an interactive dashboard for visualizing predictions</li>
            <li>Integrating with real-time weather data sources for automated predictions</li>
          </ul>
        </motion.div>
      </div>
    </div>
  );
};

export default AboutPage; 