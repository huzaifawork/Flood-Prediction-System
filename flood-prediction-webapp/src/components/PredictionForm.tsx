import { useState, FormEvent, ChangeEvent, useEffect } from "react";
import { motion } from "framer-motion";
import { FloodPredictionInput } from "../types";
import { useTheme } from "../context/ThemeContext";
import EnhancedCard from "./EnhancedCard";
import LazyWrapper from "./LazyWrapper";
import { predictionService } from "../api/predictionService";

interface PredictionFormProps {
  onSubmit: (data: FloodPredictionInput) => void;
  isLoading: boolean;
  initialData?: FloodPredictionInput;
}

const DEFAULT_FORM: FloodPredictionInput = {
  min_temp: 15,
  max_temp: 30,
  precipitation: 50,
  date: new Date().toISOString().split("T")[0],
};

const TABS = [
  { key: "manual", label: "Manual Entry" },
  { key: "json", label: "Paste JSON" },
  { key: "csv", label: "Paste CSV/TXT" },
  { key: "file", label: "Upload File" },
];

const PredictionForm: React.FC<PredictionFormProps> = ({
  onSubmit,
  isLoading,
  initialData,
}) => {
  const { theme } = useTheme();
  const [tab, setTab] = useState<"manual" | "json" | "csv" | "file">("manual");
  const [formData, setFormData] = useState<FloodPredictionInput>(initialData || DEFAULT_FORM);
  const [focusedField, setFocusedField] = useState<string | null>(null);
  const [rawInput, setRawInput] = useState("");
  const [fileName, setFileName] = useState("");
  const [parseError, setParseError] = useState<string | null>(null);
  const [parsedData, setParsedData] = useState<FloodPredictionInput | null>(
    null
  );

  // Update form data when initialData changes
  useEffect(() => {
    if (initialData && tab === "manual") {
      setFormData(initialData);
    }
  }, [initialData, tab]);

  // --- Parsing helpers ---
  function parseJSON(input: string): FloodPredictionInput | null {
    try {
      const obj = JSON.parse(input);
      if (Array.isArray(obj)) {
        return obj[0];
      }
      return obj;
    } catch {
      return null;
    }
  }

  function parseCSV(input: string): FloodPredictionInput | null {
    // Accepts header or no header, comma or tab separated
    const lines = input.trim().split(/\r?\n/).filter(Boolean);
    if (!lines.length) return null;
    let header: string[] = [];
    let dataLine: string[] = [];
    if (lines[0].toLowerCase().includes("mintemp")) {
      header = lines[0].split(/,|\t/).map((s) => s.trim());
      dataLine = lines[1]?.split(/,|\t/).map((s) => s.trim()) || [];
    } else {
      // Assume order: minTemp,maxTemp,precipitation,date
      header = ["minTemp", "maxTemp", "precipitation", "date"];
      dataLine = lines[0].split(/,|\t/).map((s) => s.trim());
    }
    if (dataLine.length < 3) return null;
    const obj: any = {};
    header.forEach((h, i) => {
      obj[h] = dataLine[i];
    });
    // Convert types
    return {
      minTemp: parseFloat(obj.minTemp),
      maxTemp: parseFloat(obj.maxTemp),
      precipitation: parseFloat(obj.precipitation),
      date: obj.date || new Date().toISOString().split("T")[0],
    };
  }

  function validateInput(data: any): data is FloodPredictionInput {
    return (
      typeof data === "object" &&
      typeof data.minTemp === "number" &&
      typeof data.maxTemp === "number" &&
      typeof data.precipitation === "number" &&
      typeof data.date === "string"
    );
  }

  // --- Handlers ---
  const handleTabChange = (key: typeof tab) => {
    setTab(key);
    setRawInput("");
    setParseError(null);
    setParsedData(null);
    setFileName("");
    if (key === "manual") setFormData(DEFAULT_FORM);
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: name === "date" ? value : parseFloat(value),
    });
  };

  const handleRawInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setRawInput(e.target.value);
    setParseError(null);
    setParsedData(null);
  };

  const handleFile = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setFileName(file.name);
    setParseError(null);
    setParsedData(null);
    const reader = new FileReader();
    reader.onload = (ev) => {
      const text = ev.target?.result as string;
      setRawInput(text);
      // Try to auto-detect format
      let data: FloodPredictionInput | null = null;
      if (file.name.endsWith(".json")) {
        data = parseJSON(text);
      } else if (file.name.endsWith(".csv") || file.name.endsWith(".txt")) {
        data = parseCSV(text);
      }
      if (data && validateInput(data)) {
        setParsedData(data);
      } else {
        setParseError("Could not parse file. Please check the format.");
      }
    };
    reader.readAsText(file);
  };

  // --- Submission ---
  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setParseError(null);

    try {
      let dataToSubmit: FloodPredictionInput;

      if (tab === "manual") {
        dataToSubmit = formData;
      } else if (tab === "json") {
        const data = parseJSON(rawInput);
        if (!data || !validateInput(data)) {
          setParseError("Invalid JSON format.");
          return;
        }
        dataToSubmit = data;
      } else if (tab === "csv") {
        const data = parseCSV(rawInput);
        if (!data || !validateInput(data)) {
          setParseError("Invalid CSV/TXT format.");
          return;
        }
        dataToSubmit = data;
      } else if (tab === "file") {
        if (!parsedData || !validateInput(parsedData)) {
          setParseError("No valid data parsed from file.");
          return;
        }
        dataToSubmit = parsedData;
      } else {
        return;
      }

      onSubmit(dataToSubmit);
    } catch (error) {
      console.error("Error submitting prediction:", error);
      setParseError("An error occurred while submitting the prediction.");
    }
  };

  const handleFocus = (field: string) => setFocusedField(field);
  const handleBlur = () => setFocusedField(null);

  // --- UI ---
  return (
    <LazyWrapper
      animation="scale"
      delay={300}
      showProgress={true}
    >
      <EnhancedCard
        variant="glass"
        hover="lift"
        animation="none"
        showParticles={false}
        rippleEffect={true}
        className="p-1"
      >
        <h2 className="text-xl font-semibold mb-6 text-light-text-primary dark:text-dark-text-primary flex items-center">
        <svg
          xmlns="http://www.w3.org/2000/svg"
          className="h-6 w-6 mr-2 text-primary-500 dark:text-primary-400"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"
          />
        </svg>
        Enter Weather Parameters
      </h2>
      {/* Tabs */}
      <div className="flex space-x-2 mb-6">
        {TABS.map((t) => (
          <button
            key={t.key}
            type="button"
            className={`btn btn-sm ${
              tab === t.key ? "btn-primary" : "btn-secondary"
            } transition-all`}
            onClick={() => handleTabChange(t.key as any)}
          >
            {t.label}
          </button>
        ))}
      </div>
      <form onSubmit={handleSubmit}>
        {/* Manual Entry */}
        {tab === "manual" && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.3 }}
              className={`relative ${
                focusedField === "min_temp" ? "scale-[1.02]" : ""
              } transition-transform duration-200`}
            >
              <label htmlFor="min_temp" className="label">
                Minimum Temperature (°C)
              </label>
              <div className="relative">
                <input
                  type="number"
                  id="min_temp"
                  name="min_temp"
                  value={formData.min_temp}
                  onChange={handleChange}
                  onFocus={() => handleFocus("min_temp")}
                  onBlur={handleBlur}
                  className="input input-animated pl-9"
                  min="-50"
                  max="60"
                  step="0.1"
                  required
                />
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="h-5 w-5 text-gray-400 dark:text-gray-500"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M15 13l-3 3m0 0l-3-3m3 3V8m0 13a9 9 0 110-18 9 9 0 010 18z"
                    />
                  </svg>
                </div>
                {focusedField === "min_temp" && (
                  <motion.div
                    className="absolute inset-0 rounded-md ring-2 ring-primary-500 dark:ring-primary-400 pointer-events-none"
                    layoutId="inputFocus"
                  />
                )}
              </div>
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                Lowest temperature of the day
              </p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.3 }}
              className={`relative ${
                focusedField === "max_temp" ? "scale-[1.02]" : ""
              } transition-transform duration-200`}
            >
              <label htmlFor="max_temp" className="label">
                Maximum Temperature (°C)
              </label>
              <div className="relative">
                <input
                  type="number"
                  id="max_temp"
                  name="max_temp"
                  value={formData.max_temp}
                  onChange={handleChange}
                  onFocus={() => handleFocus("max_temp")}
                  onBlur={handleBlur}
                  className="input input-animated pl-9"
                  min="-50"
                  max="60"
                  step="0.1"
                  required
                />
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="h-5 w-5 text-gray-400 dark:text-gray-500"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9 11l3-3m0 0l3 3m-3-3v8m0-13a9 9 0 110 18 9 9 0 010-18z"
                    />
                  </svg>
                </div>
                {focusedField === "max_temp" && (
                  <motion.div
                    className="absolute inset-0 rounded-md ring-2 ring-primary-500 dark:ring-primary-400 pointer-events-none"
                    layoutId="inputFocus"
                  />
                )}
              </div>
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                Highest temperature of the day
              </p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
              className={`relative ${
                focusedField === "precipitation" ? "scale-[1.02]" : ""
              } transition-transform duration-200`}
            >
              <label htmlFor="precipitation" className="label">
                Precipitation (mm)
              </label>
              <div className="relative">
                <input
                  type="number"
                  id="precipitation"
                  name="precipitation"
                  value={formData.precipitation}
                  onChange={handleChange}
                  onFocus={() => handleFocus("precipitation")}
                  onBlur={handleBlur}
                  className="input input-animated pl-9"
                  min="0"
                  max="2000"
                  step="0.1"
                  required
                />
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="h-5 w-5 text-gray-400 dark:text-gray-500"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M19 14l-7 7m0 0l-7-7m7 7V3"
                    />
                  </svg>
                </div>
                {focusedField === "precipitation" && (
                  <motion.div
                    className="absolute inset-0 rounded-md ring-2 ring-primary-500 dark:ring-primary-400 pointer-events-none"
                    layoutId="inputFocus"
                  />
                )}
              </div>
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                Total rainfall in millimeters
              </p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.3, delay: 0.3 }}
              className={`relative ${
                focusedField === "date" ? "scale-[1.02]" : ""
              } transition-transform duration-200`}
            >
              <label htmlFor="date" className="label">
                Date (Optional)
              </label>
              <div className="relative">
                <input
                  type="date"
                  id="date"
                  name="date"
                  value={formData.date}
                  onChange={handleChange}
                  onFocus={() => handleFocus("date")}
                  onBlur={handleBlur}
                  className="input input-animated pl-9"
                />
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="h-5 w-5 text-gray-400 dark:text-gray-500"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"
                    />
                  </svg>
                </div>
                {focusedField === "date" && (
                  <motion.div
                    className="absolute inset-0 rounded-md ring-2 ring-primary-500 dark:ring-primary-400 pointer-events-none"
                    layoutId="inputFocus"
                  />
                )}
              </div>
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                Date of the weather observation
              </p>
            </motion.div>
          </div>
        )}
        {/* Paste JSON */}
        {tab === "json" && (
          <div className="mb-6">
            <label className="label" htmlFor="json-input">
              Paste JSON
            </label>
            <textarea
              id="json-input"
              className="textarea input-animated font-mono"
              value={rawInput}
              onChange={handleRawInput}
              placeholder='{"minTemp": 15, "maxTemp": 30, "precipitation": 50, "date": "2024-06-01"}'
              required
            />
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
              Paste a JSON object or array of objects. Only the first record
              will be used.
            </p>
          </div>
        )}
        {/* Paste CSV/TXT */}
        {tab === "csv" && (
          <div className="mb-6">
            <label className="label" htmlFor="csv-input">
              Paste CSV or TXT
            </label>
            <textarea
              id="csv-input"
              className="textarea input-animated font-mono"
              value={rawInput}
              onChange={handleRawInput}
              placeholder={
                "minTemp,maxTemp,precipitation,date\n15,30,50,2024-06-01"
              }
              required
            />
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
              Paste CSV or tab-separated values. Header row is optional. Only
              the first record will be used.
            </p>
          </div>
        )}
        {/* File Upload */}
        {tab === "file" && (
          <div className="mb-6">
            <label className="label" htmlFor="file-input">
              Upload File (.json, .csv, .txt)
            </label>
            <input
              id="file-input"
              type="file"
              accept=".json,.csv,.txt"
              className="input"
              onChange={handleFile}
              required
            />
            {fileName && (
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                Selected: {fileName}
              </p>
            )}
            {parsedData && (
              <div className="mt-2 text-xs text-green-600 dark:text-green-400">
                File parsed successfully.
              </div>
            )}
          </div>
        )}
        {/* Error message */}
        {parseError && (
          <div className="alert alert-error mb-4">
            <div className="flex items-center">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="h-5 w-5 mr-2"
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
              {parseError}
            </div>
          </div>
        )}
        <div className="flex justify-end mt-8">
          <motion.button
            type="submit"
            className="btn btn-primary btn-icon"
            disabled={isLoading}
            whileHover={{
              scale: isLoading ? 1 : 1.03,
              boxShadow:
                theme === "dark"
                  ? "0 0 15px rgba(59, 130, 246, 0.6)"
                  : "0 0 10px rgba(59, 130, 246, 0.5)",
            }}
            whileTap={{ scale: isLoading ? 1 : 0.97 }}
          >
            {isLoading ? (
              <span className="flex items-center">
                <svg
                  className="animate-spin -ml-1 mr-2 h-5 w-5 text-white"
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                >
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                  ></circle>
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  ></path>
                </svg>
                Processing...
              </span>
            ) : (
              <>
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="h-5 w-5"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M13 10V3L4 14h7v7l9-11h-7z"
                  />
                </svg>
                Make Prediction
              </>
            )}
          </motion.button>
        </div>
      </form>
      </EnhancedCard>
    </LazyWrapper>
  );
};

export default PredictionForm;
