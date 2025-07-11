import { Routes, Route } from 'react-router-dom'
import { Toaster } from 'react-hot-toast'
import { Link } from 'react-router-dom'
import { ArrowRight, BarChart3, Shield, Zap, Waves, TrendingUp, AlertTriangle, Calendar, Activity, Map, Settings } from 'lucide-react'
import { motion } from 'framer-motion'
import Header from './components/Header'
import Footer from './components/Footer'
import Dashboard from './components/Dashboard'
import ErrorBoundary from './components/ErrorBoundary'
import PredictionPage from './pages/PredictionPage'
import NotFoundPage from './pages/NotFoundPage'
import EnhancedHistoricalPage from './pages/EnhancedHistoricalPage'
import ForecastingPage from './pages/ForecastingPage'
import WeatherDashboard from './pages/WeatherDashboard'
import RiskAnalysisPage from './pages/RiskAnalysisPage'
import ComprehensiveAnalytics from './pages/ComprehensiveAnalytics'

// Modern HomePage component with SUPARCO data
const HomePage = () => {
  const features = [
    {
      icon: BarChart3,
      title: "SUPARCO GCM Data",
      description: "5 GCM ensemble average for Swat River Basin climate projections.",
      link: "/forecasting"
    },
    {
      icon: Shield,
      title: "Risk Assessment",
      description: "Comprehensive multi-dimensional flood risk analysis and assessment.",
      link: "/risk-analysis"
    },
    {
      icon: Zap,
      title: "Real-time Predictions",
      description: "Instant flood discharge predictions based on current weather conditions.",
      link: "/predict"
    },
    {
      icon: TrendingUp,
      title: "200-Year Forecasting",
      description: "Long-term climate projections and flood forecasting capabilities.",
      link: "/forecasting"
    },
    {
      icon: Activity,
      title: "Weather Monitoring",
      description: "Real-time weather data integration and monitoring dashboard.",
      link: "/weather"
    },
    {
      icon: Calendar,
      title: "Historical Analysis",
      description: "Comprehensive historical data analysis and trend visualization.",
      link: "/historical"
    },
    {
      icon: Settings,
      title: "Advanced Analytics",
      description: "Model performance metrics and comprehensive system analytics.",
      link: "/analytics"
    },
    {
      icon: Map,
      title: "Interactive Dashboard",
      description: "Comprehensive visualization and monitoring dashboard.",
      link: "/dashboard"
    }
  ]

  const stats = [
    { label: "Temperature Rise", value: "+1.3°C to +3.7°C" },
    { label: "Precipitation Change", value: "-20% to +23%" },
    { label: "GCM Models", value: "5 Average" },
    { label: "Basin Coverage", value: "Swat River" }
  ]

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative py-20 lg:py-32">
        <div className="container mx-auto px-4">
          <div className="max-w-4xl mx-auto text-center">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              className="mb-8"
            >
              <div className="inline-flex items-center gap-2 bg-primary/10 text-primary px-4 py-2 rounded-full text-sm font-medium mb-6">
                <Waves className="h-4 w-4" />
                SUPARCO Climate Projections
              </div>
              <h1 className="text-4xl md:text-6xl font-bold text-foreground mb-6">
                Advanced Flood
                <span className="text-primary"> Prediction</span>
              </h1>
              <p className="text-xl text-muted-foreground mb-8 max-w-2xl mx-auto">
                AI-powered flood forecasting for the Swat River Basin using SUPARCO's recommended 5 GCM ensemble average
                with temperature rise +1.3°C to +3.7°C and precipitation change -20% to +23%.
              </p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
              className="flex flex-col sm:flex-row gap-4 justify-center"
            >
              <Link to="/predict">
                <button className="btn btn-primary btn-lg group">
                  Start Prediction
                  <ArrowRight className="h-5 w-5 transition-transform group-hover:translate-x-1" />
                </button>
              </Link>
              <Link to="/dashboard">
                <button className="btn btn-outline btn-lg">
                  View Dashboard
                  <BarChart3 className="h-5 w-5 ml-2" />
                </button>
              </Link>
            </motion.div>
          </div>
        </div>

        {/* Background Elements */}
        <div className="absolute inset-0 -z-10 overflow-hidden">
          <div className="absolute top-1/4 left-1/4 w-72 h-72 bg-primary/5 rounded-full blur-3xl" />
          <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-blue-500/5 rounded-full blur-3xl" />
        </div>
      </section>

      {/* SUPARCO Stats Section */}
      <section className="py-16 border-t bg-muted/30">
        <div className="container mx-auto px-4">
          <div className="text-center mb-12">
            <h2 className="text-2xl font-bold text-foreground mb-4">SUPARCO Climate Projections</h2>
            <p className="text-muted-foreground">Based on 5 GCM ensemble average for Swat River Basin (Sattar et al., 2020)</p>
            <p className="text-sm text-muted-foreground mt-2">Streamflow Impact: Increased Nov-May, Reduced Jun-Dec</p>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {stats.map((stat, index) => (
              <motion.div
                key={stat.label}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                className="text-center"
              >
                <div className="text-2xl md:text-3xl font-bold text-primary mb-2">
                  {stat.value}
                </div>
                <div className="text-sm text-muted-foreground">
                  {stat.label}
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20">
        <div className="container mx-auto px-4">
          <div className="max-w-2xl mx-auto text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-4">
              Climate Analysis Features
            </h2>
            <p className="text-muted-foreground">
              Advanced flood prediction using SUPARCO's climate data and machine learning models.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            {features.map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                className="card card-hover p-6 text-center group cursor-pointer"
                onClick={() => window.location.href = feature.link}
              >
                <div className="inline-flex items-center justify-center w-12 h-12 bg-primary/10 text-primary rounded-lg mb-4 group-hover:bg-primary group-hover:text-white transition-colors">
                  <feature.icon className="h-6 w-6" />
                </div>
                <h3 className="text-lg font-semibold text-foreground mb-2 group-hover:text-primary transition-colors">
                  {feature.title}
                </h3>
                <p className="text-muted-foreground text-sm">
                  {feature.description}
                </p>
                <div className="mt-4 opacity-0 group-hover:opacity-100 transition-opacity">
                  <ArrowRight className="h-4 w-4 mx-auto text-primary" />
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-primary text-primary-foreground">
        <div className="container mx-auto px-4">
          <div className="max-w-3xl mx-auto text-center">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
            >
              <AlertTriangle className="h-16 w-16 mx-auto mb-6 opacity-90" />
              <h2 className="text-3xl md:text-4xl font-bold mb-4">
                Ready to Analyze Climate Impact?
              </h2>
              <p className="text-primary-foreground/80 mb-8 text-lg">
                Use SUPARCO's climate projections to predict flood discharge and assess risks.
              </p>
              <Link to="/predict">
                <button className="btn bg-white text-primary hover:bg-white/90 btn-lg">
                  Start Climate Analysis
                  <ArrowRight className="h-5 w-5" />
                </button>
              </Link>
            </motion.div>
          </div>
        </div>
      </section>
    </div>
  )
}

function App() {
  return (
    <ErrorBoundary>
      <div className="flex flex-col min-h-screen bg-background text-foreground">
        <Header />
        <main className="flex-grow">
          <ErrorBoundary>
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route path="/predict" element={<PredictionPage />} />
              <Route path="/dashboard" element={<div className="container mx-auto px-4 py-8"><Dashboard /></div>} />
              <Route path="/historical" element={<EnhancedHistoricalPage />} />
              <Route path="/forecasting" element={<ForecastingPage />} />
              <Route path="/weather" element={<WeatherDashboard />} />
              <Route path="/risk-analysis" element={<RiskAnalysisPage />} />
              <Route path="/analytics" element={<ComprehensiveAnalytics />} />
              <Route path="*" element={<NotFoundPage />} />
            </Routes>
          </ErrorBoundary>
        </main>
        <Footer />

        {/* Toast Notifications */}
        <Toaster
          position="top-right"
          toastOptions={{
            duration: 4000,
            style: {
              background: 'hsl(var(--card))',
              color: 'hsl(var(--card-foreground))',
              border: '1px solid hsl(var(--border))',
            },
          }}
        />
      </div>
    </ErrorBoundary>
  )
}

export default App