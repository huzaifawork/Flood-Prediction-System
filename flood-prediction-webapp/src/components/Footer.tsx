import { Link } from 'react-router-dom'
import { Waves, MapPin, Database, Brain, Shield, ExternalLink } from 'lucide-react'

const Footer = () => {
  const currentYear = new Date().getFullYear()

  return (
    <footer className="border-t bg-muted/30 py-12">
      <div className="container mx-auto px-4">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          <div>
            <div className="flex items-center space-x-2 mb-4">
              <Waves className="h-6 w-6 text-primary" />
              <h3 className="text-lg font-semibold text-foreground">FloodPredict</h3>
            </div>
            <p className="text-muted-foreground mb-4">
              Advanced AI-powered flood prediction system for the Swat River Basin using SUPARCO's 5 GCM ensemble data.
            </p>
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <MapPin className="h-4 w-4" />
              <span>Swat River Basin, Pakistan</span>
            </div>
          </div>

          <div>
            <h3 className="text-lg font-semibold mb-4 text-foreground">Navigation</h3>
            <ul className="space-y-2">
              <li>
                <Link to="/" className="text-muted-foreground hover:text-primary transition-colors">
                  Home
                </Link>
              </li>
              <li>
                <Link to="/predict" className="text-muted-foreground hover:text-primary transition-colors">
                  Flood Prediction
                </Link>
              </li>
              <li>
                <Link to="/dashboard" className="text-muted-foreground hover:text-primary transition-colors">
                  Dashboard
                </Link>
              </li>
              <li>
                <Link to="/forecasting" className="text-muted-foreground hover:text-primary transition-colors">
                  200-Year Forecasting
                </Link>
              </li>
              <li>
                <Link to="/weather" className="text-muted-foreground hover:text-primary transition-colors">
                  Weather Data
                </Link>
              </li>
              <li>
                <Link to="/risk-analysis" className="text-muted-foreground hover:text-primary transition-colors">
                  Risk Analysis
                </Link>
              </li>
            </ul>
          </div>

          <div>
            <h3 className="text-lg font-semibold mb-4 text-foreground">Technology</h3>
            <ul className="space-y-2">
              <li className="flex items-center gap-2 text-muted-foreground">
                <Brain className="h-4 w-4" />
                <span>Machine Learning Models</span>
              </li>
              <li className="flex items-center gap-2 text-muted-foreground">
                <Database className="h-4 w-4" />
                <span>SUPARCO Climate Data</span>
              </li>
              <li className="flex items-center gap-2 text-muted-foreground">
                <Shield className="h-4 w-4" />
                <span>Real-time Risk Assessment</span>
              </li>
              <li className="flex items-center gap-2 text-muted-foreground">
                <Waves className="h-4 w-4" />
                <span>Discharge Forecasting</span>
              </li>
            </ul>
          </div>

          <div>
            <h3 className="text-lg font-semibold mb-4 text-foreground">Data Sources</h3>
            <ul className="space-y-2">
              <li>
                <a
                  href="https://suparco.gov.pk"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-muted-foreground hover:text-primary transition-colors flex items-center gap-1"
                >
                  SUPARCO Pakistan
                  <ExternalLink className="h-3 w-3" />
                </a>
              </li>
              <li>
                <a
                  href="https://weatherapi.com"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-muted-foreground hover:text-primary transition-colors flex items-center gap-1"
                >
                  WeatherAPI.com
                  <ExternalLink className="h-3 w-3" />
                </a>
              </li>
              <li className="text-muted-foreground">
                Historical Data: 1995-2017
              </li>
              <li className="text-muted-foreground">
                5 GCM Ensemble Average
              </li>
            </ul>
          </div>
        </div>

        <div className="border-t border-border mt-8 pt-6">
          <div className="flex flex-col md:flex-row justify-between items-center gap-4">
            <p className="text-muted-foreground text-sm">
              &copy; {currentYear} FloodPredict System. Advanced flood prediction for Swat River Basin.
            </p>
            <div className="flex items-center gap-4 text-sm text-muted-foreground">
              <span>Temperature Rise: +1.3°C to +3.7°C</span>
              <span>•</span>
              <span>Precipitation Change: -20% to +23%</span>
            </div>
          </div>
        </div>
      </div>
    </footer>
  )
}

export default Footer