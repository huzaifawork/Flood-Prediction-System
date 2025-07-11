import { useState, useEffect } from 'react'
import { Link, NavLink, useLocation } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { Menu, X, Waves } from 'lucide-react'
import ThemeToggle from './ThemeToggle'
import HealthIndicator from './HealthIndicator'
import NotificationCenter from './NotificationCenter'

const Header = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false)
  const [scrolled, setScrolled] = useState(false)
  const location = useLocation()

  useEffect(() => {
    setIsMenuOpen(false)
  }, [location.pathname])

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 10)
    }
    window.addEventListener('scroll', handleScroll)
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  return (
    <header
      className={`sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 transition-all duration-200 ${
        scrolled ? 'shadow-sm' : ''
      }`}
    >
      <div className="container mx-auto px-4 h-16 flex items-center justify-between">
        <Link to="/" className="flex items-center space-x-3 group">
          <div className="relative">
            <Waves className="h-8 w-8 text-primary transition-colors group-hover:text-primary/80" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-foreground group-hover:text-primary transition-colors">
              FloodPredict
            </h1>
            <p className="text-xs text-muted-foreground">
              Swat River Basin
            </p>
          </div>
        </Link>

        {/* Desktop Navigation */}
        <nav className="hidden lg:flex items-center space-x-4">
          <NavLink
            to="/"
            className={({ isActive }) =>
              `text-sm font-medium transition-colors hover:text-primary ${
                isActive ? 'text-primary' : 'text-muted-foreground'
              }`
            }
          >
            Home
          </NavLink>
          <NavLink
            to="/predict"
            className={({ isActive }) =>
              `text-sm font-medium transition-colors hover:text-primary ${
                isActive ? 'text-primary' : 'text-muted-foreground'
              }`
            }
          >
            Prediction
          </NavLink>
          <NavLink
            to="/dashboard"
            className={({ isActive }) =>
              `text-sm font-medium transition-colors hover:text-primary ${
                isActive ? 'text-primary' : 'text-muted-foreground'
              }`
            }
          >
            Dashboard
          </NavLink>
          <NavLink
            to="/historical"
            className={({ isActive }) =>
              `text-sm font-medium transition-colors hover:text-primary ${
                isActive ? 'text-primary' : 'text-muted-foreground'
              }`
            }
          >
            Historical
          </NavLink>
          <NavLink
            to="/forecasting"
            className={({ isActive }) =>
              `text-sm font-medium transition-colors hover:text-primary ${
                isActive ? 'text-primary' : 'text-muted-foreground'
              }`
            }
          >
            Forecasting
          </NavLink>
          <NavLink
            to="/weather"
            className={({ isActive }) =>
              `text-sm font-medium transition-colors hover:text-primary ${
                isActive ? 'text-primary' : 'text-muted-foreground'
              }`
            }
          >
            Weather
          </NavLink>
          <NavLink
            to="/risk-analysis"
            className={({ isActive }) =>
              `text-sm font-medium transition-colors hover:text-primary ${
                isActive ? 'text-primary' : 'text-muted-foreground'
              }`
            }
          >
            Risk Analysis
          </NavLink>
          <NavLink
            to="/analytics"
            className={({ isActive }) =>
              `text-sm font-medium transition-colors hover:text-primary ${
                isActive ? 'text-primary' : 'text-muted-foreground'
              }`
            }
          >
            Analytics
          </NavLink>
        </nav>

        {/* Desktop Controls */}
        <div className="hidden lg:flex items-center space-x-4">
          <HealthIndicator />
          <NotificationCenter />
          <ThemeToggle />
        </div>

        {/* Mobile Controls */}
        <div className="lg:hidden flex items-center space-x-2">
          <HealthIndicator />
          <NotificationCenter />
          <ThemeToggle />
          <button
            onClick={() => setIsMenuOpen(!isMenuOpen)}
            className="p-2 text-muted-foreground hover:text-foreground transition-colors"
            aria-label="Toggle menu"
          >
            <AnimatePresence mode="wait">
              {isMenuOpen ? (
                <motion.div
                  key="close"
                  initial={{ opacity: 0, rotate: -90 }}
                  animate={{ opacity: 1, rotate: 0 }}
                  exit={{ opacity: 0, rotate: 90 }}
                  transition={{ duration: 0.2 }}
                >
                  <X className="h-5 w-5" />
                </motion.div>
              ) : (
                <motion.div
                  key="menu"
                  initial={{ opacity: 0, rotate: 90 }}
                  animate={{ opacity: 1, rotate: 0 }}
                  exit={{ opacity: 0, rotate: -90 }}
                  transition={{ duration: 0.2 }}
                >
                  <Menu className="h-5 w-5" />
                </motion.div>
              )}
            </AnimatePresence>
          </button>
        </div>
      </div>

      {/* Mobile Navigation */}
      <AnimatePresence>
        {isMenuOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.3 }}
            className="lg:hidden border-t bg-background"
          >
            <nav className="container mx-auto px-4 py-4 space-y-4">
              <NavLink
                to="/"
                className={({ isActive }) =>
                  `block text-sm font-medium transition-colors hover:text-primary ${
                    isActive ? 'text-primary' : 'text-muted-foreground'
                  }`
                }
              >
                Home
              </NavLink>
              <NavLink
                to="/predict"
                className={({ isActive }) =>
                  `block text-sm font-medium transition-colors hover:text-primary ${
                    isActive ? 'text-primary' : 'text-muted-foreground'
                  }`
                }
              >
                Prediction
              </NavLink>
              <NavLink
                to="/dashboard"
                className={({ isActive }) =>
                  `block text-sm font-medium transition-colors hover:text-primary ${
                    isActive ? 'text-primary' : 'text-muted-foreground'
                  }`
                }
              >
                Dashboard
              </NavLink>
              <NavLink
                to="/historical"
                className={({ isActive }) =>
                  `block text-sm font-medium transition-colors hover:text-primary ${
                    isActive ? 'text-primary' : 'text-muted-foreground'
                  }`
                }
              >
                Historical Data
              </NavLink>
              <NavLink
                to="/forecasting"
                className={({ isActive }) =>
                  `block text-sm font-medium transition-colors hover:text-primary ${
                    isActive ? 'text-primary' : 'text-muted-foreground'
                  }`
                }
              >
                200-Year Forecasting
              </NavLink>
              <NavLink
                to="/weather"
                className={({ isActive }) =>
                  `block text-sm font-medium transition-colors hover:text-primary ${
                    isActive ? 'text-primary' : 'text-muted-foreground'
                  }`
                }
              >
                Weather Dashboard
              </NavLink>
              <NavLink
                to="/risk-analysis"
                className={({ isActive }) =>
                  `block text-sm font-medium transition-colors hover:text-primary ${
                    isActive ? 'text-primary' : 'text-muted-foreground'
                  }`
                }
              >
                Risk Analysis
              </NavLink>
              <NavLink
                to="/analytics"
                className={({ isActive }) =>
                  `block text-sm font-medium transition-colors hover:text-primary ${
                    isActive ? 'text-primary' : 'text-muted-foreground'
                  }`
                }
              >
                Analytics
              </NavLink>
            </nav>
          </motion.div>
        )}
      </AnimatePresence>
    </header>
  )
}

export default Header
