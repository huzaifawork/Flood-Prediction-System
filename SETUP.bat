@echo off
title Flood Prediction System - Setup
color 0A

echo.
echo ========================================
echo   FLOOD PREDICTION SYSTEM - SETUP
echo ========================================
echo.
echo This will install all required dependencies.
echo Please ensure you have:
echo - Python 3.8+ installed and added to PATH
echo - Node.js 16+ installed
echo.
pause

echo.
echo [1/6] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)
echo ✓ Python found

echo.
echo [2/6] Checking Node.js installation...
node --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Node.js is not installed or not in PATH
    echo Please install Node.js 16+ from https://nodejs.org/
    pause
    exit /b 1
)
echo ✓ Node.js found

echo.
echo [3/6] Installing Python dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install Python dependencies
    pause
    exit /b 1
)
echo ✓ Python dependencies installed

echo.
echo [4/6] Installing API dependencies...
cd flood-prediction-api
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install API dependencies
    cd ..
    pause
    exit /b 1
)
cd ..
echo ✓ API dependencies installed

echo.
echo [5/6] Installing Web App dependencies...
cd flood-prediction-webapp
npm install
if errorlevel 1 (
    echo ERROR: Failed to install Web App dependencies
    cd ..
    pause
    exit /b 1
)
cd ..
echo ✓ Web App dependencies installed

echo.
echo [6/6] Testing installation...
echo Testing Python imports...
python -c "import pandas, numpy, sklearn, joblib; print('✓ Python packages working')"
if errorlevel 1 (
    echo ERROR: Python packages not working properly
    pause
    exit /b 1
)

echo.
echo ========================================
echo   SETUP COMPLETED SUCCESSFULLY!
echo ========================================
echo.
echo You can now run START.bat to launch the application.
echo.
pause
