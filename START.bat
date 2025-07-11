@echo off
title Flood Prediction System - Launcher
color 0B

echo.
echo ========================================
echo   FLOOD PREDICTION SYSTEM - LAUNCHER
echo ========================================
echo.

REM Check if setup was completed
if not exist "flood-prediction-webapp\node_modules" (
    echo ERROR: Setup not completed!
    echo Please run SETUP.bat first as administrator.
    echo.
    pause
    exit /b 1
)

echo Starting Flood Prediction System...
echo.

REM Start the API server in background
echo [1/3] Starting API Server...
cd flood-prediction-api
start /B python app.py
cd ..
echo ✓ API Server starting on http://localhost:5000

REM Wait a moment for API to start
timeout /t 3 /nobreak >nul

REM Start the web application
echo.
echo [2/3] Starting Web Application...
cd flood-prediction-webapp
start /B npm run dev
cd ..
echo ✓ Web App starting on http://localhost:5173

REM Wait for web app to start
echo.
echo [3/3] Opening browser...
timeout /t 8 /nobreak >nul

REM Open browser
start http://localhost:5173

echo.
echo ========================================
echo   SYSTEM RUNNING SUCCESSFULLY!
echo ========================================
echo.
echo Web Interface: http://localhost:5173
echo API Endpoint:  http://localhost:5000
echo.
echo Press any key to stop the system...
pause >nul

REM Kill the processes
echo.
echo Stopping system...
taskkill /f /im python.exe >nul 2>&1
taskkill /f /im node.exe >nul 2>&1
echo System stopped.
pause
