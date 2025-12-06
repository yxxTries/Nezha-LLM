@echo off
echo Starting Nezha-LLM...
echo.

:: Start API server in a new window
echo Starting API Server on port 8000...
start "Nezha-LLM API Server" cmd /k "cd /d %~dp0 && .venv\Scripts\activate && uvicorn api.main:app --host 0.0.0.0 --port 8000"

:: Wait a moment for API to initialize
timeout /t 2 /nobreak >nul

:: Start UI server in a new window
echo Starting UI Server on port 8080...
start "Nezha-LLM UI Server" cmd /k "cd /d %~dp0\ui && python -m http.server 8080"

:: Wait a moment then open browser
timeout /t 2 /nobreak >nul
echo.
echo Opening UI in browser...
start http://localhost:8080

echo.
echo ========================================
echo Nezha-LLM is running!
echo ========================================
echo API Server: http://localhost:8000
echo API Docs:   http://localhost:8000/docs
echo UI:         http://localhost:8080
echo ========================================
echo.
echo Close the terminal windows to stop the servers.
