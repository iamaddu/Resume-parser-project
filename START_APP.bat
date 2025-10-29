@echo off
echo ========================================
echo   NEUROMATCH AI - Starting Application
echo ========================================
echo.
echo Killing any existing Streamlit processes...
taskkill /F /IM streamlit.exe 2>nul
timeout /t 2 /nobreak >nul
echo.
echo Starting NeuroMatch AI Dashboard...
echo Open your browser to: http://localhost:8501
echo.
streamlit run futuristic_app.py
