@echo off
echo ========================================
echo   Academic Paper Simplifier
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from python.org
    pause
    exit /b 1
)

echo [1/3] Checking dependencies...
pip show streamlit >nul 2>&1
if errorlevel 1 (
    echo Dependencies not found. Installing...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
) else (
    echo Dependencies already installed.
)

echo.
echo [2/3] Starting Streamlit server...
echo.
echo The app will open in your browser at http://localhost:8501
echo Press Ctrl+C to stop the server
echo.

REM Run the Streamlit app
streamlit run app.py

pause
