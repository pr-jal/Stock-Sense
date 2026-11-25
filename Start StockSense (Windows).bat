@echo off
cd /d "%~dp0"
echo Starting StockSense... the website will open in your browser.
where py >nul 2>nul && (py -m streamlit run app.py) || (python -m streamlit run app.py)
pause
