@echo off
cd /d F:\Lucy-IA
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :7860 ^| findstr LISTENING') do taskkill /f /pid %%a >nul 2>&1
call venv\Scripts\activate.bat
python main.py
pause
