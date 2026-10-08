@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run setup first: python -m venv .venv
  echo Then: .venv\Scripts\python.exe -m pip install -e .
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run app.py

