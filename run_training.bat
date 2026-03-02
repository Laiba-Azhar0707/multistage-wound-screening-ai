@echo off
title WoundAI Full Pipeline
chcp 65001 > nul
set BASE=%~dp0
set B=%BASE:~0,-1%
set PY=%B%\venv\Scripts\python.exe

echo.
echo Starting WoundAI pipeline...
echo Log will be saved to: %B%\results\pipeline_log.txt
echo.

"%PY%" -u "%B%\run_pipeline.py"
