@echo off
echo Activating virtual environment...
call venv\Scripts\activate

echo Running AI Detector and Humanizer app with GPU support...
set CUDA_VISIBLE_DEVICES=0
python ai_detector_humanizer_app_new_with_grammer_fix.py

pause
