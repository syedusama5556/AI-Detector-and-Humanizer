@echo off
echo Activating virtual environment...
call venvcpu\Scripts\activate

echo Running AI Detector and Humanizer app...
python ai_detector_humanizer_app.py

pause