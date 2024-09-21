@echo off
echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate

echo Installing dependencies...
pip install -r requirements-gpu.txt

echo Downloading spaCy English model...
python -m spacy download en_core_web_sm

echo Installation complete!
pause
