@echo off
echo Creating virtual environment...
python -m venv venvcpu

echo Activating virtual environment...
call venvcpu\Scripts\activate

echo Installing dependencies...
pip install -r requirements.txt

echo Downloading spaCy English model...
python -m spacy download en_core_web_sm

echo Installation complete!
pause
