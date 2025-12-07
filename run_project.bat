@echo off
echo ==========================================
echo      Speech Recognition Pipeline
echo ==========================================

echo [1/5] Extracting Single Best Templates (Medoids)...
python tune_templates.py

echo [2/5] Extracting Silence Template...
python extract_silence.py

echo [3/5] Converting Test WAVs to MFCC...
python extract_test_mfcc.py

echo [4/5] Compiling C Code...
gcc level_building.c -o level_building.exe
if %errorlevel% neq 0 (
    echo Compilation Failed!
    exit /b %errorlevel%
)

echo [5/5] Running Level Building...
level_building.exe

echo ==========================================
echo              Done!
echo ==========================================
pause
