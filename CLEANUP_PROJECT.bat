@echo off
echo ========================================
echo PROJECT CLEANUP - Remove Unnecessary Files
echo ========================================
echo.

echo Cleaning up temporary and unnecessary files...
echo.

REM Remove Python cache
echo [1/5] Removing Python cache files...
if exist __pycache__ rmdir /s /q __pycache__
if exist *.pyc del /q *.pyc
if exist .pytest_cache rmdir /s /q .pytest_cache

REM Remove temporary files
echo [2/5] Removing temporary files...
if exist *.tmp del /q *.tmp
if exist *.log del /q *.log
if exist temp rmdir /s /q temp

REM Remove duplicate/old documentation
echo [3/5] Keeping only essential documentation...
REM Keep: FINAL_PROJECT_SUMMARY.md, HIDDEN_GEMS_FEATURE.md, QUICK_FILTERS_GUIDE.md
REM Remove old/duplicate docs if any

REM Remove old checkpoints or backup files
echo [4/5] Removing backup files...
if exist *.bak del /q *.bak
if exist *_old.* del /q *_old.*

REM Remove IDE specific files
echo [5/5] Removing IDE files...
if exist .vscode rmdir /s /q .vscode
if exist .idea rmdir /s /q .idea

echo.
echo ========================================
echo CLEANUP COMPLETE!
echo ========================================
echo.
echo Essential files kept:
echo - futuristic_app.py (main application)
echo - ml_models.py (ML/DL models)
echo - train_and_evaluate_models.py (model evaluation)
echo - requirements_ml.txt (dependencies)
echo - 100_TEST_RESUMES.txt (test data)
echo - FINAL_PROJECT_SUMMARY.md (documentation)
echo - HIDDEN_GEMS_FEATURE.md (feature docs)
echo - QUICK_FILTERS_GUIDE.md (user guide)
echo - START_APP.bat (launcher)
echo - SETUP_ML_MODELS.bat (setup script)
echo.
pause
