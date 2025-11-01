@echo off
echo ========================================
echo CLEANUP - Remove Unnecessary Files
echo ========================================
echo.

echo This will delete:
echo - All old documentation files (20+ files)
echo - Old application versions
echo - Duplicate files
echo - Empty directories
echo.
echo KEEPING ONLY:
echo - professional_app.py (main app)
echo - ml_models.py
echo - train_and_evaluate_models.py
echo - requirements_ml.txt
echo - 100_TEST_RESUMES.txt
echo - model_evaluation_results.csv
echo - FINAL_PROJECT_DOCUMENTATION.md
echo - START_APP.bat
echo - SETUP_ML_MODELS.bat
echo.

pause
echo.

REM Remove old documentation
echo Removing old documentation files...
del /q COLOR_GUIDE.md 2>nul
del /q COMPLETE_ML_IMPLEMENTATION.md 2>nul
del /q COMPLETE_PROJECT_SUMMARY.md 2>nul
del /q COMPLETE_TEST_CHECKLIST.md 2>nul
del /q FILES_TO_KEEP.md 2>nul
del /q FINAL_CHANGES_SUMMARY.md 2>nul
del /q FINAL_STATUS.md 2>nul
del /q FIXES_APPLIED.md 2>nul
del /q GENIUS_FEATURES.md 2>nul
del /q HOW_TO_TEST_100_RESUMES.md 2>nul
del /q LATEST_IMPROVEMENTS.md 2>nul
del /q ML_MODELS_DOCUMENTATION.md 2>nul
del /q PROJECT_REPORT_AIML.md 2>nul
del /q QUICK_TEST.md 2>nul
del /q README.md 2>nul
del /q TEST_GUIDE.md 2>nul
del /q TROUBLESHOOTING.md 2>nul
del /q UNIQUE_VALUE_PROPOSITION.md 2>nul
del /q PROJECT_FILES_GUIDE.md 2>nul
del /q FINAL_CLEANUP_SUMMARY.md 2>nul
del /q CLEANUP_PROJECT.bat 2>nul
del /q RESEARCH_PAPER_METRICS.md 2>nul
del /q HIDDEN_GEMS_FEATURE.md 2>nul
del /q QUICK_FILTERS_GUIDE.md 2>nul
del /q FINAL_PROJECT_SUMMARY.md 2>nul
del /q COMPLETE_PROJECT_REPORT.md 2>nul

REM Remove old app files
echo Removing old application files...
del /q app.py 2>nul
del /q simple_app.py 2>nul
del /q simple_app_fixed.py 2>nul
del /q cleanup.py 2>nul
del /q cognitive_model.py 2>nul
del /q explainable_ai.py 2>nul
del /q ranker.py 2>nul
del /q resume_parser.py 2>nul
del /q futuristic_app.py 2>nul

REM Remove development files
echo Removing development files...
del /q Dockerfile 2>nul
del /q docker-compose.yml 2>nul
del /q setup.py 2>nul
del /q install_pdf_support.py 2>nul
del /q synthetic_cognitive_dataset.py 2>nul
del /q test_simple.py 2>nul
del /q requirements.txt 2>nul
del /q simple_requirements.txt 2>nul
del /q SAMPLE_RESUMES.txt 2>nul

REM Remove Python cache
echo Removing Python cache...
rmdir /s /q __pycache__ 2>nul
del /q *.pyc 2>nul

REM Remove empty directories
echo Removing empty directories...
rmdir /s /q .dist 2>nul
rmdir /s /q config 2>nul
rmdir /s /q core 2>nul
rmdir /s /q data 2>nul
rmdir /s /q models 2>nul
rmdir /s /q static 2>nul
rmdir /s /q templates 2>nul
rmdir /s /q tests 2>nul
rmdir /s /q uploads 2>nul
rmdir /s /q web_app 2>nul

echo.
echo ========================================
echo CLEANUP COMPLETE!
echo ========================================
echo.
echo Remaining files:
dir /b
echo.
pause
