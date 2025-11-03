@echo off
echo Cleaning up unnecessary files from NeuroMatch AI project...
echo.

REM Delete duplicate test files
del "100_TEST_RESUMES.txt"
del "200_RESUMES_BULK_TEST.txt"
del "200_TEST_RESUMES.py"
del "SAMPLE_RESUMES.txt"
del "data_science_test.txt"
del "healthcare_science_test.txt"
del "marketing_business_test.txt"
del "software_engineering_test.txt"
del "data_science_jobs.txt"
del "marketing_business_jobs.txt"
del "software_engineering_jobs.txt"

REM Delete old documentation files
del "ALL_FEATURES_RESTORED.md"
del "ALL_FIXES_APPLIED.md"
del "ALL_ISSUES_FIXED_FINAL.md"
del "COMPLETE_HR_SYSTEM.md"
del "COMPLETE_IMPLEMENTATION_GUIDE.md"
del "COMPLETE_METRICS_DASHBOARD.md"
del "COMPLETE_ML_IMPLEMENTATION.md"
del "COMPLETE_PROJECT_DETAILS.md"
del "COMPLETE_PROJECT_SUMMARY.md"
del "COMPLETE_TEST_CHECKLIST.md"
del "COMPREHENSIVE_TESTING_GUIDE.md"
del "CORRECTED_METHODOLOGY_FINAL.md"
del "CORRECTED_METHODOLOGY_WITH_ML.md"
del "CRITICAL_FIXES_NEEDED.md"
del "DATABASE_FIXED.md"
del "DATABASE_SETUP_GUIDE.md"
del "ERROR_FIXED.md"
del "EVERYTHING_FIXED_FINAL.md"
del "EVERYTHING_WORKING.md"
del "FEATURE_COMPLETE_SUMMARY.md"
del "FILES_TO_KEEP.md"
del "FINAL_CHANGES_SUMMARY.md"
del "FINAL_CLEANUP_SUMMARY.md"
del "FINAL_COMPLETE_GUIDE.md"
del "FINAL_FIX_COMPLETE.md"
del "FINAL_METHODOLOGY_SUMMARY.md"
del "FINAL_PROJECT_DOCUMENTATION.md"
del "FINAL_PROJECT_SUMMARY.md"
del "FINAL_STATUS.md"
del "FINAL_SUCCESS_PROOF.txt"
del "FINAL_SUMMARY.md"
del "FINAL_TESTING_DATASET.md"
del "FINAL_VERIFICATION.md"
del "FINAL_WORKING_SOLUTION.md"
del "FIXES_APPLIED.md"
del "FOLLOW_THESE_STEPS.md"
del "GENIUS_FEATURES.md"
del "HIDDEN_GEMS_FEATURE.md"
del "HOW_TO_TEST_100_RESUMES.md"
del "HOW_TO_USE_METHODOLOGY.md"
del "LATEST_IMPROVEMENTS.md"
del "METHODOLOGY_FIGURES_TABLES.md"
del "METHODOLOGY_QUICK_REFERENCE.md"
del "ML_MODELS_DOCUMENTATION.md"
del "ML_PROOF_FOR_REVIEWERS.md"
del "NEW_FEATURE_SUMMARY.md"
del "PROJECT_FILES_GUIDE.md"
del "PROJECT_REPORT_AIML.md"
del "QUICK_FEATURE_GUIDE.md"
del "QUICK_FILTERS_GUIDE.md"
del "QUICK_REFERENCE.md"
del "QUICK_START_GUIDE.md"
del "QUICK_START_NOW.md"
del "QUICK_TEST.md"
del "README_FINAL.md"
del "RESEARCH_METHODOLOGY_PART1.md"
del "RESEARCH_METHODOLOGY_PART2.md"
del "RESEARCH_METHODOLOGY_PART3.md"
del "RESEARCH_PAPER_METHODOLOGY_FINAL.md"
del "RESEARCH_PAPER_METRICS.md"
del "RESEARCH_READY_SUMMARY.md"
del "RESTART_APP.md"
del "SIMPLE_FIX_INSTRUCTIONS.md"
del "SIMPLE_GUIDE.md"
del "SOCIAL_INTELLIGENCE_FEATURE.md"
del "SOLUTION.md"
del "START_HERE.md"
del "TEST_GUIDE.md"
del "TROUBLESHOOTING.md"
del "WORKING_MODELS_ONLY.md"

REM Delete old app versions
del "app.py"
del "professional_app.py"
del "simple_app_fixed.py"

REM Delete test and utility files
del "check_app_database.py"
del "cleanup.py"
del "cognitive_model.py"
del "explainable_ai.py"
del "fix_app.py"
del "fix_indentation.py"
del "install_pdf_support.py"
del "ml_model_proof.py"
del "ml_scoring.py"
del "ranker.py"
del "reset_database.py"
del "setup.py"
del "synthetic_cognitive_dataset.py"
del "test_all_features.py"
del "test_database.py"
del "test_database_save.py"
del "test_diverse_patterns.py"
del "test_ml_components.py"
del "test_ml_integration.py"
del "test_ml_status.py"
del "test_parsing.py"
del "test_simple.py"
del "test_system.py"
del "TEST_DATASETS.py"
del "TEST_SOCIAL_INTELLIGENCE.py"
del "train_and_evaluate_models.py"
del "train_rf_model.py"
del "FINAL_VALIDATION_TEST.py"
del "SIMPLE_VALIDATION.py"

REM Delete other files
del "model_evaluation_results.csv"
del "model_training_results.txt"
del "validation_results.txt"
del "simple_requirements.txt"
del "EMERGENCY_FIX.txt"
del "COLOR_GUIDE.md"
del "hr_notes_manager.py"

REM Delete batch files
del "CLEANUP_ALL.bat"
del "CLEANUP_PROJECT.bat"
del "RESTART_FRESH.bat"
del "SETUP_ML_MODELS.bat"
del "START_APP.bat"

REM Delete Docker files
del "Dockerfile"
del "docker-compose.yml"

REM Remove empty directories
rmdir /s /q ".dist" 2>nul
rmdir /s /q "__pycache__" 2>nul
rmdir /s /q "config" 2>nul
rmdir /s /q "core" 2>nul
rmdir /s /q "data" 2>nul
rmdir /s /q "hr_data" 2>nul
rmdir /s /q "models" 2>nul
rmdir /s /q "neuro_match_env" 2>nul
rmdir /s /q "static" 2>nul
rmdir /s /q "templates" 2>nul
rmdir /s /q "tests" 2>nul
rmdir /s /q "uploads" 2>nul
rmdir /s /q "web_app" 2>nul

echo.
echo ✅ Cleanup complete! Your project is now clean and organized.
echo.
echo 📁 REMAINING ESSENTIAL FILES:
echo   - futuristic_app.py (MAIN APP)
echo   - ml_models.py (ML/DL MODELS)
echo   - database_manager.py (DATABASE)
echo   - indian_salary_data.py (SALARY CALCULATOR)
echo   - social_intelligence.py (SOCIAL FEATURES)
echo   - resume_parser.py (PDF SUPPORT)
echo   - requirements.txt (DEPENDENCIES)
echo   - requirements_ml.txt (ML DEPENDENCIES)
echo   - hr_database.db (YOUR DATA)
echo   - resume_scoring_model.pkl (TRAINED MODEL)
echo   - rl_weights.pkl (Q-LEARNING WEIGHTS)
echo   - BULK_TEST_FORMAT.txt (TEST DATA)
echo   - 20_COMPLETE_TEST_RESUMES.txt (YOUR 20 RESUMES)
echo   - FINAL_SUBMISSION_READY.md (SUBMISSION GUIDE)
echo   - FINAL_METHODOLOGY_UPDATED.md (YOUR METHODOLOGY)
echo   - README.md (PROJECT README)
echo.
pause
