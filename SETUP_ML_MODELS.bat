@echo off
echo ============================================
echo  Installing ML/DL Models for NeuroMatch AI
echo ============================================
echo.

echo Step 1: Fix Keras 3 compatibility...
pip install tf-keras

echo.
echo Step 2: Installing core ML libraries...
pip install transformers==4.35.0
pip install torch==2.1.0
pip install sentence-transformers==2.2.2

echo.
echo Step 2: Installing traditional ML libraries...
pip install scikit-learn==1.3.2
pip install xgboost==2.0.2

echo.
echo Step 3: Installing data processing libraries...
pip install pandas==2.1.3
pip install numpy==1.26.2

echo.
echo Step 4: Verifying installation...
python -c "import transformers; print('✅ Transformers installed')"
python -c "import torch; print('✅ PyTorch installed')"
python -c "import sentence_transformers; print('✅ Sentence-Transformers installed')"
python -c "import sklearn; print('✅ Scikit-learn installed')"

echo.
echo Step 5: Testing ML models...
python ml_models.py

echo.
echo ============================================
echo  Installation Complete!
echo ============================================
echo.
echo Next steps:
echo 1. Run: python futuristic_app.py
echo 2. Or: streamlit run futuristic_app.py
echo.
pause
