# 🔧 Troubleshooting Guide - NeuroMatch AI

## Common Errors & Solutions

---

### ❌ Error: Keras 3 Not Supported

**Error Message:**
```
ValueError: Your currently installed version of Keras is Keras 3, but this is not yet supported in Transformers. 
Please install the backwards-compatible tf-keras package with `pip install tf-keras`.
```

**Solution:**
```bash
pip install tf-keras
```

**Why:** Transformers library doesn't support Keras 3 yet. Installing `tf-keras` provides backward compatibility.

**Status:** ✅ FIXED in `requirements_ml.txt` and `SETUP_ML_MODELS.bat`

---

### ❌ Error: Transformers Not Found

**Error Message:**
```
ModuleNotFoundError: No module named 'transformers'
```

**Solution:**
```bash
pip install transformers==4.35.0
pip install torch==2.1.0
```

**Why:** BERT models require the transformers library and PyTorch.

---

### ❌ Error: Sentence-Transformers Not Found

**Error Message:**
```
ModuleNotFoundError: No module named 'sentence_transformers'
```

**Solution:**
```bash
pip install sentence-transformers==2.2.2
```

**Why:** Semantic skill matching requires sentence-transformers library.

---

### ❌ Error: Scikit-learn Not Found

**Error Message:**
```
ModuleNotFoundError: No module named 'sklearn'
```

**Solution:**
```bash
pip install scikit-learn==1.3.2
```

**Why:** Random Forest model requires scikit-learn.

---

### ❌ Error: CUDA Out of Memory

**Error Message:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
```python
# In ml_models.py, use CPU instead of GPU
import torch
device = torch.device('cpu')  # Force CPU usage
```

**Why:** BERT models are large. If you don't have a GPU or have limited GPU memory, use CPU.

---

### ❌ Error: Model Download Fails

**Error Message:**
```
OSError: Can't load tokenizer for 'dslim/bert-base-NER'
```

**Solution:**
1. Check internet connection
2. Try manual download:
   ```bash
   python -c "from transformers import pipeline; pipeline('ner', model='dslim/bert-base-NER')"
   ```
3. Wait for download to complete (~420MB)

**Why:** First run downloads models from Hugging Face.

---

### ❌ Error: Streamlit Not Found

**Error Message:**
```
'streamlit' is not recognized as an internal or external command
```

**Solution:**
```bash
pip install streamlit==1.28.2
```

**Why:** Streamlit is required to run the web application.

---

### ❌ Error: Port Already in Use

**Error Message:**
```
OSError: [Errno 98] Address already in use
```

**Solution:**
```bash
# Kill existing Streamlit process
Get-Process | Where-Object {$_.ProcessName -eq "streamlit"} | Stop-Process -Force

# Or use different port
streamlit run futuristic_app.py --server.port 8502
```

**Why:** Another Streamlit app is already running on port 8501.

---

### ❌ Error: Python Version Too Old

**Error Message:**
```
SyntaxError: invalid syntax
```

**Solution:**
```bash
# Check Python version
python --version

# Should be Python 3.8 or higher
# If not, upgrade Python
```

**Why:** Code uses modern Python features (f-strings, type hints, etc.)

---

### ❌ Warning: ML Models Not Available

**Warning Message:**
```
⚠️ ML models not available. Run: pip install -r requirements_ml.txt
```

**Solution:**
```bash
pip install -r requirements_ml.txt
```

**Why:** ML/DL models are optional but recommended for full functionality.

---

### ❌ Error: PDF Extraction Fails

**Error Message:**
```
❌ Could not extract text from PDF
```

**Solution:**
```bash
pip install PyPDF2==3.0.1
```

**Alternative:** Use text input instead of PDF upload.

**Why:** PDF extraction requires PyPDF2 library.

---

## 🚀 Quick Fix: Install Everything

If you're getting multiple errors, install all dependencies at once:

```bash
# Install all ML/DL libraries
pip install -r requirements_ml.txt

# Or use the setup script
.\SETUP_ML_MODELS.bat
```

---

## 🔍 Verify Installation

### Check if ML models are loaded:

```bash
python ml_models.py
```

**Expected Output:**
```
✅ BERT NER model loaded successfully
✅ Sentence-BERT model loaded successfully
============================================================
ML/DL Models Status
============================================================

REINFORCEMENT_LEARNING:
  loaded: True
  feedback_count: 0
  accuracy: 0.0

BERT_NER:
  loaded: True
  model: dslim/bert-base-NER

SEMANTIC_MATCHING:
  loaded: True
  model: all-MiniLM-L6-v2

ATTRITION_PREDICTION:
  loaded: True
  trained: False
  model: Random Forest (100 trees)

DIVERSITY_ANALYTICS:
  loaded: True
  model: Statistical Analysis
```

---

## 💡 Performance Tips

### 1. Speed Up BERT Loading
```python
# Use smaller model
model = "distilbert-base-uncased"  # Faster, smaller
```

### 2. Reduce Memory Usage
```python
# Process fewer resumes at once
batch_size = 10  # Instead of 100
```

### 3. Use CPU if No GPU
```python
# In ml_models.py
device = 'cpu'  # Slower but works everywhere
```

---

## 📞 Still Having Issues?

### Debug Steps:

1. **Check Python version:**
   ```bash
   python --version
   # Should be 3.8+
   ```

2. **Check installed packages:**
   ```bash
   pip list | grep -E "transformers|sentence|sklearn|streamlit"
   ```

3. **Check for conflicts:**
   ```bash
   pip check
   ```

4. **Reinstall everything:**
   ```bash
   pip uninstall transformers sentence-transformers torch
   pip install -r requirements_ml.txt
   ```

5. **Clear cache:**
   ```bash
   # Delete cached models
   rm -rf ~/.cache/huggingface
   ```

---

## ✅ Success Checklist

- [ ] Python 3.8+ installed
- [ ] tf-keras installed
- [ ] transformers installed
- [ ] sentence-transformers installed
- [ ] scikit-learn installed
- [ ] streamlit installed
- [ ] Models downloaded successfully
- [ ] App runs without errors
- [ ] Can access http://localhost:8501

---

## 🎯 Quick Start (After Fixing Errors)

```bash
# 1. Install dependencies
pip install tf-keras
pip install -r requirements_ml.txt

# 2. Test models
python ml_models.py

# 3. Run app
streamlit run futuristic_app.py

# 4. Open browser
# Go to: http://localhost:8501
```

---

**All errors fixed! App is now running successfully at http://localhost:8501** ✅
