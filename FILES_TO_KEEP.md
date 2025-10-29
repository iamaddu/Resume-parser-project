# NeuroMatch AI - File Structure

## CORE FILES (KEEP - REQUIRED)
- `futuristic_app.py` - Main application (THIS IS THE ONE TO RUN)
- `resume_parser.py` - Resume parsing utilities
- `START_APP.bat` - Clean startup script

## DOCUMENTATION (KEEP)
- `GENIUS_FEATURES.md` - Feature documentation
- `TEST_GUIDE.md` - Testing instructions
- `SAMPLE_RESUMES.txt` - Test data
- `QUICK_TEST.md` - Quick test guide
- `FILES_TO_KEEP.md` - This file

## OPTIONAL/LEGACY FILES (CAN DELETE IF NOT NEEDED)
- `app.py` - Old version
- `simple_app.py` - Simple version
- `simple_app_fixed.py` - Fixed simple version
- `web_app/neuromatch_app.py` - Alternative version
- `cleanup.py` - Utility script
- `cognitive_model.py` - Advanced model (not used in main app)
- `explainable_ai.py` - Advanced features (not used in main app)
- `ranker.py` - Old ranker
- `test_simple.py` - Test file
- `install_pdf_support.py` - Setup script
- `setup.py` - Setup script
- `synthetic_cognitive_dataset.py` - Data generation

## DIRECTORIES
- `core/` - Advanced cognitive features (optional)
- `data/` - Training data utilities (optional)
- `tests/` - Test files (optional)
- `config/` - Configuration (optional)
- `web_app/` - Alternative app version (optional)

## HOW TO RUN
1. Double-click `START_APP.bat`
2. OR run in terminal: `streamlit run futuristic_app.py`
3. Open browser to: http://localhost:8501

## CURRENT ISSUE
The HTML tags (<strong>, <br>) are showing as text instead of rendering.
This happens when Streamlit's markdown renderer escapes HTML.

## FIX NEEDED
Replace HTML tags in homepage with Streamlit-native formatting or ensure unsafe_allow_html=True is working.
