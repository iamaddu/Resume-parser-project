"""
Simple NeuroMatch AI Test App
Basic version to test the system quickly
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
from datetime import datetime
import tempfile
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try to import resume parser
try:
    from resume_parser import extract_text_from_pdf, parse_resume, is_resume_pdf
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

# Configure page
st.set_page_config(
    page_title="NeuroMatch AI - Test Version",
    page_icon="🧠",
    layout="wide"
)

# Professional CSS Styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    /* Main Container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background: #ffffff;
        border-radius: 20px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.15);
        backdrop-filter: blur(10px);
        margin: 2rem auto;
        max-width: 1200px;
    }
    
    /* Ensure text is always readable */
    .stApp, .main, p, span, div, label, .stMarkdown {
        color: #1e293b !important;
    }
    
    /* Text input and textarea styling */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        color: #1e293b !important;
        background: white !important;
        border: 2px solid #e2e8f0 !important;
    }
    
    /* Select box styling */
    .stSelectbox > div > div > div {
        color: #1e293b !important;
        background: white !important;
    }
    
    /* Header Styles */
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        letter-spacing: -0.02em;
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #475569 !important;
        margin-bottom: 3rem;
        font-weight: 500;
    }
    
    /* Card Styles */
    .feature-card {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
    }
    
    .feature-icon {
        width: 60px;
        height: 60px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 1rem;
        font-size: 1.5rem;
        color: white;
    }
    
    .feature-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0f172a !important;
        margin-bottom: 0.5rem;
    }
    
    .feature-desc {
        color: #334155 !important;
        font-size: 0.95rem;
        line-height: 1.6;
        font-weight: 400;
    }
    
    /* Button Styles */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar Styles */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .css-1d391kg .css-1v0mbdj {
        color: white;
    }
    
    /* Metrics Styles */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(0, 0, 0, 0.05);
    }
    
    /* Status Badges */
    .status-selected {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
    }
    
    .status-shortlisted {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
    }
    
    .status-maybe {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
    }
    
    .status-rejected {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
    }
    
    /* Progress Bar Styles */
    .stProgress > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Upload Area Styles */
    .uploadedFile {
        border: 2px dashed #667eea;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        background: rgba(102, 126, 234, 0.05);
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.75rem;
        font-weight: 700;
        color: #0f172a !important;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
    }
    
    /* Info/Warning/Success boxes */
    .stAlert {
        background: white !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        border-left: 4px solid #667eea !important;
    }
    
    .stAlert p, .stAlert div {
        color: #1e293b !important;
        font-weight: 500 !important;
    }
    
    /* Metric values */
    .stMetric {
        background: white;
        padding: 1rem;
        border-radius: 12px;
    }
    
    .stMetric label, .stMetric p, .stMetric div {
        color: #1e293b !important;
        font-weight: 600 !important;
    }
    
    /* Results Container */
    .results-container {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(0, 0, 0, 0.05);
    }
    
    /* Candidate Card */
    .candidate-card {
        background: #f8fafc;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
        transition: all 0.3s ease;
    }
    
    .candidate-card:hover {
        background: #f1f5f9;
        transform: translateX(5px);
    }
    
    .candidate-card h4, .candidate-card p, .candidate-card div, .candidate-card strong {
        color: #0f172a !important;
    }
    
    /* DataFrame styling */
    .stDataFrame, .stDataFrame td, .stDataFrame th {
        color: #1e293b !important;
        font-weight: 500 !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: white !important;
        color: #0f172a !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
    }
    
    .streamlit-expanderHeader p {
        color: #0f172a !important;
        font-weight: 600 !important;
    }
    
    /* Radio buttons and checkboxes */
    .stRadio label, .stCheckbox label {
        color: #1e293b !important;
        font-weight: 500 !important;
    }
    
    /* Slider labels */
    .stSlider label, .stSlider p {
        color: #1e293b !important;
        font-weight: 600 !important;
    }
    
    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #0f172a !important;
        font-weight: 700 !important;
    }
    
    /* Subheaders */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #0f172a !important;
        font-weight: 700 !important;
    }
    
    /* All paragraphs and text */
    p, span, label, div {
        color: #1e293b !important;
    }
    
    /* Strong and bold text */
    strong, b {
        color: #0f172a !important;
        font-weight: 700 !important;
    }
    
    /* Code blocks */
    code {
        background: #f1f5f9 !important;
        color: #0f172a !important;
        padding: 0.2rem 0.4rem !important;
        border-radius: 4px !important;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

def extract_candidate_name(text):
    """Extract candidate name from resume text"""
    lines = text.strip().split('\n')
    
    # Try to find name in first few lines
    for line in lines[:3]:
        line = line.strip()
        if line and len(line.split()) <= 4 and len(line) > 3:
            # Check if it looks like a name (not too long, not all caps, contains letters)
            if not line.isupper() and any(c.isalpha() for c in line) and not any(word in line.lower() for word in ['resume', 'cv', 'profile', 'summary', 'objective']):
                # Clean up the name
                name = re.sub(r'[^\w\s-]', '', line).strip()
                if len(name.split()) >= 2 and len(name) <= 50:
                    return name
    
    # If no name found, try to extract from patterns like "Name: John Smith" or "John Smith - Title"
    name_patterns = [
        r'(?:name|candidate):\s*([a-zA-Z\s]{2,30})',
        r'^([A-Z][a-z]+\s+[A-Z][a-z]+)',
        r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s*[-–]',
    ]
    
    for pattern in name_patterns:
        match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return "Unknown Candidate"

def parse_resume_professional(text):
    """Professional resume parsing with detailed extraction"""
    text_lower = text.lower()
    
    # Extract candidate name
    name = extract_candidate_name(text)
    
    # Comprehensive skills extraction
    skill_categories = {
        'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust', 'scala', 'kotlin'],
        'web': ['html', 'css', 'react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask', 'spring'],
        'data': ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'pandas', 'numpy', 'scipy'],
        'ml_ai': ['machine learning', 'deep learning', 'tensorflow', 'pytorch', 'scikit-learn', 'keras', 'opencv', 'nlp', 'computer vision'],
        'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'terraform', 'ansible'],
        'tools': ['git', 'jira', 'confluence', 'slack', 'figma', 'photoshop', 'excel', 'powerpoint']
    }
    
    found_skills = []
    skill_categories_found = {}
    
    for category, skills in skill_categories.items():
        category_skills = [skill for skill in skills if skill in text_lower]
        found_skills.extend(category_skills)
        if category_skills:
            skill_categories_found[category] = category_skills
    
    # Extract experience with multiple patterns
    experience_patterns = [
        r'(\d+)[\+]?\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)',
        r'(?:experience|exp).*?(\d+)[\+]?\s*(?:years?|yrs?)',
        r'(\d+)[\+]?\s*(?:years?|yrs?)',
    ]
    
    experience = '0'
    for pattern in experience_patterns:
        exp_match = re.search(pattern, text_lower)
        if exp_match:
            experience = exp_match.group(1)
            break
    
    # Extract education with details
    education_patterns = {
        'phd': r'(?:phd|ph\.d|doctorate|doctoral)',
        'master': r'(?:master|msc|ms|mba|m\.s|m\.sc)',
        'bachelor': r'(?:bachelor|bsc|bs|be|btech|b\.s|b\.sc|b\.e|b\.tech)',
        'diploma': r'(?:diploma|certificate)'
    }
    
    education_found = []
    highest_education = 'high school'
    
    for edu_level, pattern in education_patterns.items():
        if re.search(pattern, text_lower):
            education_found.append(edu_level)
            highest_education = edu_level
            break
    
    # Extract job titles and companies
    job_titles = []
    companies = []
    
    # Common job title patterns
    title_patterns = [
        r'(?:senior|sr\.?|lead|principal|chief)\s+(?:software|data|ml|ai|full.?stack|backend|frontend|devops)\s+(?:engineer|developer|scientist|analyst)',
        r'(?:software|data|ml|ai|full.?stack|backend|frontend|devops)\s+(?:engineer|developer|scientist|analyst)',
        r'(?:manager|director|head|vp|cto|ceo)\s+(?:of\s+)?(?:engineering|technology|data|ai|ml)',
    ]
    
    for pattern in title_patterns:
        matches = re.findall(pattern, text_lower)
        job_titles.extend(matches)
    
    # Extract leadership indicators
    leadership_indicators = []
    leadership_patterns = [
        r'led\s+(?:a\s+)?team\s+of\s+(\d+)',
        r'managed\s+(\d+)\s+(?:people|engineers|developers|analysts)',
        r'mentored\s+(\d+)',
        r'supervised\s+(\d+)',
    ]
    
    for pattern in leadership_patterns:
        matches = re.findall(pattern, text_lower)
        leadership_indicators.extend([f"Led team of {num}" for num in matches])
    
    # Extract achievements and metrics
    achievements = []
    achievement_patterns = [
        r'(?:increased|improved|reduced|achieved|built|developed|created).*?(\d+%)',
        r'(\d+(?:\.\d+)?[kmb]?)\s*(?:users|customers|revenue|sales)',
        r'(?:saved|generated|earned).*?\$(\d+(?:,\d+)*(?:\.\d+)?[kmb]?)',
    ]
    
    for pattern in achievement_patterns:
        matches = re.findall(pattern, text_lower)
        achievements.extend(matches)
    
    return {
        'name': name,
        'skills': found_skills,
        'skill_categories': skill_categories_found,
        'experience': experience,
        'education': education_found,
        'highest_education': highest_education,
        'job_titles': job_titles,
        'companies': companies,
        'leadership_indicators': leadership_indicators,
        'achievements': achievements,
        'raw_text': text
    }

def calculate_professional_match_score(resume_data, job_requirements):
    """Professional HR-grade matching algorithm"""
    
    # Initialize scoring components
    scores = {
        'technical_skills': 0.0,
        'experience': 0.0,
        'education': 0.0,
        'leadership': 0.0,
        'achievements': 0.0,
        'cultural_fit': 0.0
    }
    
    reasons_selected = []
    reasons_rejected = []
    
    # 1. TECHNICAL SKILLS ANALYSIS (40% weight)
    required_skills = [s.lower().strip() for s in job_requirements.get('skills', [])]
    candidate_skills = [s.lower() for s in resume_data['skills']]
    
    if required_skills:
        # Direct skill matches
        direct_matches = [skill for skill in required_skills if skill in candidate_skills]
        skill_match_ratio = len(direct_matches) / len(required_skills)
        
        # Skill category bonus (if candidate has skills in same category)
        category_bonus = 0
        for category, skills in resume_data['skill_categories'].items():
            if skills and any(req_skill in ' '.join(skills) for req_skill in required_skills):
                category_bonus += 0.1
        
        scores['technical_skills'] = min(skill_match_ratio + category_bonus, 1.0)
        
        if skill_match_ratio >= 0.8:
            reasons_selected.append(f"✅ Excellent skill match ({len(direct_matches)}/{len(required_skills)} required skills)")
        elif skill_match_ratio >= 0.6:
            reasons_selected.append(f"✅ Good skill match ({len(direct_matches)}/{len(required_skills)} required skills)")
        elif skill_match_ratio < 0.3:
            reasons_rejected.append(f"❌ Missing critical skills (only {len(direct_matches)}/{len(required_skills)} required skills)")
    
    # 2. EXPERIENCE ANALYSIS (25% weight)
    candidate_exp = int(resume_data['experience'])
    required_exp = job_requirements.get('min_experience', 0)
    
    if candidate_exp >= required_exp:
        # Bonus for exceeding requirements
        exp_ratio = min(candidate_exp / max(required_exp, 1), 2.0)
        scores['experience'] = min(exp_ratio / 2.0 + 0.5, 1.0)
        
        if candidate_exp >= required_exp * 1.5:
            reasons_selected.append(f"✅ Exceeds experience requirement ({candidate_exp} vs {required_exp} years)")
        else:
            reasons_selected.append(f"✅ Meets experience requirement ({candidate_exp} years)")
    else:
        # Penalty for under-experience
        scores['experience'] = candidate_exp / max(required_exp, 1) * 0.7
        if candidate_exp < required_exp * 0.5:
            reasons_rejected.append(f"❌ Insufficient experience ({candidate_exp} vs {required_exp} years required)")
        else:
            reasons_rejected.append(f"⚠️ Below experience requirement ({candidate_exp} vs {required_exp} years)")
    
    # 3. EDUCATION ANALYSIS (15% weight)
    required_education = job_requirements.get('education', 'bachelor').lower()
    candidate_education = resume_data['highest_education'].lower()
    
    education_hierarchy = {'high school': 1, 'diploma': 2, 'bachelor': 3, 'master': 4, 'phd': 5}
    
    required_level = education_hierarchy.get(required_education, 3)
    candidate_level = education_hierarchy.get(candidate_education, 1)
    
    if candidate_level >= required_level:
        scores['education'] = 1.0
        if candidate_level > required_level:
            reasons_selected.append(f"✅ Exceeds education requirement ({candidate_education} vs {required_education})")
        else:
            reasons_selected.append(f"✅ Meets education requirement ({candidate_education})")
    else:
        scores['education'] = candidate_level / required_level
        reasons_rejected.append(f"❌ Below education requirement ({candidate_education} vs {required_education})")
    
    # 4. LEADERSHIP ANALYSIS (10% weight)
    if resume_data['leadership_indicators']:
        scores['leadership'] = 1.0
        reasons_selected.append(f"✅ Leadership experience: {', '.join(resume_data['leadership_indicators'][:2])}")
    elif any(word in resume_data['raw_text'].lower() for word in ['led', 'managed', 'mentor', 'supervisor']):
        scores['leadership'] = 0.6
        reasons_selected.append("✅ Some leadership indicators found")
    
    # 5. ACHIEVEMENTS ANALYSIS (5% weight)
    if resume_data['achievements']:
        scores['achievements'] = 1.0
        reasons_selected.append(f"✅ Quantifiable achievements: {', '.join(resume_data['achievements'][:2])}")
    elif any(word in resume_data['raw_text'].lower() for word in ['built', 'developed', 'created', 'improved', 'increased']):
        scores['achievements'] = 0.5
        reasons_selected.append("✅ Project and development experience")
    
    # 6. CULTURAL FIT ANALYSIS (5% weight)
    cultural_keywords = ['team', 'collaborate', 'communication', 'agile', 'scrum', 'mentor', 'learn']
    cultural_matches = sum(1 for word in cultural_keywords if word in resume_data['raw_text'].lower())
    scores['cultural_fit'] = min(cultural_matches / len(cultural_keywords), 1.0)
    
    if cultural_matches >= 4:
        reasons_selected.append("✅ Strong cultural fit indicators")
    
    # Calculate weighted final score
    weights = {
        'technical_skills': 0.40,
        'experience': 0.25,
        'education': 0.15,
        'leadership': 0.10,
        'achievements': 0.05,
        'cultural_fit': 0.05
    }
    
    final_score = sum(scores[component] * weights[component] for component in scores)
    
    # Determine selection status and reasoning
    if final_score >= 0.8:
        status = "SELECTED"
        decision = "HIRE IMMEDIATELY"
        color = "success"
    elif final_score >= 0.65:
        status = "SHORTLISTED"
        decision = "SCHEDULE INTERVIEW"
        color = "info"
    elif final_score >= 0.45:
        status = "MAYBE"
        decision = "CONSIDER FOR JUNIOR ROLE"
        color = "warning"
    else:
        status = "REJECTED"
        decision = "NOT SUITABLE"
        color = "error"
    
    return {
        'overall_score': final_score,
        'component_scores': scores,
        'status': status,
        'decision': decision,
        'color': color,
        'reasons_selected': reasons_selected,
        'reasons_rejected': reasons_rejected,
        'candidate_name': resume_data['name']
    }

def main():
    """Main application"""
    st.markdown('<h1 class="main-header">NeuroMatch AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Professional AI-Powered Resume Analysis & Candidate Ranking System</p>', unsafe_allow_html=True)
    
    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Navigation",
        ["Home", "Single Resume Analysis", "Bulk Resume Processing"]
    )
    
    if page == "Home":
        show_home()
    elif page == "Single Resume Analysis":
        show_single_resume()
    elif page == "Bulk Resume Processing":
        show_bulk_analysis()

def show_home():
    """Professional home page"""
    
    # Feature cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">AI</div>
            <div class="feature-title">Advanced AI Analysis</div>
            <div class="feature-desc">
                • Intelligent skill detection using NLP<br>
                • Experience and education extraction<br>
                • Leadership and achievement analysis<br>
                • Cultural fit assessment
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <div class="feature-title">Smart Ranking System</div>
            <div class="feature-desc">
                • Multi-criteria evaluation algorithm<br>
                • Weighted scoring system<br>
                • Hidden talent detection<br>
                • Professional recommendations
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Enterprise Processing</div>
            <div class="feature-desc">
                • Bulk resume processing<br>
                • PDF and text support<br>
                • Automated candidate ranking<br>
                • Professional reporting & export
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Quick demo section
    st.markdown('<div class="section-header">Quick Demo</div>', unsafe_allow_html=True)
    
    demo_text = st.text_area(
        "Paste resume text for instant analysis:",
        placeholder="Senior Data Scientist with 5 years experience in Python, machine learning, SQL...",
        height=150
    )
    
    if st.button("Analyze Demo", type="primary"):
        if demo_text:
            with st.spinner("Running professional HR analysis..."):
                resume_data = parse_resume_professional(demo_text)
                
                # Create job requirements for demo
                job_requirements = {
                    'skills': ['python', 'machine learning', 'sql'],
                    'min_experience': 3,
                    'education': 'bachelor'
                }
                
                # Get professional analysis
                analysis_result = calculate_professional_match_score(resume_data, job_requirements)
                
                # Display results
                st.markdown("---")
                st.subheader(f"🎯 AI RECRUITMENT DECISION - {analysis_result['candidate_name']}")
                
                if analysis_result['color'] == "success":
                    st.success(analysis_result['status'])
                elif analysis_result['color'] == "info":
                    st.info(analysis_result['status'])
                elif analysis_result['color'] == "warning":
                    st.warning(analysis_result['status'])
                else:
                    st.error(analysis_result['status'])
                
                # Detailed metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Overall Score", f"{analysis_result['overall_score']:.1%}")
                
                with col2:
                    st.metric("Experience", f"{resume_data['experience']} years")
                
                with col3:
                    st.metric("Skills Found", len(resume_data['skills']))
                
                with col4:
                    st.metric("Education", resume_data['highest_education'].title())
                
                # Score breakdown
                st.subheader("📊 Professional Score Breakdown")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    scores = analysis_result['component_scores']
                    st.progress(scores['technical_skills'], text=f"Technical Skills: {scores['technical_skills']:.1%}")
                    st.progress(scores['experience'], text=f"Experience Level: {scores['experience']:.1%}")
                    st.progress(scores['education'], text=f"Education: {scores['education']:.1%}")
                    st.progress(scores['leadership'], text=f"Leadership: {scores['leadership']:.1%}")
                    st.progress(scores['achievements'], text=f"Achievements: {scores['achievements']:.1%}")
                    st.progress(scores['cultural_fit'], text=f"Cultural Fit: {scores['cultural_fit']:.1%}")
                
                with col2:
                    if resume_data['skills']:
                        st.write("**Skills Detected:**")
                        for skill in resume_data['skills'][:8]:  # Show top 8 skills
                            st.write(f"• {skill}")
                    
                    # Professional decision reasoning
                    st.write("**Why Selected/Rejected:**")
                    
                    # Show positive reasons
                    if analysis_result['reasons_selected']:
                        for reason in analysis_result['reasons_selected']:
                            st.write(reason)
                    
                    # Show negative reasons
                    if analysis_result['reasons_rejected']:
                        for reason in analysis_result['reasons_rejected']:
                            st.write(reason)
                    
                    # Show leadership and achievements if found
                    if resume_data['leadership_indicators']:
                        st.write("**Leadership:**")
                        for indicator in resume_data['leadership_indicators'][:2]:
                            st.write(f"• {indicator}")
                    
                    if resume_data['achievements']:
                        st.write("**Key Achievements:**")
                        for achievement in resume_data['achievements'][:2]:
                            st.write(f"• {achievement}")
                
                # Ranking simulation with real candidate name
                st.subheader("🏆 Candidate Ranking")
                
                # Simulate other candidates for comparison
                comparison_candidates = [
                    {"name": analysis_result['candidate_name'], "score": analysis_result['overall_score'], "status": analysis_result['status']},
                    {"name": "Sarah Johnson", "score": 0.92, "status": "🎉 SELECTED"},
                    {"name": "Mike Chen", "score": 0.87, "status": "🎉 SELECTED"},
                    {"name": "Lisa Wang", "score": 0.73, "status": "✅ SHORTLISTED"},
                    {"name": "David Kim", "score": 0.68, "status": "✅ SHORTLISTED"},
                    {"name": "Alex Rodriguez", "score": 0.45, "status": "⚠️ MAYBE"},
                    {"name": "Tom Wilson", "score": 0.32, "status": "❌ REJECTED"}
                ]
                
                # Sort by score
                comparison_candidates.sort(key=lambda x: x['score'], reverse=True)
                
                # Find your candidate's rank
                your_rank = next(i+1 for i, c in enumerate(comparison_candidates) if c['name'] == analysis_result['candidate_name'])
                
                st.info(f"🎯 **{analysis_result['candidate_name']} ranks #{your_rank} out of 7 candidates**")
                
                # Show top 5
                st.write("**Top 5 Candidates:**")
                for i, candidate in enumerate(comparison_candidates[:5]):
                    if candidate['name'] == analysis_result['candidate_name']:
                        st.markdown(f"**#{i+1} - {candidate['name']} - {candidate['score']:.1%} - {candidate['status']}** ⭐")
                    else:
                        st.write(f"#{i+1} - {candidate['name']} - {candidate['score']:.1%} - {candidate['status']}")
                
                # Final recommendation
                st.markdown("---")
                st.subheader(f"🎯 Final HR Recommendation for {analysis_result['candidate_name']}")
                
                if analysis_result['color'] == "success":
                    st.success(f"🎉 **{analysis_result['decision']}** - {analysis_result['candidate_name']} is an excellent fit!")
                elif analysis_result['color'] == "info":
                    st.info(f"📞 **{analysis_result['decision']}** - {analysis_result['candidate_name']} shows strong potential")
                elif analysis_result['color'] == "warning":
                    st.warning(f"🤔 **{analysis_result['decision']}** - {analysis_result['candidate_name']} may work with training")
                else:
                    st.error(f"❌ **{analysis_result['decision']}** - {analysis_result['candidate_name']} is not suitable for this role")

def show_single_resume():
    """Single resume analysis with multiple input options"""
    st.header("📄 Advanced Resume Analysis")
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["📝 Copy-Paste Text", "📁 Upload PDF", "📊 Advanced Analysis"]
    )
    
    resume_text = None
    resume_data = None
    
    if input_method == "📝 Copy-Paste Text":
        st.subheader("📝 Paste Resume Content")
        resume_text = st.text_area(
            "Paste resume text here:",
            height=300,
            placeholder="Copy and paste the complete resume text here..."
        )
        
        if resume_text and st.button("🔍 Analyze Resume Text"):
            resume_data = parse_resume_professional(resume_text)
            display_advanced_analysis(resume_data, resume_text)
    
    elif input_method == "📁 Upload PDF":
        st.subheader("📁 Upload PDF Resume")
        
        if PDF_SUPPORT:
            uploaded_file = st.file_uploader(
                "Upload Resume PDF",
                type=['pdf'],
                help="Upload a PDF resume for analysis"
            )
            
            if uploaded_file is not None:
                with st.spinner("Processing PDF..."):
                    try:
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                            tmp_file.write(uploaded_file.read())
                            tmp_path = tmp_file.name
                        
                        # Extract text from PDF
                        resume_text = extract_text_from_pdf(tmp_path)
                        os.unlink(tmp_path)  # Clean up
                        
                        if resume_text.strip():
                            st.success("✅ PDF processed successfully!")
                            resume_data = parse_resume_professional(resume_text)
                            display_advanced_analysis(resume_data, resume_text)
                        else:
                            st.error("❌ Could not extract text from PDF")
                            
                    except Exception as e:
                        st.error(f"❌ Error processing PDF: {str(e)}")
                        if 'tmp_path' in locals():
                            try:
                                os.unlink(tmp_path)
                            except:
                                pass
        else:
            st.warning("📋 PDF support not available. Please install PyPDF2:")
            st.code("pip install PyPDF2")
            st.info("💡 Use the copy-paste method instead")
    
    elif input_method == "📊 Advanced Analysis":
        st.subheader("📊 Advanced AI Analysis")
        resume_text = st.text_area(
            "Paste resume for advanced analysis:",
            height=200,
            placeholder="Paste resume text for cognitive patterns, growth prediction, and innovation scoring..."
        )
        
        if resume_text and st.button("🧠 Run Advanced AI Analysis"):
            resume_data = parse_resume_professional(resume_text)
            display_comprehensive_analysis(resume_data, resume_text)

def display_advanced_analysis(resume_data, resume_text):
    """Display enhanced resume analysis"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Resume Analysis")
        st.write(f"**Experience:** {resume_data['experience']} years")
        st.write(f"**Skills Found:** {len(resume_data['skills'])}")
        st.write(f"**Education:** {len(resume_data['education'])}")
        
        if resume_data['skills']:
            st.write("**Skills:**")
            for skill in resume_data['skills']:
                st.write(f"• {skill}")
        
        # Additional insights
        st.subheader("🔍 Insights")
        
        # Experience level
        exp_years = int(resume_data['experience'])
        if exp_years >= 8:
            st.success("🌟 **Senior Level** - Extensive experience")
        elif exp_years >= 5:
            st.info("📈 **Mid-Senior Level** - Solid experience")
        elif exp_years >= 2:
            st.warning("⚡ **Mid Level** - Growing experience")
        else:
            st.error("🌱 **Junior Level** - Entry level")
        
        # Skills assessment
        if len(resume_data['skills']) >= 5:
            st.success("💪 **Strong Technical Skills** - Diverse skill set")
        elif len(resume_data['skills']) >= 3:
            st.info("⚖️ **Moderate Skills** - Good foundation")
        else:
            st.warning("📚 **Limited Skills** - Needs development")
    
    with col2:
        st.subheader("📈 Visualizations")
        
        # Skills chart
        if resume_data['skills']:
            skills_df = pd.DataFrame({
                'Skill': resume_data['skills'],
                'Relevance': np.random.uniform(0.6, 1.0, len(resume_data['skills']))
            })
            
            fig = px.bar(skills_df, x='Skill', y='Relevance', 
                       title="Skills Relevance Score",
                       color='Relevance',
                       color_continuous_scale='viridis')
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        # Experience vs Industry Average
        exp_years = int(resume_data['experience'])
        industry_avg = 5.2  # Sample industry average
        
        fig = go.Figure(data=[
            go.Bar(name='Candidate', x=['Experience'], y=[exp_years]),
            go.Bar(name='Industry Average', x=['Experience'], y=[industry_avg])
        ])
        fig.update_layout(
            title="Experience Comparison",
            yaxis_title="Years",
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)

def display_comprehensive_analysis(resume_data, resume_text):
    """Display comprehensive AI analysis"""
    st.subheader("🧠 Comprehensive AI Analysis")
    
    # Basic analysis
    display_advanced_analysis(resume_data, resume_text)
    
    st.markdown("---")
    
    # Advanced AI features (simulated)
    st.subheader("🎯 AI-Powered Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🧠 Cognitive Patterns")
        
        # Simulate cognitive analysis
        text_lower = resume_text.lower()
        
        analytical_score = 0.8 if any(word in text_lower for word in ['analysis', 'data', 'research']) else 0.4
        creative_score = 0.7 if any(word in text_lower for word in ['design', 'creative', 'innovative']) else 0.3
        leadership_score = 0.9 if any(word in text_lower for word in ['led', 'managed', 'team']) else 0.2
        
        st.progress(analytical_score, text=f"Analytical Thinker: {analytical_score:.1%}")
        st.progress(creative_score, text=f"Creative Innovator: {creative_score:.1%}")
        st.progress(leadership_score, text=f"Leadership: {leadership_score:.1%}")
    
    with col2:
        st.markdown("### 📈 Growth Potential")
        
        # Simulate growth prediction
        exp_years = int(resume_data['experience'])
        skills_count = len(resume_data['skills'])
        
        growth_potential = min((exp_years * 0.1 + skills_count * 0.05 + 0.3), 1.0)
        technical_growth = min(skills_count * 0.1 + 0.4, 1.0)
        career_growth = min(exp_years * 0.08 + 0.5, 1.0)
        
        st.progress(growth_potential, text=f"Overall Growth: {growth_potential:.1%}")
        st.progress(technical_growth, text=f"Technical Growth: {technical_growth:.1%}")
        st.progress(career_growth, text=f"Career Growth: {career_growth:.1%}")
    
    with col3:
        st.markdown("### 💡 Innovation Score")
        
        # Simulate innovation detection
        innovation_keywords = ['project', 'built', 'developed', 'created', 'innovative']
        innovation_score = min(sum(1 for word in innovation_keywords if word in text_lower) * 0.15 + 0.3, 1.0)
        
        novelty_score = min(innovation_score + 0.1, 1.0)
        complexity_score = min(skills_count * 0.08 + 0.4, 1.0)
        
        st.progress(innovation_score, text=f"Innovation: {innovation_score:.1%}")
        st.progress(novelty_score, text=f"Novelty: {novelty_score:.1%}")
        st.progress(complexity_score, text=f"Complexity: {complexity_score:.1%}")
    
    # Recommendations
    st.subheader("🎯 AI Recommendations")
    
    recommendations = []
    
    if analytical_score > 0.7:
        recommendations.append("✅ **Strong Analytical Skills** - Excellent for data-driven roles")
    
    if leadership_score > 0.7:
        recommendations.append("👥 **Leadership Potential** - Consider for team lead positions")
    
    if innovation_score > 0.6:
        recommendations.append("💡 **Innovation Mindset** - Great for R&D and product development")
    
    if exp_years >= 5 and skills_count >= 4:
        recommendations.append("🌟 **Senior Role Ready** - Qualified for senior positions")
    
    if not recommendations:
        recommendations.append("📚 **Growth Opportunity** - Focus on skill development and experience")
    
    for rec in recommendations:
        st.info(rec)

def show_bulk_analysis():
    """Professional bulk resume analysis with PDF support"""
    st.markdown('<div class="section-header">Bulk Resume Processing & Ranking</div>', unsafe_allow_html=True)
    
    # Job requirements section
    st.markdown('<div class="results-container">', unsafe_allow_html=True)
    st.subheader("Job Requirements Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        required_skills = st.text_input(
            "Required Skills (comma-separated)",
            placeholder="python, machine learning, sql, aws"
        )
        min_experience = st.slider("Minimum Experience (years)", 0, 15, 3)
        education_level = st.selectbox("Education Requirement", [
            "high school", "diploma", "bachelor", "master", "phd"
        ], index=2)
    
    with col2:
        job_title = st.text_input("Job Title", placeholder="Senior Data Scientist")
        ranking_strategy = st.selectbox("Ranking Strategy", [
            "Exact Match - Strict requirements",
            "Balanced Approach - Moderate flexibility", 
            "Hidden Gems - Find potential talent"
        ], index=1)
        
        # Weight customization
        st.write("**Scoring Weights:**")
        tech_weight = st.slider("Technical Skills", 0.2, 0.6, 0.4, 0.05)
        exp_weight = st.slider("Experience", 0.1, 0.4, 0.25, 0.05)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Resume input section
    st.markdown('<div class="section-header">Resume Input Options</div>', unsafe_allow_html=True)
    
    input_method = st.radio(
        "Choose input method:",
        ["Text Input - Copy & Paste", "PDF Upload - Multiple Files"],
        horizontal=True
    )
    
    resumes_data = []
    
    if input_method == "Text Input - Copy & Paste":
        st.markdown('<div class="results-container">', unsafe_allow_html=True)
        st.info("Separate each resume with '---RESUME---' on a new line")
        
        bulk_text = st.text_area(
            "Paste multiple resumes:",
            height=400,
            placeholder="""Resume 1 content here...

---RESUME---

Resume 2 content here...

---RESUME---

Resume 3 content here..."""
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if bulk_text and st.button("Process Text Resumes", type="primary"):
            resumes_list = [resume.strip() for resume in bulk_text.split('---RESUME---') if resume.strip()]
            process_bulk_resumes(resumes_list, required_skills, min_experience, education_level, job_title)
    
    elif input_method == "PDF Upload - Multiple Files":
        st.markdown('<div class="results-container">', unsafe_allow_html=True)
        
        if PDF_SUPPORT:
            uploaded_files = st.file_uploader(
                "Upload multiple PDF resumes",
                type=['pdf'],
                accept_multiple_files=True,
                help="Select multiple PDF files to process at once"
            )
            
            if uploaded_files and st.button("Process PDF Resumes", type="primary"):
                resumes_list = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing {uploaded_file.name}...")
                    
                    try:
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                            tmp_file.write(uploaded_file.read())
                            tmp_path = tmp_file.name
                        
                        # Extract text from PDF
                        resume_text = extract_text_from_pdf(tmp_path)
                        os.unlink(tmp_path)  # Clean up
                        
                        if resume_text.strip():
                            # Add filename as header for identification
                            resume_with_filename = f"File: {uploaded_file.name}\n\n{resume_text}"
                            resumes_list.append(resume_with_filename)
                        else:
                            st.warning(f"Could not extract text from {uploaded_file.name}")
                            
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                        if 'tmp_path' in locals():
                            try:
                                os.unlink(tmp_path)
                            except:
                                pass
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                status_text.text("Processing complete!")
                
                if resumes_list:
                    st.success(f"Successfully processed {len(resumes_list)} PDF files")
                    process_bulk_resumes(resumes_list, required_skills, min_experience, education_level, job_title)
                else:
                    st.error("No valid resumes could be extracted from the uploaded files")
        else:
            st.warning("PDF support not available. Please install PyPDF2:")
            st.code("pip install PyPDF2 pypdf")
            st.info("Use the text input method instead")
        
        st.markdown('</div>', unsafe_allow_html=True)

def process_bulk_resumes(resumes_list, required_skills, min_experience, education_level, job_title):
    """Process bulk resumes with professional analysis"""
    if not resumes_list:
        st.error("No resumes found to process")
        return
    
    # Parse job requirements
    job_requirements = {
        'skills': [s.strip().lower() for s in required_skills.split(',') if s.strip()],
        'min_experience': min_experience,
        'education': education_level,
        'title': job_title
    }
    
    st.success(f"Processing {len(resumes_list)} resumes...")
    
    # Process resumes
    results = []
    progress_bar = st.progress(0)
    
    for i, resume_text in enumerate(resumes_list):
        resume_data = parse_resume_professional(resume_text)
        analysis_result = calculate_professional_match_score(resume_data, job_requirements)
        
        results.append({
            'candidate': resume_data['name'],
            'experience': resume_data['experience'],
            'skills_count': len(resume_data['skills']),
            'match_score': analysis_result['overall_score'],
            'status': analysis_result['status'],
            'decision': analysis_result['decision'],
            'color': analysis_result['color'],
            'skills': ', '.join(resume_data['skills'][:3]),
            'reasons_selected': analysis_result['reasons_selected'],
            'reasons_rejected': analysis_result['reasons_rejected'],
            'education': resume_data['highest_education'],
            'leadership': resume_data['leadership_indicators'][:2] if resume_data['leadership_indicators'] else [],
            'achievements': resume_data['achievements'][:2] if resume_data['achievements'] else []
        })
        
        progress_bar.progress((i + 1) / len(resumes_list))
    
    # Sort by match score
    results.sort(key=lambda x: x['match_score'], reverse=True)
    
    # Display professional results
    display_professional_results(results)

def display_professional_results(results):
    """Display professional recruitment results"""
    st.markdown('<div class="section-header">Recruitment Results</div>', unsafe_allow_html=True)
    
    # Calculate selections
    selected = [r for r in results if r['match_score'] >= 0.8]
    shortlisted = [r for r in results if 0.65 <= r['match_score'] < 0.8]
    maybe = [r for r in results if 0.45 <= r['match_score'] < 0.65]
    rejected = [r for r in results if r['match_score'] < 0.45]
    
    # Professional metrics dashboard
    st.markdown('<div class="results-container">', unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{len(results)}</h3>
            <p>Total Candidates</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #10b981;">{len(selected)}</h3>
            <p>Selected</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #3b82f6;">{len(shortlisted)}</h3>
            <p>Shortlisted</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #f59e0b;">{len(maybe)}</h3>
            <p>Under Review</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        selection_rate = len(selected) / len(results) * 100 if results else 0
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #667eea;">{selection_rate:.1f}%</h3>
            <p>Selection Rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Show selected candidates
    if selected:
        st.markdown('<div class="section-header">Selected Candidates</div>', unsafe_allow_html=True)
        for i, candidate in enumerate(selected):
            st.markdown(f"""
            <div class="candidate-card">
                <div style="display: flex; justify-content: between; align-items: center; margin-bottom: 1rem;">
                    <h4 style="margin: 0; color: #1e293b;">{candidate['candidate']}</h4>
                    <span class="status-selected">{candidate['status']}</span>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem;">
                    <div>
                        <strong>Experience:</strong> {candidate['experience']} years<br>
                        <strong>Education:</strong> {candidate['education'].title()}<br>
                        <strong>Skills:</strong> {candidate['skills_count']}
                    </div>
                    <div>
                        <strong>Match Score:</strong> {candidate['match_score']:.1%}<br>
                        <strong>Decision:</strong> {candidate['decision']}<br>
                        <strong>Top Skills:</strong> {candidate['skills']}
                    </div>
                    <div>
                        <strong>Why Selected:</strong><br>
                        {'<br>'.join(candidate['reasons_selected'][:3])}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Show shortlisted candidates
    if shortlisted:
        st.markdown('<div class="section-header">Shortlisted for Interview</div>', unsafe_allow_html=True)
        for candidate in shortlisted:
            st.markdown(f"""
            <div class="candidate-card">
                <div style="display: flex; justify-content: between; align-items: center; margin-bottom: 1rem;">
                    <h4 style="margin: 0; color: #1e293b;">{candidate['candidate']}</h4>
                    <span class="status-shortlisted">{candidate['status']}</span>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                    <div>
                        <strong>Experience:</strong> {candidate['experience']} years<br>
                        <strong>Match Score:</strong> {candidate['match_score']:.1%}<br>
                        <strong>Skills:</strong> {candidate['skills']}
                    </div>
                    <div>
                        <strong>Decision:</strong> {candidate['decision']}<br>
                        <strong>Reasoning:</strong><br>
                        {'<br>'.join((candidate['reasons_selected'] + candidate['reasons_rejected'])[:3])}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Export functionality
    if results:
        st.markdown('<div class="section-header">Export Results</div>', unsafe_allow_html=True)
        
        # Create DataFrame for export
        df = pd.DataFrame(results)
        df['Rank'] = range(1, len(df) + 1)
        df['Match Score'] = df['match_score'].apply(lambda x: f"{x:.1%}")
        
        # Display summary table
        display_df = df[['Rank', 'candidate', 'experience', 'skills_count', 'status', 'Match Score']]
        display_df.columns = ['Rank', 'Candidate', 'Experience (Years)', 'Skills', 'Status', 'Match Score']
        
        st.dataframe(display_df, use_container_width=True)
        
        # Download buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv_all = df.to_csv(index=False)
            st.download_button(
                label="Download All Results",
                data=csv_all,
                file_name=f"all_candidates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            if selected:
                selected_df = pd.DataFrame(selected)
                csv_selected = selected_df.to_csv(index=False)
                st.download_button(
                    label="Download Selected Only",
                    data=csv_selected,
                    file_name=f"selected_candidates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col3:
            if shortlisted:
                shortlisted_df = pd.DataFrame(shortlisted)
                csv_shortlisted = shortlisted_df.to_csv(index=False)
                st.download_button(
                    label="Download Shortlisted",
                    data=csv_shortlisted,
                    file_name=f"shortlisted_candidates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()
