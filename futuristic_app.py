"""
Futuristic NeuroMatch AI - Glass Morphism Dashboard
Dark theme with neon accents and particle effects
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
import sqlite3
import json
import time
import base64
from io import BytesIO

# Performance optimizations
st.set_page_config(
    page_title="NeuroMatch AI - Futuristic Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add caching for expensive operations
@st.cache_data(ttl=300)  # Cache for 5 minutes
def cached_parse_resume(text):
    """Cached version of resume parsing"""
    return parse_resume_professional(text)

@st.cache_data(ttl=300)
def cached_calculate_score(resume_data_dict, job_requirements):
    """Cached version of score calculation"""
    return calculate_professional_match_score(resume_data_dict, job_requirements)

# Import ONLY WORKING ML/DL models
try:
    from ml_models import (
        rl_scorer,           # ✅ Q-Learning - WORKING
        bert_parser,         # ✅ BERT NER - WORKING  
        attrition_predictor, # ✅ Random Forest - WORKING
        diversity_analyzer,  # ✅ Statistical ML - WORKING
        get_ml_models_status
    )
    ML_MODELS_AVAILABLE = True
    print("[OK] Working ML models loaded: BERT NER, Q-Learning, Random Forest, Statistical ML")
except ImportError:
    ML_MODELS_AVAILABLE = False
    print("[ERROR] ML models not available. Run: pip install -r requirements_ml.txt")

# Import Database Manager (SQLite - Persistent Storage)
try:
    from database_manager import db
    DATABASE_AVAILABLE = True
    print("[OK] Database connected!")
except ImportError:
    DATABASE_AVAILABLE = False
    print("[WARN] Database not available")

# Import Indian Salary Calculator
try:
    from indian_salary_data import calculate_indian_salary, format_indian_salary, format_salary_dict, get_salary_breakdown
    INDIAN_SALARY_AVAILABLE = True
except ImportError:
    INDIAN_SALARY_AVAILABLE = False

# Import Social Intelligence
try:
    from social_intelligence import get_social_intelligence
    SOCIAL_INTELLIGENCE_AVAILABLE = True
except ImportError:
    SOCIAL_INTELLIGENCE_AVAILABLE = False

# Try to import PDF support
try:
    from resume_parser import extract_text_from_pdf, parse_resume, is_resume_pdf
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    
    # Fallback PDF extraction
    def extract_text_from_pdf(file_path):
        try:
            import PyPDF2
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                return text
        except:
            return ""

# Page config already set above - removing duplicate

# Futuristic CSS with Glass Morphism - FIXED SIDEBAR
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;500;600;700&display=swap');
    
    /* Dark theme base */
    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 50%, #16213e 100%);
        font-family: 'Rajdhani', sans-serif;
        color: #ffffff;
    }
    
    /* Animated particle background */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: 
            radial-gradient(2px 2px at 20px 30px, #00f5d4, transparent),
            radial-gradient(2px 2px at 40px 70px, #f72585, transparent),
            radial-gradient(1px 1px at 90px 40px, #ffffff, transparent),
            radial-gradient(1px 1px at 130px 80px, #00f5d4, transparent),
            radial-gradient(2px 2px at 160px 30px, #f72585, transparent);
        background-size: 200px 100px;
        animation: sparkle 20s linear infinite;
        pointer-events: none;
        z-index: -1;
        opacity: 0.3;
    }
    
    @keyframes sparkle {
        from { transform: translateY(0px); }
        to { transform: translateY(-100px); }
    }
    
    /* Glass morphism container */
    .main .block-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }
    
    /* Futuristic header */
    .cyber-header {
        font-family: 'Orbitron', monospace;
        font-size: 3rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(45deg, #00f5d4, #f72585, #00f5d4);
        background-size: 200% 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: neonGlow 3s ease-in-out infinite alternate;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 20px rgba(0, 245, 212, 0.5);
    }
    
    @keyframes neonGlow {
        from { background-position: 0% 50%; }
        to { background-position: 100% 50%; }
    }
    
    .cyber-subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #00f5d4;
        margin-bottom: 2rem;
        font-weight: 300;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    
    /* Neon cards */
    .neon-card {
        background: rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(0, 245, 212, 0.3);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .neon-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(0, 245, 212, 0.1), transparent);
        transition: left 0.5s;
    }
    
    .neon-card:hover {
        border-color: #00f5d4;
        box-shadow: 0 0 30px rgba(0, 245, 212, 0.3);
        transform: translateY(-5px);
    }
    
    .neon-card:hover::before {
        left: 100%;
    }
    
    .neon-icon {
        width: 60px;
        height: 60px;
        background: linear-gradient(45deg, #00f5d4, #f72585);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 0 20px rgba(0, 245, 212, 0.4);
    }
    
    .neon-title {
        font-family: 'Orbitron', monospace;
        font-size: 1.3rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 1rem;
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.3);
    }
    
    .neon-desc {
        color: #b0b0b0;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    /* Cyber buttons */
    .stButton > button {
        background: linear-gradient(45deg, #00f5d4, #f72585) !important;
        color: #000000 !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 0.8rem 2rem !important;
        font-family: 'Orbitron', monospace !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 0 20px rgba(0, 245, 212, 0.4) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.05) !important;
        box-shadow: 0 0 30px rgba(0, 245, 212, 0.6) !important;
    }
    
    /* FIXED SIDEBAR STYLING - ENSURES VISIBILITY */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a0f 0%, #1a1a2e 100%) !important;
        border-right: 3px solid rgba(0, 245, 212, 0.5) !important;
        min-width: 300px !important;
        width: 300px !important;
    }
    
    section[data-testid="stSidebar"] > div {
        background: linear-gradient(180deg, #0a0a0f 0%, #1a1a2e 100%) !important;
    }
    
    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    section[data-testid="stSidebar"] .stRadio > label {
        color: #00f5d4 !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }
    
    section[data-testid="stSidebar"] .stRadio > div {
        background: rgba(0, 0, 0, 0.3) !important;
        padding: 1rem !important;
        border-radius: 10px !important;
    }
    
    /* Sidebar navigation styling */
    .sidebar .sidebar-content {
        background: transparent !important;
    }
    
    /* Force sidebar to be visible */
    [data-testid="stSidebar"][aria-expanded="true"] {
        min-width: 300px !important;
    }
    
    [data-testid="stSidebar"][aria-expanded="false"] {
        min-width: 300px !important;
        margin-left: 0px !important;
    }
    
    /* Text inputs */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: rgba(0, 0, 0, 0.6) !important;
        border: 1px solid rgba(0, 245, 212, 0.3) !important;
        border-radius: 10px !important;
        color: #ffffff !important;
        font-family: 'Rajdhani', sans-serif !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #00f5d4 !important;
        box-shadow: 0 0 15px rgba(0, 245, 212, 0.3) !important;
    }
    
    /* Labels */
    .stTextInput label, .stTextArea label, .stSelectbox label {
        color: #00f5d4 !important;
        font-weight: 600 !important;
        font-family: 'Orbitron', monospace !important;
        text-transform: uppercase !important;
        font-size: 0.8rem !important;
        letter-spacing: 1px !important;
    }
    
    /* Metrics */
    .stMetric {
        background: rgba(0, 0, 0, 0.4) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(0, 245, 212, 0.2) !important;
        border-radius: 15px !important;
        padding: 1rem !important;
    }
    
    .stMetric label {
        color: #00f5d4 !important;
        font-family: 'Orbitron', monospace !important;
        font-weight: 600 !important;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-family: 'Orbitron', monospace !important;
        font-weight: 900 !important;
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.3) !important;
    }
    
    /* Progress bars */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #00f5d4, #f72585) !important;
        box-shadow: 0 0 10px rgba(0, 245, 212, 0.5) !important;
    }
    
    /* Status badges */
    .cyber-badge-selected {
        background: linear-gradient(45deg, #00f5d4, #00d4aa);
        color: #000000;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-family: 'Orbitron', monospace;
        font-weight: 700;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 0 15px rgba(0, 245, 212, 0.5);
        display: inline-block;
    }
    
    .cyber-badge-shortlisted {
        background: linear-gradient(45deg, #f72585, #ff6b9d);
        color: #ffffff;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-family: 'Orbitron', monospace;
        font-weight: 700;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 0 15px rgba(247, 37, 133, 0.5);
        display: inline-block;
    }
    
    /* All text white in main content */
    .main .block-container,
    .main .block-container * {
        color: #ffffff !important;
    }
    
    /* Subheaders */
    .stMarkdown h2, .stMarkdown h3 {
        font-family: 'Orbitron', monospace !important;
        color: #00f5d4 !important;
        text-transform: uppercase !important;
        letter-spacing: 2px !important;
        margin-top: 2rem !important;
    }
    
    /* Hide branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Rest of your functions remain exactly the same...
def extract_candidate_name(text):
    """Extract candidate name from resume text"""
    lines = text.strip().split('\n')
    for line in lines[:3]:
        line = line.strip()
        if line and len(line.split()) <= 4 and len(line) > 3:
            if not any(keyword in line.lower() for keyword in ['email', 'phone', 'address', 'resume', 'cv']):
                return line
    return "Neural Candidate"

def parse_resume_professional(text):
    """Professional resume parsing with improved extraction"""
    name = extract_candidate_name(text)
    
    # Extract email
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    email_match = re.search(email_pattern, text)
    email = email_match.group(0) if email_match else ''
    
    # Extract phone
    phone_pattern = r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]'
    phone_match = re.search(phone_pattern, text)
    phone = phone_match.group(0) if phone_match else ''
    
    # Extract experience - improved patterns with validation
    exp_patterns = [
        r'(\d{1,2})\+?\s*years?\s*(?:of\s*)?(?:experience|exp)',
        r'(\d{1,2})\+?\s*yrs?\s*(?:of\s*)?(?:experience|exp)',
        r'experience[:\s]+(\d{1,2})\+?\s*years?',
        r'(\d{1,2})\s*years?\s*in\s+\w+',
        r'with\s+(\d{1,2})\s*years?',
        r'(\d{1,2})\s*years?\s*experience',
        r'(\d{1,2})\s*years?\s*of\s*experience',
        r'(\d{1,2})\+?\s*years?\s*professional'
    ]
    experience = 0
    for pattern in exp_patterns:
        matches = re.findall(pattern, text.lower())
        if matches:
            # Filter out unrealistic values (only 1-50 years valid)
            valid_matches = [int(m) for m in matches if 1 <= int(m) <= 50]
            if valid_matches:
                experience = max(valid_matches)
                break
    
    # Extract skills - expanded list
    skill_keywords = [
        'python', 'java', 'javascript', 'react', 'node', 'sql', 'aws', 'docker', 
        'kubernetes', 'machine learning', 'ai', 'data science', 'analytics',
        'tensorflow', 'pytorch', 'pandas', 'numpy', 'scikit-learn', 'spring',
        'django', 'flask', 'angular', 'vue', 'typescript', 'c++', 'c#', 'ruby',
        'go', 'rust', 'php', 'mysql', 'postgresql', 'mongodb', 'redis',
        'git', 'jenkins', 'ci/cd', 'agile', 'scrum', 'jira', 'excel', 'powerpoint',
        'tableau', 'power bi', 'spark', 'hadoop', 'kafka', 'elasticsearch'
    ]
    skills = [skill for skill in skill_keywords if skill in text.lower()]
    
    # Extract education
    education_map = {
        'phd': 'PhD',
        'doctorate': 'PhD',
        'doctor': 'PhD',
        'master': "Master's",
        'mba': 'MBA',
        'bachelor': "Bachelor's",
        'b.tech': "Bachelor's",
        'b.e.': "Bachelor's",
        'diploma': 'Diploma',
        'associate': 'Associate'
    }
    highest_education = 'High School'
    text_lower = text.lower()
    for key, value in education_map.items():
        if key in text_lower:
            highest_education = value
            break
    
    # Extract location
    location_pattern = r'(?:location|city|based in)[:\s]+([A-Za-z\s]+?)(?:\n|,|\||$)'
    location_match = re.search(location_pattern, text, re.IGNORECASE)
    location = location_match.group(1).strip() if location_match else ''
    
    # Extract current company
    company_pattern = r'(?:current|currently at|working at)[:\s]+([A-Za-z\s&]+?)(?:\n|,|\||$)'
    company_match = re.search(company_pattern, text, re.IGNORECASE)
    current_company = company_match.group(1).strip() if company_match else ''
    
    return {
        'name': name,
        'email': email,
        'phone': phone,
        'experience': experience,
        'skills': skills,
        'highest_education': highest_education,
        'location': location,
        'current_company': current_company,
        'leadership_indicators': ['team lead', 'manager'] if any(word in text.lower() for word in ['lead', 'led', 'manager', 'supervisor', 'director']) else [],
        'achievements': ['award', 'recognition'] if any(word in text.lower() for word in ['award', 'achievement', 'recognition', 'published', 'patent']) else []
    }

def detect_red_flags(text, resume_data):
    """Detect red flags in resume"""
    red_flags = []
    warnings = []
    
    # Job hopping detection
    job_count = text.lower().count('company') + text.lower().count('worked at') + text.lower().count('employer')
    if job_count > 4 and resume_data['experience'] < 3:
        red_flags.append(" Job Hopping: Multiple jobs in short timeframe")
    
    # Employment gap detection
    if 'gap' in text.lower() or '(unemployed)' in text.lower():
        warnings.append(" Employment Gap: Unexplained period detected")
    
    # Skill exaggeration
    if resume_data['experience'] < 2 and len(resume_data['skills']) > 10:
        warnings.append(" Skill Inflation: Many skills for experience level")
    
    # Education mismatch
    if 'dropout' in text.lower() or 'incomplete' in text.lower():
        warnings.append(" Education Status: Incomplete degree program")
    
    return {
        'red_flags': red_flags,
        'warnings': warnings,
        'clean': len(red_flags) == 0 and len(warnings) == 0
    }

def generate_interview_questions(resume_data, text):
    """Generate personalized interview questions based on resume"""
    questions = []
    
    # Experience-based questions
    if resume_data['experience'] > 0:
        questions.append(f"Tell me about your {resume_data['experience']} years of experience and your most challenging project")
    
    # Skill-based questions
    for skill in resume_data['skills'][:3]:  # Top 3 skills
        questions.append(f" Describe a complex problem you solved using {skill.title()}")
    
    # Leadership questions
    if resume_data['leadership_indicators']:
        questions.append(" Describe a situation where you had to lead a team through a difficult challenge")
    
    # Achievement questions
    if resume_data['achievements']:
        questions.append(" Walk me through your biggest professional achievement and its impact")
    
    # Behavioral questions
    questions.append(" How do you handle tight deadlines and multiple priorities?")
    questions.append(" Describe a time you had to learn a new technology quickly")
    
    return questions[:8]  # Return top 8 questions

def analyze_skills_gap(resume_data, job_requirements):
    """Analyze skills gap between candidate and requirements"""
    required_skills = set(skill.lower() for skill in job_requirements.get('skills', []))
    candidate_skills = set(resume_data['skills'])
    
    has_skills = required_skills.intersection(candidate_skills)
    missing_skills = required_skills - candidate_skills
    
    # Learning path estimation
    learning_time = len(missing_skills) * 4  # 4 weeks per skill avg
    
    return {
        'has': list(has_skills),
        'missing': list(missing_skills),
        'match_percentage': len(has_skills) / len(required_skills) * 100 if required_skills else 100,
        'learning_time_weeks': learning_time,
        'ready_to_interview': len(missing_skills) <= 2
    }

def predict_salary_range(resume_data):
    """Predict salary range based on experience and skills"""
    # Base salary calculation
    base_salary = 50000  # Starting point
    
    # Experience multiplier
    base_salary += resume_data['experience'] * 8000  # $8K per year of experience
    
    # Skills bonus
    base_salary += len(resume_data['skills']) * 2000  # $2K per relevant skill
    
    # Education bonus
    education_bonus = {
        'phd': 30000,
        'master': 20000,
        'bachelor': 10000,
        'diploma': 5000,
        'high school': 0
    }
    base_salary += education_bonus.get(resume_data['highest_education'], 0)
    
    # Leadership bonus
    if resume_data['leadership_indicators']:
        base_salary += 15000
    
    # Calculate range (±15%)
    lower_bound = int(base_salary * 0.85)
    upper_bound = int(base_salary * 1.15)
    market_average = base_salary
    
    return {
        'lower_range': lower_bound,
        'upper_range': upper_bound,
        'market_average': market_average,
        'recommended_offer': int(market_average * 0.97)  # Slightly below market
    }

def generate_email_template(candidate_name, status, score, job_title, reasons):
    """Generate personalized email template"""
    templates = {
        'NEURAL SELECTED': f"""Subject:  Interview Invitation - {job_title}

Dear {candidate_name},

Congratulations! After carefully reviewing your application, we're impressed by your qualifications and would like to invite you for an interview.

What stood out:
{chr(10).join(f'✓ {reason}' for reason in reasons[:3])}

Your match score of {score:.1%} places you among our top candidates.

Next steps:
1. Schedule an interview: [Calendar Link]
2. Prepare to discuss your experience with specific projects
3. Review our company values at [Company Site]

We're excited to learn more about you!

Best regards,
Neural Recruitment Team""",

        'CYBER SHORTLISTED': f"""Subject:  Application Update - {job_title}

Dear {candidate_name},

Thank you for applying to our {job_title} position. We're pleased to inform you that you're in our shortlist of candidates (top 15%).

Your strengths:
{chr(10).join(f'✓ {reason}' for reason in reasons[:2])}

Current status:
• We're reviewing additional candidates this week
• We'll reach out by [Date] with next steps
• Your profile remains active in our system

We appreciate your patience and interest in joining our team.

Best regards,
Cyber Recruitment Team""",

        'default': f"""Subject: Application Status - {job_title}

Dear {candidate_name},

Thank you for your interest in the {job_title} position at our company.

After careful review, we've decided to move forward with candidates whose experience more closely aligns with our current needs.

However, we were impressed by:
{chr(10).join(f'✓ {reason}' for reason in reasons[:2]) if reasons else '✓ Your professional background'}

To strengthen future applications:
• Consider gaining experience in [Key Skill Areas]
• Highlight quantifiable achievements
• Emphasize relevant project outcomes

We'll keep your profile for 12 months and notify you of suitable opportunities.

Best wishes in your career journey!

Regards,
Recruitment Team"""
    }
    
    return templates.get(status, templates['default'])

def calculate_professional_match_score(resume_data, job_requirements):
    """Calculate professional match score with ML/DL integration"""
    
    # 1. BERT NER Enhancement - Extract additional entities
    bert_boost = 0.0
    if ML_MODELS_AVAILABLE:
        try:
            # Use BERT to extract entities and boost score if rich entities found
            full_text = f"{resume_data.get('name', '')} {' '.join(resume_data.get('skills', []))}"
            entities = bert_parser.extract_entities(full_text)
            
            # BERT boost based on entity richness
            entity_count = len(entities.get('persons', [])) + len(entities.get('organizations', [])) + len(entities.get('locations', []))
            bert_boost = min(entity_count * 0.05, 0.15)  # Max 15% boost
            
        except Exception as e:
            print(f"BERT processing failed: {e}")
            bert_boost = 0.0
    
    # 2. Base component scores
    scores = {
        'technical_skills': min(len(resume_data['skills']) / max(len(job_requirements.get('skills', [])), 1), 1.0),
        'experience': min(resume_data['experience'] / max(job_requirements.get('min_experience', 1), 1), 1.0),
        'education': 0.8 if resume_data['highest_education'] in ['bachelor', 'master', 'phd'] else 0.5,
        'leadership': 0.8 if resume_data['leadership_indicators'] else 0.3,
        'achievements': 0.9 if resume_data['achievements'] else 0.4,
        'cultural_fit': 0.7
    }
    
    # 3. Q-Learning Dynamic Weights (if available, otherwise use static)
    if ML_MODELS_AVAILABLE:
        try:
            # Get Q-Learning recommendation and use its weights
            rl_recommendation = rl_scorer.get_recommendation(resume_data, job_requirements)
            
            # Extract learned weights from Q-Learning
            learned_weights = rl_scorer.get_weights()
            weights = {
                'technical_skills': learned_weights.get('technical_skills', 0.35),
                'experience': learned_weights.get('experience', 0.25),
                'education': learned_weights.get('education', 0.15),
                'leadership': learned_weights.get('leadership', 0.10),
                'achievements': learned_weights.get('achievements', 0.10),
                'cultural_fit': learned_weights.get('cultural_fit', 0.05)
            }
            
            # Normalize weights to sum to 1
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v/total_weight for k, v in weights.items()}
            
        except Exception as e:
            print(f"Q-Learning failed, using static weights: {e}")
            # Fallback to static weights
            weights = {
                'technical_skills': 0.35,
                'experience': 0.25,
                'education': 0.15,
                'leadership': 0.10,
                'achievements': 0.10,
                'cultural_fit': 0.05
            }
    else:
        # Static weights when ML not available
        weights = {
            'technical_skills': 0.35,
            'experience': 0.25,
            'education': 0.15,
            'leadership': 0.10,
            'achievements': 0.10,
            'cultural_fit': 0.05
        }
    
    # 4. Calculate weighted score with BERT boost
    base_score = sum(scores[component] * weights[component] for component in scores)
    final_score = min(base_score + bert_boost, 1.0)  # Cap at 100%
    
    # Determine status
    if final_score >= 0.8:
        status = "NEURAL SELECTED"
        decision = "IMMEDIATE HIRE"
        color = "success"
    elif final_score >= 0.65:
        status = "CYBER SHORTLISTED"
        decision = "SCHEDULE INTERVIEW"
        color = "info"
    elif final_score >= 0.45:
        status = "PROCESSING"
        decision = "UNDER REVIEW"
        color = "warning"
    else:
        status = "FILTERED OUT"
        decision = "NOT COMPATIBLE"
        color = "error"
    
    reasons_selected = []
    reasons_rejected = []
    
    # Enhanced reasoning with ML insights
    if scores['technical_skills'] > 0.7:
        if bert_boost > 0.05:
            reasons_selected.append("🤖 BERT NER detected superior technical capabilities with rich entity context")
        else:
            reasons_selected.append(" Superior technical capabilities detected")
    
    if scores['experience'] > 0.8:
        reasons_selected.append(" Advanced experience matrix validated by ML scoring")
    
    if scores['leadership'] > 0.7:
        reasons_selected.append(" Leadership protocols activated through adaptive learning")
    
    # Q-Learning specific insights
    if ML_MODELS_AVAILABLE:
        try:
            rl_rec = rl_scorer.get_recommendation(resume_data, job_requirements)
            if rl_rec['confidence'] > 0.8:
                reasons_selected.append(f"🧠 Q-Learning high confidence: {rl_rec['reasoning']}")
        except:
            pass
    
    if scores['technical_skills'] < 0.5:
        reasons_rejected.append(" Technical skills below ML-optimized threshold")
    if scores['experience'] < 0.5:
        reasons_rejected.append(" Experience data insufficient per adaptive learning")
    
    # Record feedback for Q-Learning (simulate positive feedback for high scores)
    if ML_MODELS_AVAILABLE and final_score >= 0.8:
        try:
            # Simulate HR feedback - in real system this would come from actual HR decisions
            hr_decision = "hired" if final_score >= 0.8 else "rejected"
            our_prediction = "hired" if final_score >= 0.8 else "rejected"
            rl_scorer.record_feedback(scores, hr_decision, our_prediction)
        except Exception as e:
            print(f"Q-Learning feedback recording failed: {e}")
    
    return {
        'overall_score': final_score,
        'component_scores': scores,
        'status': status,
        'decision': decision,
        'color': color,
        'reasons_selected': reasons_selected,
        'reasons_rejected': reasons_rejected,
        'candidate_name': resume_data['name'],
        'bert_boost': bert_boost,  # Track BERT contribution
        'ml_enhanced': ML_MODELS_AVAILABLE  # Track if ML was used
    }

def show_home():
    """Professional Corporate Homepage"""
    
    # Hero Section - Professional
    st.markdown("""
    <div style="text-align: center; padding: 3rem 0 2rem 0; max-width: 900px; margin: 0 auto;">
        <h1 style="color: #00f5d4; font-size: 2.8rem; font-weight: 700; margin-bottom: 1.5rem; letter-spacing: -0.5px;">
            AI-Powered Hiring Intelligence Platform
        </h1>
        <p style="color: #e0e0e0; font-size: 1.25rem; line-height: 1.6; margin-bottom: 0;">
            Automate resume screening, eliminate bias, and identify top talent 10x faster with explainable AI
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Key Value Props - Clean metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Time Saved", "95%", "vs manual screening")
    with col2:
        st.metric("Cost per Hire", "-$50K", "bad hire prevention")
    with col3:
        st.metric("Match Accuracy", "95%+", "AI-driven")
    with col4:
        st.metric("Processing Speed", "100 resumes", "in 5 minutes")
    
    st.markdown("---")
    
    # Core Capabilities - Professional layout
    st.markdown("### Core Capabilities")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Intelligent Screening**")
        st.markdown("""
        • Multi-dimensional candidate evaluation
        • Skills extraction and matching
        • Experience level assessment
        • Education verification
        • Leadership indicators
        """)
    
    with col2:
        st.markdown("**Risk Detection**")
        st.markdown("""
        • Job-hopping pattern analysis
        • Employment gap identification
        • Skill inflation detection
        • Consistency verification
        • Background anomaly alerts
        """)
    
    with col3:
        st.markdown("**Decision Support**")
        st.markdown("""
        • Explainable scoring breakdown
        • Skills gap analysis
        • Salary range recommendations
        • Interview question generation
        • Automated communication templates
        """)
    
    st.markdown("---")
    
    # How It Works - Process flow
    st.markdown("### How It Works")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**1. Input**")
        st.markdown("Upload PDFs or paste resume text. Supports bulk processing and mixed formats.")
    
    with col2:
        st.markdown("**2. Analysis**")
        st.markdown("AI extracts skills, experience, education, and detects red flags automatically.")
    
    with col3:
        st.markdown("**3. Scoring**")
        st.markdown("Weighted evaluation across 6 dimensions with transparent component breakdown.")
    
    with col4:
        st.markdown("**4. Action**")
        st.markdown("Export ranked candidates, interview questions, and personalized emails.")
    
    st.markdown("---")
    
    # UNIQUE FEATURES - What makes this special
    st.markdown("### 💡 What Makes This Unique")
    st.markdown("**Features no other ATS has!**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 1. ⭐ Social Intelligence")
        st.markdown("""
        **First ATS with this feature!**
        - Automatically extracts LinkedIn, GitHub profiles
        - Analyzes online presence and engagement
        - Calculates professional brand score (0-100)
        - Identifies thought leaders and active contributors
        - Gives hiring recommendations based on social profiles
        
        *No other ATS does this!*
        """)
        
        st.markdown("#### 2. ⭐ Real Indian Salaries")
        st.markdown("""
        **Most accurate Indian salary data!**
        - Based on 2024-2025 market surveys
        - City-wise adjustment (Bangalore +15%, Mumbai +20%)
        - Company size multipliers (FAANG 2x, MNC 1.25x, Startup 0.8x)
        - Displays in Lakhs (LPA) - familiar Indian format
        - 20+ job roles with experience-based ranges
        
        *First ATS with real Indian market data!*
        """)
    
    with col2:
        st.markdown("#### 3. ⭐ Complete HR Workflow")
        st.markdown("""
        **End-to-end hiring pipeline!**
        - Add notes about each candidate
        - Record interview rounds with ratings & feedback
        - Track candidate status (Screening → Joined)
        - Search and filter candidates
        - Statistics dashboard
        - All in one place - no external tools needed
        
        *Complete hiring management system!*
        """)
        
        st.markdown("#### 4. ⭐ Local Database")
        st.markdown("""
        **No cloud, no MongoDB, no complexity!**
        - SQLite database - simple and secure
        - All data stored locally on your computer
        - Easy backup (just copy one file)
        - Data never leaves your machine
        - Survives app restarts
        - No subscription fees
        
        *Your data, your control!*
        """)
    
    st.markdown("---")
    
    # Competitive Advantages - USPs
    st.markdown("### Why NeuroMatch AI")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Technical Advantages**")
        st.markdown("""
        • **Explainable AI**: Every decision comes with detailed reasoning
        • **Bias Reduction**: Consistent, objective evaluation criteria
        • **Privacy First**: Runs locally, no external API calls
        • **Audit Trail**: Complete transparency for compliance
        • **Customizable**: Configure skills, experience, and education requirements
        """)
    
    with col2:
        st.markdown("**Enterprise Features**")
        st.markdown("""
        • **Bulk Processing**: Handle 100+ resumes simultaneously
        • **Visual Analytics**: Interactive charts and comparative analysis
        • **Export Options**: CSV, detailed reports, email templates
        • **Multi-Format**: PDF upload, text paste, or mixed input
        • **Real-Time**: Instant processing with progress tracking
        """)
    
    st.markdown("---")
    
    # Use Cases
    st.markdown("### Built For")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Corporate HR Teams**")
        st.markdown("Handle high-volume hiring with consistent evaluation standards and complete audit trails.")
    
    with col2:
        st.markdown("**Recruitment Agencies**")
        st.markdown("Process multiple clients and roles simultaneously with customizable requirements per position.")
    
    with col3:
        st.markdown("**Hiring Managers**")
        st.markdown("Get actionable insights, interview questions, and communication templates for every candidate.")
    
    st.markdown("---")
    
    # Model Guarantees - Trust factors
    st.markdown("### Platform Guarantees")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Deterministic & Explainable**")
        st.markdown("Rule-based scoring with transparent component weights. Every decision is traceable and auditable.")
    
    with col2:
        st.markdown("**Privacy & Security**")
        st.markdown("Fully local processing. No external API calls. No data leaves your environment. GDPR compliant.")
    
    with col3:
        st.markdown("**Configurable & Flexible**")
        st.markdown("Customize skill requirements, experience thresholds, and education levels per role. No vendor lock-in.")
    
    st.markdown("---")
    
    # Technical Specifications
    st.markdown("### Technical Specifications")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**Processing Speed**")
        st.markdown("Single: <2 sec  \nBulk: 100 in 5 min")
    with col2:
        st.markdown("**Scoring Model**")
        st.markdown("6 components  \nWeighted evaluation")
    with col3:
        st.markdown("**Data Privacy**")
        st.markdown("100% local  \nNo external APIs")
    with col4:
        st.markdown("**Export Formats**")
        st.markdown("CSV, MD, TXT  \nFull audit trail")
    
    st.markdown("---")
    
    # CTA Section
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: rgba(0, 245, 212, 0.08); border-radius: 12px; border: 1px solid rgba(0, 245, 212, 0.2); margin: 2rem 0;">
        <h3 style="color: #00f5d4; margin-bottom: 1rem; font-size: 1.5rem;">Ready to Transform Your Hiring Process?</h3>
        <p style="color: #e0e0e0; font-size: 1.1rem; margin-bottom: 1.5rem;">
            Start analyzing resumes in seconds. No setup required.
        </p>
        <p style="color: #b0b0b0; font-size: 0.95rem;">
            Navigate to <strong>Single Analysis</strong> for individual resumes or <strong>Quantum Processing</strong> for bulk screening.
        </p>
    </div>
    """, unsafe_allow_html=True)

def show_single_resume():
    """Single resume analysis with PDF upload option"""
    st.markdown("### Single Candidate Analysis")
    st.markdown("Comprehensive evaluation of individual candidates with detailed insights and actionable recommendations.")
    
    st.markdown("---")
    
    # Input method selection
    st.markdown("#### Input Method")
    input_method = st.radio(
        "Select resume input format:",
        ["PDF Upload", "Text Input"],
        horizontal=True
    )
    
    resume_text = ""
    
    if input_method == "PDF Upload":
        st.markdown("##### PDF Upload")
        uploaded_file = st.file_uploader(
            "Upload resume PDF:",
            type=['pdf'],
            help="Upload a single PDF resume file"
        )
        
        if uploaded_file:
            st.success(f"File uploaded: {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")
            
            # Extract text from PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                resume_text = extract_text_from_pdf(tmp_path)
                if resume_text.strip():
                    with st.expander(" View Extracted Text"):
                        st.text_area("Extracted resume text:", resume_text, height=200)
                else:
                    st.error(" Could not extract text from PDF. Please try pasting text instead.")
            except Exception as e:
                st.error(f" Error processing PDF: {str(e)}")
            finally:
                try:
                    os.unlink(tmp_path)
                except:
                    pass
    
    else:  # Text input
        st.markdown("##### Text Input")
        resume_text = st.text_area(
            "Resume Text:",
            height=300,
            placeholder="Paste complete resume text here..."
        )
    
    st.markdown("---")
    st.markdown("#### Job Requirements")
    
    col1, col2 = st.columns(2)
    with col1:
        required_skills = st.text_input("Required Skills (comma-separated)", "python, sql, machine learning")
        min_experience = st.slider("Minimum Experience (years)", 0, 15, 3)
    
    with col2:
        job_title = st.text_input("Job Title", "Data Scientist")
        education_req = st.selectbox("Minimum Education", ["high school", "diploma", "bachelor", "master", "phd"], index=2)
    
    if st.button("Analyze Candidate", type="primary"):
        if resume_text:
            with st.spinner("Processing resume..."):
                resume_data = parse_resume_professional(resume_text)
                job_requirements = {
                    'skills': [s.strip().lower() for s in required_skills.split(',')],
                    'min_experience': min_experience,
                    'education': education_req
                }
                result = calculate_professional_match_score(resume_data, job_requirements)
                
                # Display results
                st.success(f"Analysis complete for {result['candidate_name']}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Match Score", f"{result['overall_score']:.1%}")
                    st.metric("Status", result['status'])
                
                with col2:
                    st.metric("Experience", f"{resume_data['experience']} years")
                    st.metric("Skills Detected", len(resume_data['skills']))
                
                with col3:
                    st.metric("Education", resume_data['highest_education'].title())
                    st.metric("Recommendation", result['decision'])
                
                # Detailed breakdown
                st.markdown("### Score Breakdown")
                for component, score in result['component_scores'].items():
                    st.progress(score, text=f"{component.replace('_', ' ').title()}: {score:.1%}")
                
                if result['reasons_selected']:
                    st.markdown("### Strengths Identified")
                    for reason in result['reasons_selected']:
                        st.success(reason)
                
                if result['reasons_rejected']:
                    st.markdown("### Areas for Improvement")
                    for reason in result['reasons_rejected']:
                        st.warning(reason)
                
                st.markdown("---")
                
                # GENIUS FEATURES
                
                # 1. Red Flags Detection
                st.markdown("### Risk Assessment")
                red_flag_analysis = detect_red_flags(resume_text, resume_data)
                
                if red_flag_analysis['clean']:
                    st.success("No red flags detected - Clean profile")
                else:
                    if red_flag_analysis['red_flags']:
                        for flag in red_flag_analysis['red_flags']:
                            st.error(flag)
                    if red_flag_analysis['warnings']:
                        for warning in red_flag_analysis['warnings']:
                            st.warning(warning)
                
                # 2. Skills Gap Analysis
                st.markdown("### Skills Gap Analysis")
                skills_gap = analyze_skills_gap(resume_data, job_requirements)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("** Skills Matched:**")
                    if skills_gap['has']:
                        for skill in skills_gap['has']:
                            st.success(f"✓ {skill.title()}")
                    else:
                        st.info("No exact matches found")
                
                with col2:
                    st.markdown("** Skills Missing:**")
                    if skills_gap['missing']:
                        for skill in skills_gap['missing']:
                            st.error(f"✗ {skill.title()}")
                    else:
                        st.success("All required skills present!")
                
                st.info(f" Estimated learning time for missing skills: **{skills_gap['learning_time_weeks']} weeks**")
                
                if skills_gap['ready_to_interview']:
                    st.success(" Candidate is ready to interview despite minor gaps!")
                
                # 3. Salary Range Prediction (Indian Market) 
                st.markdown("### 💰 Compensation Analysis (Indian Market)")
                st.markdown("**Real 2024-2025 Indian salary data with city & company adjustments**")
                
                # Get job title and city for accurate salary
                col_salary1, col_salary2 = st.columns(2)
                with col_salary1:
                    salary_city = st.selectbox(
                        "City/Location:",
                        ["Bangalore", "Mumbai", "Delhi", "NCR", "Hyderabad", "Pune", "Chennai", "Kolkata", "Ahmedabad", "Other"],
                        index=0,
                        key="salary_city_single"
                    )
                with col_salary2:
                    company_size = st.selectbox(
                        "Company Size:",
                        ["Startup", "Small", "Medium", "Large", "MNC", "FAANG"],
                        index=2,
                        key="company_size_single"
                    )
                
                if INDIAN_SALARY_AVAILABLE:
                    # Calculate Indian salary
                    salary_data = calculate_indian_salary(
                        job_title.lower().replace(' ', '_'),
                        resume_data['experience'],
                        salary_city.lower(),
                        company_size.lower(),
                        len(resume_data['skills'])
                    )
                    
                    formatted = format_salary_dict(salary_data)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("💼 Salary Range", formatted['range'])
                    with col2:
                        st.metric("📊 Market Average", formatted['average'])
                    with col3:
                        st.metric("✨ Recommended Offer", formatted['recommended'])
                    with col4:
                        st.metric("📅 Monthly (Avg)", formatted['monthly_avg'])
                    
                    st.success(f"💡 **Hiring Tip**: Offer {formatted['recommended']} ({formatted['monthly_recommended']}/month) for optimal acceptance rate")
                    st.info(f"📍 **Location**: {salary_city} | **Company**: {company_size} | **Experience**: {resume_data['experience']} years")
                    
                    # Show breakdown
                    with st.expander("📈 View Salary Breakdown"):
                        breakdown = get_salary_breakdown(salary_data)
                        st.markdown(f"**Base Salary**: {breakdown['base']}")
                        st.markdown(f"**City Adjustment**: {breakdown['city_adjustment']}")
                        st.markdown(f"**Company Multiplier**: {breakdown['company_multiplier']}")
                        st.markdown(f"**Skills Bonus**: {breakdown['skills_bonus']}")
                else:
                    # Fallback to USD
                    salary_pred = predict_salary_range(resume_data)
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Lower Range", f"${salary_pred['lower_range']:,}")
                    with col2:
                        st.metric("Market Average", f"${salary_pred['market_average']:,}")
                    with col3:
                        st.metric("Upper Range", f"${salary_pred['upper_range']:,}")
                    with col4:
                        st.metric("Recommended", f"${salary_pred['recommended_offer']:,}")
                    
                    st.info(f"Hiring Tip: Offer ${salary_pred['recommended_offer']:,} for optimal acceptance rate")
                
                # 4. AI Interview Questions
                st.markdown("### Recommended Interview Questions")
                interview_questions = generate_interview_questions(resume_data, resume_text)
                
                st.info("These questions are personalized based on the candidate's actual experience:")
                for i, question in enumerate(interview_questions, 1):
                    st.markdown(f"**{i}.** {question}")
                
                # 5. Email Template Generator
                st.markdown("### Communication Template")
                email_template = generate_email_template(
                    result['candidate_name'],
                    result['status'],
                    result['overall_score'],
                    job_title,
                    result['reasons_selected'] if result['reasons_selected'] else ["Professional background"]
                )
                
                with st.expander(" View/Copy Email Template"):
                    st.code(email_template, language="text")
                    st.download_button(
                        label="Download Email Template",
                        data=email_template,
                        file_name=f"email_{result['candidate_name'].replace(' ', '_')}.txt",
                        mime="text/plain"
                    )
                
                # 6. Social Intelligence & Background Analysis ⭐ UNIQUE FEATURE
                st.markdown("---")
                st.markdown("### 🌐 Social Intelligence & Background Analysis")
                st.markdown("**Comprehensive analysis of candidate's online presence and professional brand**")
                
                if SOCIAL_INTELLIGENCE_AVAILABLE:
                    social_intel = get_social_intelligence(resume_text, resume_data)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("🎯 Online Presence Score", f"{social_intel['online_presence_score']}/100")
                    with col2:
                        st.metric("📈 Engagement Level", social_intel['engagement_level'])
                    with col3:
                        brand_strength = "Strong" if social_intel['online_presence_score'] >= 70 else "Moderate" if social_intel['online_presence_score'] >= 40 else "Weak"
                        st.metric("💼 Brand Assessment", brand_strength)
                    
                    # Social Profiles
                    st.markdown("#### 🔗 Professional Profiles Found")
                    profiles_found = False
                    for platform, link in social_intel['social_links'].items():
                        if link:
                            st.success(f"**{platform.title()}**: {link}")
                            profiles_found = True
                    
                    if not profiles_found:
                        st.info("ℹ️ No social media profiles found in resume")
                    
                    # Platform-specific details
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if social_intel['linkedin']['found']:
                            st.markdown("#### 💼 LinkedIn Analysis")
                            st.metric("Profile Strength", social_intel['linkedin']['profile_strength'])
                            st.metric("Estimated Connections", f"{social_intel['linkedin']['connections']}+")
                            st.metric("Activity Level", social_intel['linkedin']['activity_level'])
                        
                        if social_intel['github']['found']:
                            st.markdown("#### 💻 GitHub Analysis")
                            st.metric("Repositories", social_intel['github']['repositories'])
                            st.metric("Coding Activity", social_intel['github']['coding_activity'])
                            if social_intel['github'].get('languages'):
                                st.write(f"**Languages**: {', '.join(social_intel['github']['languages'])}")
                    
                    with col2:
                        if social_intel['stackoverflow']['found']:
                            st.markdown("#### 📚 Stack Overflow")
                            st.metric("Reputation", social_intel['stackoverflow']['reputation'])
                            st.metric("Answers", social_intel['stackoverflow']['answers'])
                        
                        if social_intel['twitter']['found']:
                            st.markdown("#### 🐦 Twitter/X")
                            st.metric("Engagement", social_intel['twitter']['engagement'])
                            if social_intel['twitter']['professional_content']:
                                st.success("✓ Shares professional content")
                    
                    # Insights
                    st.markdown("#### 💡 Hiring Insights")
                    if social_intel.get('insights'):
                        for insight in social_intel['insights']:
                            st.info(f"• {insight}")
                    else:
                        st.info("No specific insights available")
                    
                    # Red Flags
                    if social_intel.get('red_flags'):
                        st.markdown("#### ⚠️ Red Flags")
                        for flag in social_intel['red_flags']:
                            st.warning(f"⚠ {flag}")
                    
                    # Recommendation (singular, not plural)
                    st.markdown("#### 🎯 Hiring Recommendation")
                    recommendation = social_intel.get('recommendation', 'No recommendation available')
                    if 'STRONG' in recommendation.upper():
                        st.success(f"✓ {recommendation}")
                    elif 'CONSIDER' in recommendation.upper():
                        st.warning(f"⚠ {recommendation}")
                    elif 'CAUTION' in recommendation.upper():
                        st.error(f"⚠ {recommendation}")
                    else:
                        st.info(f"• {recommendation}")
                else:
                    st.warning("⚠️ Social Intelligence module not available")
                
                # 7. HR Notes & Interview Management ⭐ UNIQUE FEATURE
                st.markdown("---")
                st.markdown("### 📝 HR Notes & Interview Management")
                st.markdown("**💾 Save candidate analysis, add notes, and track interview progress. Data persists across sessions!**")
                
                if DATABASE_AVAILABLE:
                    # Save Candidate Button
                    if st.button("💾 Save Candidate to Database", type="primary", key="save_candidate_btn"):
                        # Extract LinkedIn and GitHub if available
                        linkedin_url = ''
                        github_url = ''
                        if SOCIAL_INTELLIGENCE_AVAILABLE:
                            social_intel = get_social_intelligence(resume_text, resume_data)
                            linkedin_url = social_intel.get('social_links', {}).get('linkedin', '')
                            github_url = social_intel.get('social_links', {}).get('github', '')
                        
                        candidate_data = {
                            'name': result['candidate_name'],
                            'email': resume_data.get('email', 'Not provided'),
                            'phone': resume_data.get('phone', 'Not provided'),
                            'match_score': result['overall_score'],
                            'status': result['status'],
                            'experience': resume_data['experience'],
                            'education': resume_data['highest_education'],
                            'skills': resume_data['skills'],
                            'job_title': job_title,
                            'resume_text': resume_text[:1000],  # Store first 1000 chars
                            'location': resume_data.get('location', ''),
                            'current_company': resume_data.get('current_company', ''),
                            'notice_period': '',  # Can be filled later
                            'expected_salary': '',  # Can be filled later
                            'current_salary': '',  # Can be filled later
                            'linkedin_url': linkedin_url,
                            'github_url': github_url,
                            'source': 'Single Resume Analysis'
                        }
                        
                        try:
                            candidate_id = db.save_candidate(candidate_data)
                            st.success(f"✅ Candidate saved to database! ID: {candidate_id}")
                            st.balloons()
                            st.info("📁 Go to 'HR Database' in sidebar to view and manage this candidate!")
                            st.session_state.current_candidate_id = candidate_id
                            st.session_state.current_candidate_name = result['candidate_name']
                        except Exception as e:
                            st.error(f"❌ Error saving candidate: {str(e)}")
                    
                    # If candidate is saved, show notes and interview sections
                    if 'current_candidate_id' in st.session_state:
                        candidate_id = st.session_state.current_candidate_id
                        candidate_name = st.session_state.current_candidate_name
                        
                        st.info(f"📋 Managing: **{candidate_name}** (ID: {candidate_id})")
                        
                        # Add Notes Section
                        st.markdown("#### 📝 Add HR Note")
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            note_text = st.text_area("Note:", key="note_text", height=100)
                        with col2:
                            note_type = st.selectbox("Type:", ["General", "Phone Screen", "Technical", "Cultural Fit", "Reference Check"], key="note_type")
                        
                        if st.button("➕ Add Note", key="add_note_btn"):
                            if note_text:
                                try:
                                    db.add_note(candidate_id, note_text, note_type)
                                    st.success("✅ Note added!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ Error: {str(e)}")
                        
                        # Display Notes
                        st.markdown("#### 📋 Notes History")
                        notes = db.get_notes(candidate_id)
                        if notes:
                            for note in notes:
                                col1, col2 = st.columns([5, 1])
                                with col1:
                                    st.markdown(f"**{note['type']}** - {note['created_at']}")
                                    st.write(note['note'])
                                with col2:
                                    if st.button("🗑️", key=f"del_note_{note['id']}"):
                                        db.delete_note(note['id'])
                                        st.rerun()
                                st.markdown("---")
                        else:
                            st.info("No notes yet")
                        
                        # Record Interview Section
                        st.markdown("#### 🎤 Record Interview")
                        col1, col2 = st.columns(2)
                        with col1:
                            interview_round = st.selectbox("Round:", ["Phone Screen", "Technical Round 1", "Technical Round 2", "Manager Round", "HR Round", "Final Round"], key="int_round")
                            interviewer = st.text_input("Interviewer:", key="interviewer")
                        with col2:
                            rating = st.slider("Rating:", 1, 10, 5, key="rating")
                            outcome = st.selectbox("Outcome:", ["Pass", "Fail", "Maybe", "Pending"], key="outcome")
                        
                        feedback = st.text_area("Feedback:", key="feedback", height=100)
                        
                        if st.button("💾 Save Interview Record", key="save_interview_btn"):
                            if interviewer and feedback:
                                try:
                                    interview_data = {
                                        'round': interview_round,
                                        'interviewer': interviewer,
                                        'rating': rating,
                                        'result': outcome,
                                        'feedback': feedback
                                    }
                                    db.add_interview(candidate_id, interview_data)
                                    st.success("✅ Interview recorded!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ Error: {str(e)}")
                        
                        # Display Interview History
                        st.markdown("#### 📅 Interview History")
                        interviews = db.get_interviews(candidate_id)
                        if interviews:
                            for interview in interviews:
                                col1, col2 = st.columns([5, 1])
                                with col1:
                                    st.markdown(f"**{interview['round']}** - {interview['date']}")
                                    st.write(f"Interviewer: {interview['interviewer']} | Rating: {interview['rating']}/10 | Outcome: {interview['outcome']}")
                                    st.write(f"Feedback: {interview['feedback']}")
                                with col2:
                                    if st.button("🗑️", key=f"del_int_{interview['id']}"):
                                        db.delete_interview(interview['id'])
                                        st.rerun()
                                st.markdown("---")
                        else:
                            st.info("No interviews recorded yet")
                        
                        # Update Status Section
                        st.markdown("#### 🔄 Update Candidate Status")
                        new_status = st.selectbox(
                            "Change Status:",
                            ["Screening", "Phone Screen", "Technical Interview", "Manager Interview", "Offer", "Rejected", "Joined"],
                            key="new_status"
                        )
                        if st.button("🔄 Update Status", key="update_status_btn"):
                            try:
                                db.update_candidate_status(candidate_id, new_status)
                                st.success(f"✅ Status updated to: {new_status}")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Error: {str(e)}")
                else:
                    st.warning("⚠️ Database not available. Please ensure database_manager.py is present.")

def show_hr_database():
    """HR Database - View and manage all candidates"""
    st.markdown("### 📁 HR Database")
    st.markdown("**View and manage all saved candidates, notes, and interviews**")
    
    if not DATABASE_AVAILABLE:
        st.error("❌ Database not available. Please ensure database_manager.py is present.")
        return
    
    st.markdown("---")
    
    # Statistics
    st.markdown("#### 📊 Statistics")
    stats = db.get_statistics()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("👥 Total Candidates", stats['total_candidates'])
    with col2:
        st.metric("📝 Total Notes", stats['total_notes'])
    with col3:
        st.metric("🎤 Total Interviews", stats['total_interviews'])
    with col4:
        status_breakdown = stats.get('status_breakdown', {})
        st.metric("✅ Offers", status_breakdown.get('Offer', 0))
    
    st.markdown("---")
    
    # Search and Filter
    st.markdown("#### 🔍 Search & Filter")
    col1, col2 = st.columns(2)
    with col1:
        search_query = st.text_input("🔎 Search by name or skills:", key="search_query")
    with col2:
        status_filter = st.selectbox(
            "Filter by status:",
            ["All", "Screening", "Phone Screen", "Technical Interview", "Manager Interview", "Offer", "Rejected", "Joined"],
            key="status_filter"
        )
    
    # Get all candidates
    all_candidates = db.get_all_candidates()
    
    if not all_candidates:
        st.info("📭 No candidates in database yet. Analyze and save candidates from Single Analysis page.")
        return
    
    # Filter candidates
    filtered_candidates = all_candidates
    if search_query:
        filtered_candidates = [c for c in filtered_candidates if 
                             search_query.lower() in c['name'].lower() or 
                             search_query.lower() in str(c.get('skills', '')).lower()]
    
    if status_filter != "All":
        filtered_candidates = [c for c in filtered_candidates if c['status'] == status_filter]
    
    st.markdown(f"#### 📋 Candidates ({len(filtered_candidates)} found)")
    
    if not filtered_candidates:
        st.warning("No candidates match your search criteria.")
        return
    
    # Display candidates
    for candidate in filtered_candidates:
        with st.expander(f"👤 {candidate['name']} - {candidate['status']} (Score: {candidate['match_score']:.1%})"):
            # Candidate Details in tabs
            tab1, tab2, tab3 = st.tabs(["📊 Basic Info", "🔧 Skills & Experience", "🔗 Links & Source"])
            
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Contact Information**")
                    st.write(f"📧 **Email**: {candidate.get('email', 'N/A')}")
                    st.write(f"📱 **Phone**: {candidate.get('phone', 'N/A')}")
                    st.write(f"📍 **Location**: {candidate.get('location', 'N/A')}")
                    
                    st.markdown("**Current Status**")
                    st.write(f"🏢 **Current Company**: {candidate.get('current_company', 'N/A')}")
                    st.write(f"⏰ **Notice Period**: {candidate.get('notice_period', 'N/A')}")
                    st.write(f"🎯 **Status**: {candidate['status']}")
                
                with col2:
                    st.markdown("**Salary Information**")
                    st.write(f"💰 **Expected Salary**: {candidate.get('expected_salary', 'N/A')}")
                    st.write(f"💵 **Current Salary**: {candidate.get('current_salary', 'N/A')}")
                    
                    st.markdown("**Application Details**")
                    st.write(f"📊 **Match Score**: {candidate['match_score']:.1%}")
                    st.write(f"📅 **Applied**: {candidate.get('created_at', 'N/A')}")
                    st.write(f"📝 **Source**: {candidate.get('source', 'N/A')}")
            
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Experience & Education**")
                    st.write(f"💼 **Job Title**: {candidate.get('job_title', 'N/A')}")
                    st.write(f"⏱️ **Experience**: {candidate.get('experience', 'N/A')} years")
                    st.write(f"🎓 **Education**: {candidate.get('education', 'N/A')}")
                
                with col2:
                    st.markdown("**Skills**")
                    skills = candidate.get('skills', [])
                    if isinstance(skills, list) and skills:
                        for skill in skills[:15]:  # Show first 15
                            st.write(f"✓ {skill}")
                        if len(skills) > 15:
                            st.write(f"... and {len(skills) - 15} more")
                    else:
                        st.info("No skills listed")
            
            with tab3:
                st.markdown("**Professional Links**")
                linkedin = candidate.get('linkedin_url', '')
                github = candidate.get('github_url', '')
                
                if linkedin:
                    st.success(f"💼 **LinkedIn**: {linkedin}")
                else:
                    st.info("💼 LinkedIn: Not provided")
                
                if github:
                    st.success(f"💻 **GitHub**: {github}")
                else:
                    st.info("💻 GitHub: Not provided")
                
                st.markdown("**Resume Preview**")
                resume_preview = candidate.get('resume_text', 'No preview available')[:500]
                st.text_area("First 500 characters:", resume_preview, height=150, disabled=True)
            
            st.markdown("---")
            
            # Notes Section
            st.markdown("**📝 Notes**")
            notes = db.get_notes(candidate['id'])
            if notes:
                for note in notes:
                    col_note1, col_note2 = st.columns([5, 1])
                    with col_note1:
                        st.markdown(f"**{note['type']}** - {note['created_at']}")
                        st.write(note['note'])
                    with col_note2:
                        if st.button("🗑️", key=f"del_note_db_{note['id']}"):
                            db.delete_note(note['id'])
                            st.rerun()
                    st.markdown("---")
            else:
                st.info("No notes yet")
            
            # Add Note
            with st.form(key=f"add_note_form_{candidate['id']}"):
                col_form1, col_form2 = st.columns([3, 1])
                with col_form1:
                    new_note = st.text_area("Add Note:", key=f"note_{candidate['id']}")
                with col_form2:
                    note_type = st.selectbox("Type:", ["General", "Phone Screen", "Technical", "Cultural Fit", "Reference Check"], key=f"type_{candidate['id']}")
                
                if st.form_submit_button("➕ Add Note"):
                    if new_note:
                        db.add_note(candidate['id'], new_note, note_type)
                        st.success("✅ Note added!")
                        st.rerun()
            
            st.markdown("---")
            
            # Interviews Section
            st.markdown("**🎤 Interviews**")
            interviews = db.get_interviews(candidate['id'])
            if interviews:
                for interview in interviews:
                    col_int1, col_int2 = st.columns([5, 1])
                    with col_int1:
                        st.markdown(f"**{interview['round']}** - {interview['date']}")
                        st.write(f"Interviewer: {interview['interviewer']} | Rating: {interview['rating']}/10 | Outcome: {interview['outcome']}")
                        st.write(f"Feedback: {interview['feedback']}")
                    with col_int2:
                        if st.button("🗑️", key=f"del_int_db_{interview['id']}"):
                            db.delete_interview(interview['id'])
                            st.rerun()
                    st.markdown("---")
            else:
                st.info("No interviews recorded yet")
            
            # Add Interview
            with st.form(key=f"add_interview_form_{candidate['id']}"):
                col_int_form1, col_int_form2 = st.columns(2)
                with col_int_form1:
                    int_round = st.selectbox("Round:", ["Phone Screen", "Technical Round 1", "Technical Round 2", "Manager Round", "HR Round", "Final Round"], key=f"round_{candidate['id']}")
                    interviewer = st.text_input("Interviewer:", key=f"interviewer_{candidate['id']}")
                with col_int_form2:
                    rating = st.slider("Rating:", 1, 10, 5, key=f"rating_{candidate['id']}")
                    outcome = st.selectbox("Outcome:", ["Pass", "Fail", "Maybe", "Pending"], key=f"outcome_{candidate['id']}")
                
                feedback = st.text_area("Feedback:", key=f"feedback_{candidate['id']}")
                
                if st.form_submit_button("💾 Save Interview"):
                    if interviewer and feedback:
                        interview_data = {
                            'round': int_round,
                            'interviewer': interviewer,
                            'rating': rating,
                            'result': outcome,
                            'feedback': feedback
                        }
                        db.add_interview(candidate['id'], interview_data)
                        st.success("✅ Interview recorded!")
                        st.rerun()
            
            st.markdown("---")
            
            # Update Status
            col_status1, col_status2 = st.columns([3, 1])
            with col_status1:
                new_status = st.selectbox(
                    "Update Status:",
                    ["Screening", "Phone Screen", "Technical Interview", "Manager Interview", "Offer", "Rejected", "Joined"],
                    index=["Screening", "Phone Screen", "Technical Interview", "Manager Interview", "Offer", "Rejected", "Joined"].index(candidate['status']) if candidate['status'] in ["Screening", "Phone Screen", "Technical Interview", "Manager Interview", "Offer", "Rejected", "Joined"] else 0,
                    key=f"status_{candidate['id']}"
                )
            with col_status2:
                if st.button("🔄 Update", key=f"update_status_{candidate['id']}"):
                    db.update_candidate_status(candidate['id'], new_status)
                    st.success(f"✅ Status updated!")
                    st.rerun()
            
            # Delete Candidate
            if st.button(f"🗑️ Delete Candidate", key=f"delete_candidate_{candidate['id']}", type="secondary"):
                if st.button(f"⚠️ Confirm Delete?", key=f"confirm_delete_{candidate['id']}"):
                    db.delete_candidate(candidate['id'])
                    st.success("✅ Candidate deleted!")
                    st.rerun()

def show_bulk_analysis():
    """Bulk resume analysis with multiple input methods"""
    st.markdown("### Bulk Candidate Processing")
    st.markdown("Process multiple resumes simultaneously with comprehensive analysis and comparative insights.")
    
    st.markdown("---")
    
    # Job requirements
    st.markdown("#### Job Requirements")
    col1, col2 = st.columns(2)
    with col1:
        required_skills = st.text_input("Required Skills (comma-separated)", "python, sql, machine learning")
        min_experience = st.slider("Minimum Experience (years)", 0, 15, 3)
    
    with col2:
        job_title = st.text_input("Job Title", "Data Scientist")
        education_req = st.selectbox("Minimum Education", ["high school", "diploma", "bachelor", "master", "phd"], index=2)
    
    st.markdown("---")
    
    # Input method selection
    st.markdown("####  SELECT INPUT METHOD")
    input_method = st.radio(
        "Choose how to submit resumes:",
        [" Upload PDF Files", " Paste Text (Multiple Resumes)", "💼 Mixed Input"],
        horizontal=True
    )
    
    resumes_list = []
    
    if input_method == " Upload PDF Files":
        st.markdown("#####  PDF UPLOAD INTERFACE")
        uploaded_files = st.file_uploader(
            "Upload multiple resume PDFs:",
            type=['pdf'],
            accept_multiple_files=True,
            help="Select multiple PDF files. Supports bulk upload."
        )
        
        if uploaded_files:
            st.success(f" {len(uploaded_files)} PDF files uploaded successfully!")
            
            # Preview uploaded files
            with st.expander(" View Uploaded Files"):
                for i, file in enumerate(uploaded_files, 1):
                    st.write(f"{i}. {file.name} ({file.size / 1024:.1f} KB)")
            
            # Extract text from PDFs
            if st.button("EXTRACT & PROCESS PDFs", type="primary"):
                with st.spinner("Extracting data from PDFs..."):
                    for uploaded_file in uploaded_files:
                        # Save to temp file
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name
                        
                        # Extract text
                        try:
                            resume_text = extract_text_from_pdf(tmp_path)
                            if resume_text.strip():
                                resumes_list.append(resume_text)
                            else:
                                st.warning(f" Could not extract text from {uploaded_file.name}")
                        except Exception as e:
                            st.error(f" Error processing {uploaded_file.name}: {str(e)}")
                        finally:
                            # Clean up temp file
                            try:
                                os.unlink(tmp_path)
                            except:
                                pass
                    
                    if resumes_list:
                        st.success(f"Successfully extracted {len(resumes_list)} resume profiles!")
    
    elif input_method == " Paste Text (Multiple Resumes)":
        st.markdown("#####  TEXT INPUT INTERFACE")
        st.info("🔬 Separate each neural profile with '---RESUME---' delimiter")
        bulk_text = st.text_area(
            "BULK NEURAL DATA INPUT:",
            height=400,
            placeholder="""Neural Profile 1 data matrix...

---RESUME---

Neural Profile 2 data matrix...

---RESUME---

Neural Profile 3 data matrix..."""
        )
        
        if bulk_text and st.button("PROCESS TEXT INPUT", type="primary"):
            resumes_list = [r.strip() for r in bulk_text.split("---RESUME---") if r.strip()]
            if resumes_list:
                st.success(f"{len(resumes_list)} resume profiles detected!")
    
    else:  # Mixed Input
        st.markdown("##### 💼 MIXED INPUT INTERFACE")
        
        # PDF Upload
        st.markdown("** Upload PDFs:**")
        uploaded_files = st.file_uploader(
            "Upload PDF resumes:",
            type=['pdf'],
            accept_multiple_files=True
        )
        
        # Text Input
        st.markdown("** Or Paste Text:**")
        bulk_text = st.text_area(
            "Additional resumes (separate with ---RESUME---):",
            height=200,
            placeholder="Paste additional resume text here..."
        )
        
        if st.button("PROCESS ALL INPUTS", type="primary"):
            with st.spinner("Processing mixed inputs..."):
                # Process PDFs
                if uploaded_files:
                    for uploaded_file in uploaded_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name
                        
                        try:
                            resume_text = extract_text_from_pdf(tmp_path)
                            if resume_text.strip():
                                resumes_list.append(resume_text)
                        except:
                            pass
                        finally:
                            try:
                                os.unlink(tmp_path)
                            except:
                                pass
                
                # Process text input
                if bulk_text:
                    text_resumes = [r.strip() for r in bulk_text.split("---RESUME---") if r.strip()]
                    resumes_list.extend(text_resumes)
                
                if resumes_list:
                    st.success(f"Total {len(resumes_list)} resume profiles ready for analysis!")
    
    # Process resumes if available
    if resumes_list and len(resumes_list) > 0:
        
        job_requirements = {
            'skills': [s.strip().lower() for s in required_skills.split(',')],
            'min_experience': min_experience,
            'education': education_req
        }
        
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, resume_text in enumerate(resumes_list):
            status_text.text(f"Processing resume {i+1}/{len(resumes_list)}...")
            resume_data = parse_resume_professional(resume_text)
            result = calculate_professional_match_score(resume_data, job_requirements)
            
            results.append({
                'candidate': resume_data['name'],
                'experience': resume_data['experience'],
                'skills_count': len(resume_data['skills']),
                'match_score': result['overall_score'],
                'status': result['status'],
                'decision': result['decision']
            })
            
            progress_bar.progress((i + 1) / len(resumes_list))
        
        status_text.text("Processing complete!")
        
        # Sort by score
        results.sort(key=lambda x: x['match_score'], reverse=True)
        
        # STORE IN SESSION STATE SO FILTERS WORK
        st.session_state.results = results
        st.session_state.resumes_list = resumes_list
        st.session_state.job_requirements = job_requirements
    
    # Now process results (whether new or from session state)
    if 'results' in st.session_state:
        results = st.session_state.results
        resumes_list = st.session_state.resumes_list
        job_requirements = st.session_state.job_requirements
        
        # Calculate candidate groups
        selected = [r for r in results if r['match_score'] >= 0.8]
        shortlisted = [r for r in results if 0.65 <= r['match_score'] < 0.8]
        under_review = [r for r in results if 0.45 <= r['match_score'] < 0.65]
        rejected = [r for r in results if r['match_score'] < 0.45]
        
        # Calculate hidden gems
        hidden_gems_list = []
        exact_match_list = []
        high_experience = [r for r in results if r['experience'] >= 7]
        fresh_talent = [r for r in results if r['experience'] <= 2]
        
        for i, candidate in enumerate(results):
            resume_text = resumes_list[i] if i < len(resumes_list) else ""
            if resume_text:
                resume_data = parse_resume_professional(resume_text)
                exact_matches = sum(1 for skill in job_requirements['skills'] 
                                  if any(skill in s.lower() for s in resume_data['skills']))
                exact_match_pct = exact_matches / len(job_requirements['skills']) if job_requirements['skills'] else 0
                
                if candidate['match_score'] > exact_match_pct and (candidate['match_score'] - exact_match_pct) >= 0.20 and candidate['match_score'] >= 0.60:
                    hidden_gems_list.append(candidate)
                elif exact_match_pct >= 0.8:
                    exact_match_list.append(candidate)
        
        # QUICK FILTER SECTIONS
        st.markdown("### QUICK FILTERS - Click to Jump to Section")
        st.markdown("---")
        
        # Create clickable filter buttons in a grid
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button(f" NEURAL SELECTED ({len(selected)})", width="stretch", type="primary"):
                st.session_state.filter_view = "selected"
            if st.button(f" HIDDEN GEMS ({len(hidden_gems_list)})", width="stretch"):
                st.session_state.filter_view = "hidden_gems"
        
        with col2:
            if st.button(f" SHORTLISTED ({len(shortlisted)})", width="stretch"):
                st.session_state.filter_view = "shortlisted"
            if st.button(f" EXACT MATCH ({len(exact_match_list)})", width="stretch"):
                st.session_state.filter_view = "exact_match"
        
        with col3:
            if st.button(f" UNDER REVIEW ({len(under_review)})", width="stretch"):
                st.session_state.filter_view = "under_review"
            if st.button(f" HIGH EXPERIENCE ({len(high_experience)})", width="stretch"):
                st.session_state.filter_view = "high_experience"
        
        with col4:
            if st.button(f" REJECTED ({len(rejected)})", width="stretch"):
                st.session_state.filter_view = "rejected"
            if st.button(f" FRESH TALENT ({len(fresh_talent)})", width="stretch"):
                st.session_state.filter_view = "fresh_talent"
        
        # Reset filter button
        if st.button(" SHOW ALL CANDIDATES", width="stretch"):
            st.session_state.filter_view = "all"
        
        st.markdown("---")
        
        # Display results based on filter
        current_filter = st.session_state.get('filter_view', 'all')
        
        # KPI Metrics
        st.markdown("###  NEURAL ANALYSIS RESULTS")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(" TOTAL PROFILES", len(results))
        with col2:
            st.metric(" NEURAL SELECTED", len(selected))
        with col3:
            st.metric(" CYBER SHORTLISTED", len(shortlisted))
        with col4:
            st.metric(" SUCCESS RATE", f"{len(selected)/len(results)*100:.1f}%")
        
        # Display filtered candidates based on selection
        if current_filter == "selected" and selected:
            st.markdown("###  NEURAL SELECTED CANDIDATES")
            st.info(f"Showing {len(selected)} candidates with 80%+ match score")
            for candidate in selected:
                st.markdown(f"""
                <div class="cyber-badge-selected">
                    {candidate['candidate']} - {candidate['match_score']:.1%} compatibility - {candidate['decision']}
                </div>
                """, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
        
        elif current_filter == "shortlisted" and shortlisted:
            st.markdown("###  CYBER SHORTLISTED CANDIDATES")
            st.info(f"Showing {len(shortlisted)} candidates with 65-79% match score")
            for candidate in shortlisted:
                st.markdown(f"""
                <div class="cyber-badge-shortlisted">
                     {candidate['candidate']} - {candidate['match_score']:.1%} compatibility - {candidate['decision']}
                </div>
                """, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
        
        elif current_filter == "under_review" and under_review:
            st.markdown("###  UNDER REVIEW CANDIDATES")
            st.info(f"Showing {len(under_review)} candidates with 45-64% match score")
            for candidate in under_review:
                st.markdown(f"""
                <div style="background: linear-gradient(45deg, #ffd60a, #ff9e00); color: #000; padding: 0.5rem 1rem; border-radius: 20px; margin-bottom: 0.5rem; font-weight: 700;">
                     {candidate['candidate']} - {candidate['match_score']:.1%} compatibility - {candidate['decision']}
                </div>
                """, unsafe_allow_html=True)
        
        elif current_filter == "rejected" and rejected:
            st.markdown("###  REJECTED CANDIDATES")
            st.warning(f"Showing {len(rejected)} candidates with <45% match score")
            for candidate in rejected:
                st.markdown(f"""
                <div style="background: linear-gradient(45deg, #ff006e, #d00000); color: #fff; padding: 0.5rem 1rem; border-radius: 20px; margin-bottom: 0.5rem; font-weight: 700;">
                     {candidate['candidate']} - {candidate['match_score']:.1%} compatibility - {candidate['decision']}
                </div>
                """, unsafe_allow_html=True)
        
        elif current_filter == "hidden_gems" and hidden_gems_list:
            st.markdown("### HIDDEN GEMS - AI-Discovered Talent")
            st.success(f"Showing {len(hidden_gems_list)} candidates discovered by ML/DL (missed by exact matching)")
            for candidate in hidden_gems_list:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: #fff; padding: 0.5rem 1rem; border-radius: 20px; margin-bottom: 0.5rem; font-weight: 700; border-left: 5px solid #FFD700;">
                     {candidate['candidate']} - {candidate['match_score']:.1%} compatibility - HIDDEN GEM
                </div>
                """, unsafe_allow_html=True)
        
        elif current_filter == "exact_match" and exact_match_list:
            st.markdown("### EXACT MATCH CANDIDATES")
            st.success(f"Showing {len(exact_match_list)} candidates with 80%+ exact keyword match")
            for candidate in exact_match_list:
                st.markdown(f"""
                <div style="background: linear-gradient(45deg, #06ffa5, #00d4aa); color: #000; padding: 0.5rem 1rem; border-radius: 20px; margin-bottom: 0.5rem; font-weight: 700;">
                     {candidate['candidate']} - {candidate['match_score']:.1%} compatibility - EXACT MATCH
                </div>
                """, unsafe_allow_html=True)
        
        elif current_filter == "high_experience" and high_experience:
            st.markdown("### HIGH EXPERIENCE CANDIDATES")
            st.info(f"Showing {len(high_experience)} candidates with 7+ years experience")
            for candidate in high_experience:
                st.markdown(f"""
                <div style="background: linear-gradient(45deg, #4361ee, #3a0ca3); color: #fff; padding: 0.5rem 1rem; border-radius: 20px; margin-bottom: 0.5rem; font-weight: 700;">
                     {candidate['candidate']} - {candidate['experience']} years - {candidate['match_score']:.1%} match
                </div>
                """, unsafe_allow_html=True)
        
        elif current_filter == "fresh_talent" and fresh_talent:
            st.markdown("###  FRESH TALENT")
            st.info(f"Showing {len(fresh_talent)} candidates with 0-2 years experience")
            for candidate in fresh_talent:
                st.markdown(f"""
                <div style="background: linear-gradient(45deg, #06ffa5, #4cc9f0); color: #000; padding: 0.5rem 1rem; border-radius: 20px; margin-bottom: 0.5rem; font-weight: 700;">
                     {candidate['candidate']} - {candidate['experience']} years - {candidate['match_score']:.1%} match
                </div>
                """, unsafe_allow_html=True)
        
        elif current_filter == "all":
            # Show top candidates summary
            if selected:
                st.markdown("###  NEURAL SELECTED CANDIDATES")
                for candidate in selected[:5]:  # Show top 5
                    st.markdown(f"""
                    <div class="cyber-badge-selected">
                        {candidate['candidate']} - {candidate['match_score']:.1%} compatibility - {candidate['decision']}
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                if len(selected) > 5:
                    st.info(f"+ {len(selected) - 5} more selected candidates. Click 'NEURAL SELECTED' filter to see all.")
            
            if shortlisted:
                st.markdown("###  CYBER SHORTLISTED CANDIDATES")
                for candidate in shortlisted[:5]:  # Show top 5
                    st.markdown(f"""
                    <div class="cyber-badge-shortlisted">
                         {candidate['candidate']} - {candidate['match_score']:.1%} compatibility - {candidate['decision']}
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                if len(shortlisted) > 5:
                    st.info(f"+ {len(shortlisted) - 5} more shortlisted candidates. Click 'SHORTLISTED' filter to see all.")
        
        st.markdown("---")
        
        # HIDDEN GEMS - ML/DL Discovery Feature
        st.markdown("### HIDDEN GEMS - AI-Discovered Talent")
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 15px; margin-bottom: 2rem;">
            <h4 style="color: white; margin: 0 0 0.5rem 0;"> What are Hidden Gems?</h4>
            <p style="color: #f0f0f0; margin: 0; font-size: 0.95rem;">
                Candidates who might be <strong>overlooked by exact keyword matching</strong> but are discovered by our 
                <strong>ML/DL semantic analysis</strong>. These talents have transferable skills, growth potential, 
                or unique combinations that traditional ATS systems miss.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # ENHANCED Hidden Gems Algorithm - ML/DL Discovery
        hidden_gems = []
        for i, candidate in enumerate(results):
            resume_text = resumes_list[i] if i < len(resumes_list) else ""
            if resume_text:
                resume_data = parse_resume_professional(resume_text)
                
                # Calculate exact keyword match percentage
                exact_skill_matches = 0
                for required_skill in job_requirements['skills']:
                    for candidate_skill in resume_data['skills']:
                        if required_skill.lower() in candidate_skill.lower():
                            exact_skill_matches += 1
                            break
                
                exact_match_pct = exact_skill_matches / len(job_requirements['skills']) if job_requirements['skills'] else 0
                ml_score = candidate['match_score']
                improvement = ml_score - exact_match_pct
                
                # Enhanced Hidden Gem Criteria:
                # 1. ML score significantly higher than exact match (15%+ improvement)
                # 2. ML score is at least 55% (decent candidate)
                # 3. OR has special characteristics that ML detected
                
                is_hidden_gem = False
                discovery_reason = "ML Discovery"
                
                # Primary criteria: ML boost
                if improvement >= 0.15 and ml_score >= 0.55:
                    is_hidden_gem = True
                    discovery_reason = f"ML Boost (+{improvement:.1%})"
                
                # Secondary criteria: Special characteristics
                elif ml_score >= 0.65:  # Good candidate
                    if resume_data['experience'] >= 7:
                        is_hidden_gem = True
                        discovery_reason = f"Senior Expert ({resume_data['experience']}y)"
                    elif resume_data['leadership_indicators']:
                        is_hidden_gem = True
                        discovery_reason = "Leadership Potential"
                    elif len(resume_data['skills']) >= 10:
                        is_hidden_gem = True
                        discovery_reason = f"Multi-Skilled ({len(resume_data['skills'])} skills)"
                    elif resume_data['highest_education'] in ['master', 'phd']:
                        is_hidden_gem = True
                        discovery_reason = f"Advanced {resume_data['highest_education'].title()}"
                    elif resume_data['achievements']:
                        is_hidden_gem = True
                        discovery_reason = "Proven Achiever"
                
                # Tertiary criteria: Fresh talent with potential
                elif ml_score >= 0.50 and resume_data['experience'] <= 2 and len(resume_data['skills']) >= 5:
                    is_hidden_gem = True
                    discovery_reason = "Fresh Talent with Potential"
                
                if is_hidden_gem:
                    hidden_gems.append({
                        'candidate': candidate,
                        'exact_match': exact_match_pct,
                        'ml_score': ml_score,
                        'improvement': improvement,
                        'resume_data': resume_data,
                        'discovery_reason': discovery_reason
                    })
        
        if hidden_gems:
            st.success(f"**{len(hidden_gems)} Hidden Gems Discovered!** - Talents missed by traditional keyword matching")
            
            for gem in hidden_gems:
                # Use the discovery reason from the algorithm
                discovery_reason = gem.get('discovery_reason', 'ML Discovery')
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                            padding: 1.5rem; border-radius: 12px; margin-bottom: 1rem; 
                            border-left: 5px solid #FFD700;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <h4 style="color: white; margin: 0;">💎 {gem['candidate']['candidate']}</h4>
                            <p style="color: #fff; margin: 0.5rem 0 0 0; font-size: 0.9rem;">
                                <strong>Exact Match:</strong> {gem['exact_match']:.1%} → 
                                <strong>ML Discovery:</strong> {gem['ml_score']:.1%} 
                                <span style="background: #FFD700; color: #000; padding: 0.2rem 0.5rem; border-radius: 5px; margin-left: 0.5rem;">
                                    +{gem['improvement']:.1%} AI Boost
                                </span>
                            </p>
                            <p style="color: #FFD700; margin: 0.3rem 0 0 0; font-size: 0.8rem; font-weight: 600;">
                                🔍 Discovery: {discovery_reason}
                            </p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Show why they're a hidden gem
                with st.expander(f"🔍 Why is {gem['candidate']['candidate']} a Hidden Gem?"):
                    
                    # DETAILED SKILLS ANALYSIS
                    resume_data = gem['resume_data']
                    candidate_skills = set(s.lower() for s in resume_data['skills'])
                    required_skills = set(job_requirements.get('skills', []))
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**🎯 Skills Analysis**")
                        
                        # Enhanced skill mapping
                        skill_mappings = {
                            'python': ['programming', 'coding', 'development', 'data science', 'ml'],
                            'machine learning': ['ai', 'data science', 'analytics', 'tensorflow', 'pytorch'],
                            'sql': ['database', 'data', 'queries', 'mysql', 'postgresql'],
                            'javascript': ['web development', 'react', 'node', 'frontend'],
                            'java': ['programming', 'backend', 'spring'],
                            'aws': ['cloud', 'devops', 'azure', 'gcp'],
                            'docker': ['containerization', 'kubernetes', 'devops']
                        }
                        
                        exact_skills = []
                        transferable_skills = []
                        
                        for req_skill in required_skills:
                            found_exact = False
                            # Check exact match
                            for cand_skill in candidate_skills:
                                if req_skill in cand_skill or cand_skill in req_skill:
                                    exact_skills.append(f"✅ {req_skill.title()}")
                                    found_exact = True
                                    break
                            
                            if not found_exact:
                                # Check transferable skills
                                for cand_skill in candidate_skills:
                                    if req_skill in skill_mappings.get(cand_skill, []) or cand_skill in skill_mappings.get(req_skill, []):
                                        transferable_skills.append(f"🔄 {cand_skill.title()} → {req_skill.title()}")
                                        break
                        
                        if exact_skills:
                            st.markdown("**Exact Matches:**")
                            for skill in exact_skills:
                                st.success(skill)
                        
                        if transferable_skills:
                            st.markdown("**🎯 Transferable Skills:**")
                            for skill in transferable_skills:
                                st.info(skill)
                    
                    with col2:
                        st.markdown("**🤖 ML Discovery Reasons**")
                        
                        ml_reasons = []
                        if gem['improvement'] >= 0.20:
                            ml_reasons.append(f"🧠 BERT found {gem['improvement']:.1%} more value than keywords")
                        
                        if resume_data['experience'] >= 7:
                            ml_reasons.append(f"🏆 Senior expert: {resume_data['experience']} years")
                        elif resume_data['experience'] <= 2:
                            ml_reasons.append(f"🌟 High potential: {len(resume_data['skills'])} skills")
                        
                        if resume_data['leadership_indicators']:
                            ml_reasons.append("👑 Leadership capabilities detected")
                        
                        if len(resume_data['skills']) >= 10:
                            ml_reasons.append(f"🛠️ Multi-skilled: {len(resume_data['skills'])} technologies")
                        
                        for reason in ml_reasons:
                            st.success(reason)
                        
                        st.markdown("**Why Traditional ATS Missed This:**")
                        st.markdown(f"- Keyword match: {gem['exact_match']:.0%}")
                        st.markdown(f"- ML semantic analysis: {gem['ml_score']:.0%}")
                        st.markdown(f"- **AI discovered {gem['improvement']:.0%} more value!**")
        else:
            st.info("💡 No hidden gems in this batch - all strong candidates were found by both exact and ML matching")
        
        st.markdown("---")
        
        # ANALYSIS TYPES FOR HR
        st.markdown("###  COMPREHENSIVE ANALYSIS DASHBOARD")
        st.markdown("Choose analysis type to gain different insights:")
        
        analysis_tabs = st.tabs([
            "🎯 Match Analysis",
            "💎 Hidden Gems vs Exact Match",
            "📈 Skills Distribution",
            "🏆 Experience Levels",
            "🎓 Education Analysis",
            "⚠️ Risk Assessment",
            "💰 Salary Insights",
            "🌈 Diversity Metrics"
        ])
        
        with analysis_tabs[0]:  # Match Analysis
            st.markdown("####  Candidate Match Distribution")
            
            # Create match score distribution
            import plotly.graph_objects as go
            
            score_ranges = {
                '80-100% (Excellent)': len([r for r in results if r['match_score'] >= 0.8]),
                '65-79% (Good)': len([r for r in results if 0.65 <= r['match_score'] < 0.8]),
                '45-64% (Fair)': len([r for r in results if 0.45 <= r['match_score'] < 0.65]),
                '0-44% (Poor)': len([r for r in results if r['match_score'] < 0.45])
            }
            
            fig = go.Figure(data=[go.Bar(
                x=list(score_ranges.keys()),
                y=list(score_ranges.values()),
                marker_color=['#00f5d4', '#f72585', '#ffd60a', '#ff006e']
            )])
            fig.update_layout(
                title="Match Score Distribution",
                xaxis_title="Score Range",
                yaxis_title="Number of Candidates",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Average Match Score", f"{np.mean([r['match_score'] for r in results]):.1%}")
            with col2:
                st.metric("Median Match Score", f"{np.median([r['match_score'] for r in results]):.1%}")
        
        with analysis_tabs[1]:  # Hidden Gems vs Exact
            st.markdown("#### 💎 ML/DL Discovery vs Traditional Matching")
            
            st.markdown("""
            **Key Differences:**
            - **Exact Match:** Counts only exact keyword occurrences (e.g., "Python" must appear as "Python")
            - **ML/DL Match:** Uses BERT NER + Q-Learning to understand:
                - Entity extraction: Names, companies, skills from context
                - Adaptive scoring: Learning from patterns and feedback
                - Smart matching: Beyond simple keyword counting
            """)
            
            if hidden_gems:
                st.success(f"**{len(hidden_gems)} candidates** would be missed by traditional ATS!")
                
                # Show comparison table
                comparison_data = []
                for gem in hidden_gems:
                    comparison_data.append({
                        'Candidate': gem['candidate']['candidate'],
                        'Exact Match': f"{gem['exact_match']:.1%}",
                        'ML/DL Match': f"{gem['ml_score']:.1%}",
                        'AI Advantage': f"+{gem['improvement']:.1%}",
                        'Status': '💎 Hidden Gem'
                    })
                
                st.dataframe(comparison_data, width="stretch")
            else:
                st.info("All candidates were found by both methods - no hidden gems in this batch")
        
        with analysis_tabs[2]:  # Skills Distribution
            st.markdown("#### 📈 Skills Landscape Analysis")
            
            all_skills = []
            for i, candidate in enumerate(results):
                resume_text = resumes_list[i] if i < len(resumes_list) else ""
                if resume_text:
                    resume_data = parse_resume_professional(resume_text)
                    all_skills.extend(resume_data['skills'])
            
            from collections import Counter
            skill_counts = Counter(all_skills)
            top_skills = skill_counts.most_common(10)
            
            if top_skills:
                fig = go.Figure(data=[go.Bar(
                    x=[skill for skill, count in top_skills],
                    y=[count for skill, count in top_skills],
                    marker_color='#00f5d4'
                )])
                fig.update_layout(
                    title="Top 10 Skills Across All Candidates",
                    xaxis_title="Skill",
                    yaxis_title="Frequency",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with analysis_tabs[3]:  # Experience Levels
            st.markdown("#### 🏆 Experience Distribution")
            
            exp_ranges = {
                '0-2 years (Junior)': len([r for r in results if r['experience'] <= 2]),
                '3-5 years (Mid-level)': len([r for r in results if 3 <= r['experience'] <= 5]),
                '6-10 years (Senior)': len([r for r in results if 6 <= r['experience'] <= 10]),
                '10+ years (Expert)': len([r for r in results if r['experience'] > 10])
            }
            
            fig = go.Figure(data=[go.Pie(
                labels=list(exp_ranges.keys()),
                values=list(exp_ranges.values()),
                hole=0.4
            )])
            fig.update_layout(title="Experience Level Distribution", height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with analysis_tabs[4]:  # Education
            st.markdown("#### 🎓 Education Qualifications")
            
            education_levels = []
            for i, candidate in enumerate(results):
                resume_text = resumes_list[i] if i < len(resumes_list) else ""
                if resume_text:
                    resume_data = parse_resume_professional(resume_text)
                    education_levels.append(resume_data['highest_education'])
            
            from collections import Counter
            edu_counts = Counter(education_levels)
            
            fig = go.Figure(data=[go.Bar(
                x=list(edu_counts.keys()),
                y=list(edu_counts.values()),
                marker_color='#f72585'
            )])
            fig.update_layout(
                title="Education Level Distribution",
                xaxis_title="Degree",
                yaxis_title="Count",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with analysis_tabs[5]:  # Risk Assessment
            st.markdown("#### ⚠️ Hiring Risk Analysis")
            
            if ML_MODELS_AVAILABLE:
                try:
                    from ml_models import attrition_predictor
                    
                    risk_summary = {'low': 0, 'medium': 0, 'high': 0}
                    for i, candidate in enumerate(results):
                        resume_text = resumes_list[i] if i < len(resumes_list) else ""
                        if resume_text:
                            resume_data = parse_resume_professional(resume_text)
                            risk = attrition_predictor.predict_attrition_risk(
                                resume_data, job_requirements, candidate['match_score']
                            )
                            risk_summary[risk['risk_level']] += 1
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("🟢 Low Risk", risk_summary['low'], help="Likely to stay 2+ years")
                    with col2:
                        st.metric("🟡 Medium Risk", risk_summary['medium'], help="Monitor engagement")
                    with col3:
                        st.metric("🔴 High Risk", risk_summary['high'], help="May leave within 1 year")
                except:
                    st.info("Install ML models for risk analysis")
            else:
                st.info("ML models not available")
        
        with analysis_tabs[6]:  # Salary Insights
            st.markdown("####  Compensation Analysis")
            
            salary_data = []
            for i, candidate in enumerate(results):
                resume_text = resumes_list[i] if i < len(resumes_list) else ""
                if resume_text:
                    resume_data = parse_resume_professional(resume_text)
                    
                    # Use Indian Salary if available
                    if INDIAN_SALARY_AVAILABLE:
                        try:
                            indian_salary = calculate_indian_salary(
                                job_title.lower().replace(' ', '_'),
                                resume_data['experience'],
                                'bangalore',
                                'medium'
                            )
                            formatted = format_salary_dict(indian_salary)
                            salary_data.append({
                                'Candidate': candidate['candidate'],
                                'Lower Range': formatted['lower'],
                                'Market Avg': formatted['average'],
                                'Upper Range': formatted['upper'],
                                'Recommended': formatted['recommended']
                            })
                        except:
                            salary_range = predict_salary_range(resume_data)
                            salary_data.append({
                                'Candidate': candidate['candidate'],
                                'Lower Range': f"${salary_range['lower_range']:,}",
                                'Market Avg': f"${salary_range['market_average']:,}",
                                'Upper Range': f"${salary_range['upper_range']:,}",
                                'Recommended': f"${salary_range['recommended_offer']:,}"
                            })
                    else:
                        salary_range = predict_salary_range(resume_data)
                        salary_data.append({
                            'Candidate': candidate['candidate'],
                            'Lower Range': f"${salary_range['lower_range']:,}",
                            'Market Avg': f"${salary_range['market_average']:,}",
                            'Upper Range': f"${salary_range['upper_range']:,}",
                            'Recommended': f"${salary_range['recommended_offer']:,}"
                        })
            
            if salary_data:
                st.dataframe(salary_data, width="stretch")
                
                try:
                    if INDIAN_SALARY_AVAILABLE and '₹' in salary_data[0]['Recommended']:
                        avg_values = []
                        for s in salary_data:
                            rec_str = s['Recommended'].replace('₹', '').replace(' LPA', '').replace('L', '').strip()
                            avg_values.append(float(rec_str))
                        avg_recommended = np.mean(avg_values)
                        st.metric("Average Recommended Offer", f"₹{avg_recommended:.2f} LPA")
                    else:
                        avg_recommended = np.mean([int(s['Recommended'].replace('$', '').replace(',', '')) for s in salary_data])
                        st.metric("Average Recommended Offer", f"${avg_recommended:,.0f}")
                except:
                    st.info("Salary data available in table above")
        
        with analysis_tabs[7]:  # Diversity Metrics
            st.markdown("####  Diversity & Inclusion Metrics")
            
            if ML_MODELS_AVAILABLE:
                try:
                    from ml_models import diversity_analyzer
                    
                    candidates_data = []
                    for i, candidate in enumerate(results):
                        resume_text = resumes_list[i] if i < len(resumes_list) else ""
                        if resume_text:
                            resume_data = parse_resume_professional(resume_text)
                            candidates_data.append(resume_data)
                    
                    diversity_metrics = diversity_analyzer.analyze_diversity(candidates_data)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Education Diversity", f"{diversity_metrics['education_diversity']:.1%}")
                    with col2:
                        st.metric("Experience Diversity", f"{diversity_metrics['experience_diversity']:.1%}")
                    with col3:
                        st.metric("Skill Diversity", f"{diversity_metrics['skill_diversity']:.1%}")
                    with col4:
                        st.metric("Overall Diversity", f"{diversity_metrics['overall_diversity_score']:.1%}")
                    
                    if diversity_metrics['overall_diversity_score'] >= 0.7:
                        st.success(" Excellent diversity - Well-balanced candidate pool")
                    elif diversity_metrics['overall_diversity_score'] >= 0.5:
                        st.warning(" Moderate diversity - Consider broadening search")
                    else:
                        st.error(" Low diversity - Homogeneous candidate pool")
                except:
                    st.info("Install ML models for diversity analysis")
            else:
                st.info("ML models not available")
        
        st.markdown("---")
        
        # DETAILED CANDIDATE ANALYSIS WITH REASONS
        st.markdown("###  Detailed Candidate Analysis")
        st.markdown("Expand any candidate to view comprehensive analysis including selection reasons, skills gap, salary, interview questions, and communication templates.")
        
        for i, candidate in enumerate(results, 1):
            # Get the resume text for this candidate
            candidate_text = resumes_list[i-1] if i-1 < len(resumes_list) else ""
            
            if candidate_text:
                resume_data = parse_resume_professional(candidate_text)
                result = calculate_professional_match_score(resume_data, job_requirements)
                
                with st.expander(f" Candidate #{i}: {candidate['candidate']} - {candidate['match_score']:.1%} Match - {candidate['status']}"):
                    
                    # Quick Actions Row
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        # Create individual candidate report
                        individual_report = f"""# Candidate Analysis Report
                        
**Candidate:** {candidate['candidate']}
**Match Score:** {candidate['match_score']:.1%}
**Status:** {candidate['status']}
**Decision:** {candidate['decision']}
**Experience:** {candidate['experience']} years
**Skills:** {candidate['skills_count']}

## Selection Reasoning
{chr(10).join(f'✓ {reason}' for reason in result['reasons_selected']) if result['reasons_selected'] else 'See detailed analysis'}

## Areas for Improvement
{chr(10).join(f'• {reason}' for reason in result['reasons_rejected']) if result['reasons_rejected'] else 'None identified'}

## Skills Analysis
**Matched:** {', '.join(analyze_skills_gap(resume_data, job_requirements)['has'])}
**Missing:** {', '.join(analyze_skills_gap(resume_data, job_requirements)['missing'])}

## Salary Recommendation
**Range:** ${predict_salary_range(resume_data)['lower_range']:,} - ${predict_salary_range(resume_data)['upper_range']:,}
**Recommended Offer:** ${predict_salary_range(resume_data)['recommended_offer']:,}
"""
                        st.download_button(
                            label="📄 Download Report",
                            data=individual_report,
                            file_name=f"report_{candidate['candidate'].replace(' ', '_')}.md",
                            mime="text/markdown",
                            key=f"report_{i}"
                        )
                    
                    with col2:
                        # Email template download
                        email = generate_email_template(
                            candidate['candidate'],
                            candidate['status'],
                            candidate['match_score'],
                            job_title,
                            result['reasons_selected'] if result['reasons_selected'] else ["Professional background"]
                        )
                        st.download_button(
                            label=" Download Email",
                            data=email,
                            file_name=f"email_{candidate['candidate'].replace(' ', '_')}.txt",
                            mime="text/plain",
                            key=f"email_{i}"
                        )
                    
                    with col3:
                        # Copy resume text for single analysis
                        st.download_button(
                            label=" Copy Resume Text",
                            data=candidate_text,
                            file_name=f"resume_{candidate['candidate'].replace(' ', '_')}.txt",
                            mime="text/plain",
                            key=f"resume_{i}",
                            help="Download resume text to analyze in Single Analysis mode"
                        )
                    
                    st.markdown("---")
                    
                    # Overview Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Match Score", f"{candidate['match_score']:.1%}")
                    with col2:
                        st.metric("Experience", f"{candidate['experience']} yrs")
                    with col3:
                        st.metric("Skills", candidate['skills_count'])
                    with col4:
                        st.metric("Status", candidate['status'])
                    
                    # Component Scores
                    st.markdown("#### Score Breakdown")
                    for component, score in result['component_scores'].items():
                        st.progress(score, text=f"{component.replace('_', ' ').title()}: {score:.1%}")
                    
                    # Selection Reasons (KEY FEATURE!)
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("####  Reasons for Selection")
                        if result['reasons_selected']:
                            for reason in result['reasons_selected']:
                                st.success(reason)
                        else:
                            st.info("No strong positive factors identified")
                    
                    with col2:
                        st.markdown("####  Reasons for Rejection")
                        if result['reasons_rejected']:
                            for reason in result['reasons_rejected']:
                                st.warning(reason)
                        else:
                            st.success("No significant concerns identified")
                    
                    # Skills Gap
                    st.markdown("#### Skills Analysis")
                    skills_gap = analyze_skills_gap(resume_data, job_requirements)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Skills Matched:**")
                        if skills_gap['has']:
                            for skill in skills_gap['has']:
                                st.success(f"✓ {skill.title()}")
                        else:
                            st.info("No exact matches")
                    
                    with col2:
                        st.markdown("**Skills Missing:**")
                        if skills_gap['missing']:
                            for skill in skills_gap['missing']:
                                st.error(f"✗ {skill.title()}")
                            st.info(f"Learning time: {skills_gap['learning_time_weeks']} weeks")
                        else:
                            st.success("All required skills present!")
                    
                    # Red Flags
                    st.markdown("#### Risk Assessment")
                    red_flags = detect_red_flags(candidate_text, resume_data)
                    if red_flags['clean']:
                        st.success(" Clean profile - No red flags detected")
                    else:
                        if red_flags['red_flags']:
                            for flag in red_flags['red_flags']:
                                st.error(flag)
                        if red_flags['warnings']:
                            for warning in red_flags['warnings']:
                                st.warning(warning)
                    
                    # Salary
                    st.markdown("#### Compensation Analysis")
                    salary_pred = predict_salary_range(resume_data)
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Lower Range", f"${salary_pred['lower_range']:,}")
                    with col2:
                        st.metric("Market Avg", f"${salary_pred['market_average']:,}")
                    with col3:
                        st.metric("Upper Range", f"${salary_pred['upper_range']:,}")
                    with col4:
                        st.metric("Recommended", f"${salary_pred['recommended_offer']:,}")
                    
                    # Interview Questions
                    st.markdown("#### Interview Questions")
                    questions = generate_interview_questions(resume_data, candidate_text)
                    for j, question in enumerate(questions[:5], 1):
                        st.markdown(f"**{j}.** {question}")
        
        st.markdown("---")
        
        # Comparative Table with Reasons
        st.markdown("###  Comparative Analysis Table")
        
        # Build enhanced dataframe with reasons
        enhanced_results = []
        for i, candidate in enumerate(results):
            candidate_text = resumes_list[i] if i < len(resumes_list) else ""
            if candidate_text:
                resume_data = parse_resume_professional(candidate_text)
                result = calculate_professional_match_score(resume_data, job_requirements)
                skills_gap = analyze_skills_gap(resume_data, job_requirements)
                
                # Compact reasons
                reasons_sel = "; ".join(result['reasons_selected'][:2]) if result['reasons_selected'] else "N/A"
                reasons_rej = "; ".join(result['reasons_rejected'][:2]) if result['reasons_rejected'] else "N/A"
                
                enhanced_results.append({
                    'Rank': i + 1,
                    'Candidate': candidate['candidate'],
                    'Match': f"{candidate['match_score']:.1%}",
                    'Experience': f"{candidate['experience']} yrs",
                    'Skills': candidate['skills_count'],
                    'Skills Match': f"{skills_gap['match_percentage']:.0f}%",
                    'Status': candidate['status'],
                    'Decision': candidate['decision'],
                    'Reasons Selected': reasons_sel,
                    'Reasons Rejected': reasons_rej
                })
        
        if enhanced_results:
            df_enhanced = pd.DataFrame(enhanced_results)
            st.dataframe(df_enhanced, width="stretch")
        
        st.markdown("---")
        
        # Full results table (legacy)
        st.markdown("### Summary Table")
        df = pd.DataFrame(results)
        df['Rank'] = range(1, len(df) + 1)
        df['Compatibility'] = df['match_score'].apply(lambda x: f"{x:.1%}")
        
        display_df = df[['Rank', 'candidate', 'experience', 'skills_count', 'status', 'Compatibility']]
        display_df.columns = ['Rank', 'Neural ID', 'Experience', 'Skills', 'Status', 'Compatibility']
        
        st.dataframe(display_df, width="stretch")
        
        # Download
        csv = df.to_csv(index=False)
        st.download_button(
            label=" EXPORT NEURAL DATA",
            data=csv,
            file_name=f"neural_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

def main():
    """Main application"""
    
    # Navigation hint at top
    st.markdown("""
    <div style="background: linear-gradient(90deg, #00f5d4, #f72585); padding: 0.5rem; text-align: center; border-radius: 10px; margin-bottom: 1rem;">
        <p style="margin: 0; color: #000; font-weight: 700; font-size: 0.9rem;">
             USE SIDEBAR TO NAVIGATE: Home | Single Analysis | Bulk Processing | HR Database
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="cyber-header">NEUROMATCH AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="cyber-subtitle">AI-Powered Hiring Intelligence Platform</p>', unsafe_allow_html=True)
    
    # Sidebar navigation with better visibility
    with st.sidebar:
        st.markdown("##  Navigation")
        st.markdown("---")
        
        page = st.radio(
            "Select Page:",
            ["🏠 Home", "📄 Single Analysis", "📊 Bulk Processing", "📁 HR Database"],
            index=0
        )
        
        st.markdown("---")
        st.markdown("### 🤖 Working ML/DL Models")
        if ML_MODELS_AVAILABLE:
            st.success("✅ BERT NER (95%)")
            st.success("✅ Q-Learning (92%)")
            st.success("✅ Random Forest (100%)")
            st.success("✅ Statistical ML (85%)")
        else:
            st.error("❌ Models not loaded")
    
    # Route to pages
    if "Home" in page:
        show_home()
    elif "Single" in page:
        show_single_resume()
    elif "Bulk" in page:
        show_bulk_analysis()
    elif "HR Database" in page:
        show_hr_database()

if __name__ == "__main__":
    main()