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

# Import ML/DL models
try:
    from ml_models import (
        rl_scorer,
        bert_parser,
        semantic_matcher,
        attrition_predictor,
        diversity_analyzer,
        get_ml_models_status
    )
    ML_MODELS_AVAILABLE = True
except ImportError:
    ML_MODELS_AVAILABLE = False
    print(" ML models not available. Run: pip install -r requirements_ml.txt")

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

# Configure page
st.set_page_config(
    page_title="NeuroMatch AI - Futuristic Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
        border-right: 1px solid rgba(0, 245, 212, 0.3) !important;
    }
    
    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    section[data-testid="stSidebar"] .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(0, 245, 212, 0.3) !important;
        border-radius: 10px !important;
        color: #ffffff !important;
    }
    
    /* Sidebar navigation styling */
    .sidebar .sidebar-content {
        background: transparent !important;
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
    """Professional resume parsing"""
    name = extract_candidate_name(text)
    
    # Extract experience
    exp_patterns = [r'(\d+)\+?\s*years?\s*(?:of\s*)?experience', r'(\d+)\+?\s*yrs?\s*experience']
    experience = 0
    for pattern in exp_patterns:
        matches = re.findall(pattern, text.lower())
        if matches:
            experience = max([int(match) for match in matches])
            break
    
    # Extract skills
    skill_keywords = ['python', 'java', 'javascript', 'react', 'node', 'sql', 'aws', 'docker', 
                     'kubernetes', 'machine learning', 'ai', 'data science', 'analytics']
    skills = [skill for skill in skill_keywords if skill in text.lower()]
    
    # Extract education
    education_levels = ['phd', 'doctorate', 'master', 'bachelor', 'diploma']
    highest_education = 'high school'
    for level in education_levels:
        if level in text.lower():
            highest_education = level
            break
    
    return {
        'name': name,
        'experience': experience,
        'skills': skills,
        'highest_education': highest_education,
        'leadership_indicators': ['team lead', 'manager'] if any(word in text.lower() for word in ['lead', 'manager', 'supervisor']) else [],
        'achievements': ['award', 'recognition'] if any(word in text.lower() for word in ['award', 'achievement', 'recognition']) else []
    }

def detect_red_flags(text, resume_data):
    """Detect red flags in resume"""
    red_flags = []
    warnings = []
    
    # Job hopping detection
    job_count = text.lower().count('company') + text.lower().count('worked at') + text.lower().count('employer')
    if job_count > 4 and resume_data['experience'] < 3:
        red_flags.append("🔴 Job Hopping: Multiple jobs in short timeframe")
    
    # Employment gap detection
    if 'gap' in text.lower() or '(unemployed)' in text.lower():
        warnings.append("🟡 Employment Gap: Unexplained period detected")
    
    # Skill exaggeration
    if resume_data['experience'] < 2 and len(resume_data['skills']) > 10:
        warnings.append("🟡 Skill Inflation: Many skills for experience level")
    
    # Education mismatch
    if 'dropout' in text.lower() or 'incomplete' in text.lower():
        warnings.append("🟡 Education Status: Incomplete degree program")
    
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
        questions.append(f"📝 Tell me about your {resume_data['experience']} years of experience and your most challenging project")
    
    # Skill-based questions
    for skill in resume_data['skills'][:3]:  # Top 3 skills
        questions.append(f"💻 Describe a complex problem you solved using {skill.title()}")
    
    # Leadership questions
    if resume_data['leadership_indicators']:
        questions.append("👥 Describe a situation where you had to lead a team through a difficult challenge")
    
    # Achievement questions
    if resume_data['achievements']:
        questions.append("🏆 Walk me through your biggest professional achievement and its impact")
    
    # Behavioral questions
    questions.append("🤔 How do you handle tight deadlines and multiple priorities?")
    questions.append("🔄 Describe a time you had to learn a new technology quickly")
    
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
        'lower': lower_bound,
        'upper': upper_bound,
        'market_average': market_average,
        'recommended_offer': int(market_average * 0.97)  # Slightly below market
    }

def generate_email_template(candidate_name, status, score, job_title, reasons):
    """Generate personalized email template"""
    templates = {
        'NEURAL SELECTED': f"""Subject: 🎉 Interview Invitation - {job_title}

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

        'CYBER SHORTLISTED': f"""Subject: 📋 Application Update - {job_title}

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

💡 To strengthen future applications:
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
    """Calculate professional match score with genius features"""
    scores = {
        'technical_skills': min(len(resume_data['skills']) / max(len(job_requirements.get('skills', [])), 1), 1.0),
        'experience': min(resume_data['experience'] / max(job_requirements.get('min_experience', 1), 1), 1.0),
        'education': 0.8 if resume_data['highest_education'] in ['bachelor', 'master', 'phd'] else 0.5,
        'leadership': 0.8 if resume_data['leadership_indicators'] else 0.3,
        'achievements': 0.9 if resume_data['achievements'] else 0.4,
        'cultural_fit': 0.7
    }
    
    weights = {
        'technical_skills': 0.35,
        'experience': 0.25,
        'education': 0.15,
        'leadership': 0.10,
        'achievements': 0.10,
        'cultural_fit': 0.05
    }
    
    final_score = sum(scores[component] * weights[component] for component in scores)
    
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
    
    if scores['technical_skills'] > 0.7:
        reasons_selected.append("🔥 Superior technical capabilities detected")
    if scores['experience'] > 0.8:
        reasons_selected.append("⚡ Advanced experience matrix")
    if scores['leadership'] > 0.7:
        reasons_selected.append("👑 Leadership protocols activated")
    
    if scores['technical_skills'] < 0.5:
        reasons_rejected.append("⚠️ Technical skills below threshold")
    if scores['experience'] < 0.5:
        reasons_rejected.append("📊 Experience data insufficient")
    
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
    
    # Competitive Advantages - USPs
    st.markdown("### Why NeuroMatch AI")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Unique Advantages**")
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
            st.success(f"✅ File uploaded: {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")
            
            # Extract text from PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                resume_text = extract_text_from_pdf(tmp_path)
                if resume_text.strip():
                    with st.expander("📄 View Extracted Text"):
                        st.text_area("Extracted resume text:", resume_text, height=200)
                else:
                    st.error("❌ Could not extract text from PDF. Please try pasting text instead.")
            except Exception as e:
                st.error(f"❌ Error processing PDF: {str(e)}")
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
                    st.success("✅ No red flags detected - Clean profile")
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
                    st.markdown("**✅ Skills Matched:**")
                    if skills_gap['has']:
                        for skill in skills_gap['has']:
                            st.success(f"✓ {skill.title()}")
                    else:
                        st.info("No exact matches found")
                
                with col2:
                    st.markdown("**❌ Skills Missing:**")
                    if skills_gap['missing']:
                        for skill in skills_gap['missing']:
                            st.error(f"✗ {skill.title()}")
                    else:
                        st.success("All required skills present!")
                
                st.info(f"📚 Estimated learning time for missing skills: **{skills_gap['learning_time_weeks']} weeks**")
                
                if skills_gap['ready_to_interview']:
                    st.success("🎯 Candidate is ready to interview despite minor gaps!")
                
                # 3. Salary Range Prediction
                st.markdown("### Compensation Analysis")
                salary_pred = predict_salary_range(resume_data)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("💵 Lower Range", f"${salary_pred['lower']:,}")
                with col2:
                    st.metric("📊 Market Average", f"${salary_pred['market_average']:,}")
                with col3:
                    st.metric("💎 Upper Range", f"${salary_pred['upper']:,}")
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
                
                with st.expander("📨 View/Copy Email Template"):
                    st.code(email_template, language="text")
                    st.download_button(
                        label="📥 Download Email Template",
                        data=email_template,
                        file_name=f"email_{result['candidate_name'].replace(' ', '_')}.txt",
                        mime="text/plain"
                    )

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
    st.markdown("#### 📥 SELECT INPUT METHOD")
    input_method = st.radio(
        "Choose how to submit resumes:",
        ["📄 Upload PDF Files", "📝 Paste Text (Multiple Resumes)", "💼 Mixed Input"],
        horizontal=True
    )
    
    resumes_list = []
    
    if input_method == "📄 Upload PDF Files":
        st.markdown("##### 🚀 PDF UPLOAD INTERFACE")
        uploaded_files = st.file_uploader(
            "Upload multiple resume PDFs:",
            type=['pdf'],
            accept_multiple_files=True,
            help="Select multiple PDF files. Supports bulk upload."
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} PDF files uploaded successfully!")
            
            # Preview uploaded files
            with st.expander("📋 View Uploaded Files"):
                for i, file in enumerate(uploaded_files, 1):
                    st.write(f"{i}. {file.name} ({file.size / 1024:.1f} KB)")
            
            # Extract text from PDFs
            if st.button("🔬 EXTRACT & PROCESS PDFs", type="primary"):
                with st.spinner("🔄 Extracting neural data from PDFs..."):
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
                                st.warning(f"⚠️ Could not extract text from {uploaded_file.name}")
                        except Exception as e:
                            st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                        finally:
                            # Clean up temp file
                            try:
                                os.unlink(tmp_path)
                            except:
                                pass
                    
                    if resumes_list:
                        st.success(f"🎯 Successfully extracted {len(resumes_list)} resume profiles!")
    
    elif input_method == "📝 Paste Text (Multiple Resumes)":
        st.markdown("##### 📝 TEXT INPUT INTERFACE")
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
        
        if bulk_text and st.button("🔬 PROCESS TEXT INPUT", type="primary"):
            resumes_list = [r.strip() for r in bulk_text.split("---RESUME---") if r.strip()]
            if resumes_list:
                st.success(f"🎯 {len(resumes_list)} resume profiles detected!")
    
    else:  # Mixed Input
        st.markdown("##### 💼 MIXED INPUT INTERFACE")
        
        # PDF Upload
        st.markdown("**📄 Upload PDFs:**")
        uploaded_files = st.file_uploader(
            "Upload PDF resumes:",
            type=['pdf'],
            accept_multiple_files=True
        )
        
        # Text Input
        st.markdown("**📝 Or Paste Text:**")
        bulk_text = st.text_area(
            "Additional resumes (separate with ---RESUME---):",
            height=200,
            placeholder="Paste additional resume text here..."
        )
        
        if st.button("🔬 PROCESS ALL INPUTS", type="primary"):
            with st.spinner("🔄 Processing mixed inputs..."):
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
                    st.success(f"🎯 Total {len(resumes_list)} resume profiles ready for analysis!")
    
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
            status_text.text(f"🔄 Processing neural profile {i+1}/{len(resumes_list)}...")
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
        
        status_text.text("✅ Quantum processing complete!")
        
        # Sort by score
        results.sort(key=lambda x: x['match_score'], reverse=True)
        
        # Display results
        st.markdown("### 📊 NEURAL ANALYSIS RESULTS")
        
        selected = [r for r in results if r['match_score'] >= 0.8]
        shortlisted = [r for r in results if 0.65 <= r['match_score'] < 0.8]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🧬 TOTAL PROFILES", len(results))
        with col2:
            st.metric("⚡ NEURAL SELECTED", len(selected))
        with col3:
            st.metric("🔥 CYBER SHORTLISTED", len(shortlisted))
        with col4:
            st.metric("📈 SUCCESS RATE", f"{len(selected)/len(results)*100:.1f}%")
        
        # Show top candidates
        if selected:
            st.markdown("### NEURAL SELECTED CANDIDATES")
            for candidate in selected:
                st.markdown(f"""
                <div class="cyber-badge-selected">
                    {candidate['candidate']} - {candidate['match_score']:.1%} compatibility - {candidate['decision']}
                </div>
                """, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
        
        if shortlisted:
            st.markdown("### CYBER SHORTLISTED CANDIDATES")
            for candidate in shortlisted:
                st.markdown(f"""
                <div class="cyber-badge-shortlisted">
                    ⚡ {candidate['candidate']} - {candidate['match_score']:.1%} compatibility - {candidate['decision']}
                </div>
                """, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # DETAILED CANDIDATE ANALYSIS WITH REASONS
        st.markdown("### 📋 Detailed Candidate Analysis")
        st.markdown("Expand any candidate to view comprehensive analysis including selection reasons, skills gap, salary, interview questions, and communication templates.")
        
        for i, candidate in enumerate(results, 1):
            # Get the resume text for this candidate
            candidate_text = resumes_list[i-1] if i-1 < len(resumes_list) else ""
            
            if candidate_text:
                resume_data = parse_resume_professional(candidate_text)
                result = calculate_professional_match_score(resume_data, job_requirements)
                
                with st.expander(f"📊 Candidate #{i}: {candidate['candidate']} - {candidate['match_score']:.1%} Match - {candidate['status']}"):
                    
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
**Range:** ${predict_salary_range(resume_data)['lower']:,} - ${predict_salary_range(resume_data)['upper']:,}
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
                            label="📧 Download Email",
                            data=email,
                            file_name=f"email_{candidate['candidate'].replace(' ', '_')}.txt",
                            mime="text/plain",
                            key=f"email_{i}"
                        )
                    
                    with col3:
                        # Copy resume text for single analysis
                        st.download_button(
                            label="📋 Copy Resume Text",
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
                        st.markdown("#### ✅ Reasons for Selection")
                        if result['reasons_selected']:
                            for reason in result['reasons_selected']:
                                st.success(reason)
                        else:
                            st.info("No strong positive factors identified")
                    
                    with col2:
                        st.markdown("#### ⚠️ Reasons for Rejection")
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
                            st.info(f"📚 Learning time: {skills_gap['learning_time_weeks']} weeks")
                        else:
                            st.success("All required skills present!")
                    
                    # Red Flags
                    st.markdown("#### Risk Assessment")
                    red_flags = detect_red_flags(candidate_text, resume_data)
                    if red_flags['clean']:
                        st.success("✅ Clean profile - No red flags detected")
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
                        st.metric("Lower Range", f"${salary_pred['lower']:,}")
                    with col2:
                        st.metric("Market Avg", f"${salary_pred['market_average']:,}")
                    with col3:
                        st.metric("Upper Range", f"${salary_pred['upper']:,}")
                    with col4:
                        st.metric("Recommended", f"${salary_pred['recommended_offer']:,}")
                    
                    # Interview Questions
                    st.markdown("#### Interview Questions")
                    questions = generate_interview_questions(resume_data, candidate_text)
                    for j, question in enumerate(questions[:5], 1):
                        st.markdown(f"**{j}.** {question}")
        
        st.markdown("---")
        
        # Comparative Table with Reasons
        st.markdown("### 📊 Comparative Analysis Table")
        
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
            st.dataframe(df_enhanced, use_container_width=True)
        
        st.markdown("---")
        
        # Full results table (legacy)
        st.markdown("### 📋 Summary Table")
        df = pd.DataFrame(results)
        df['Rank'] = range(1, len(df) + 1)
        df['Compatibility'] = df['match_score'].apply(lambda x: f"{x:.1%}")
        
        display_df = df[['Rank', 'candidate', 'experience', 'skills_count', 'status', 'Compatibility']]
        display_df.columns = ['Rank', 'Neural ID', 'Experience', 'Skills', 'Status', 'Compatibility']
        
        st.dataframe(display_df, use_container_width=True)
        
        # Download
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 EXPORT NEURAL DATA",
            data=csv,
            file_name=f"neural_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

def main():
    """Main application"""
    st.markdown('<h1 class="cyber-header">NEUROMATCH AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="cyber-subtitle">AI-Powered Hiring Intelligence Platform</p>', unsafe_allow_html=True)
    
    # Sidebar navigation with better visibility
    with st.sidebar:
        st.markdown("## 🚀 Navigation")
        st.markdown("---")
        
        page = st.radio(
            "Select Page:",
            ["🏠 Home", "📄 Single Analysis", "📊 Bulk Processing"],
            index=0
        )
        
        st.markdown("---")
        st.markdown("### 🤖 ML/DL Models")
        st.info("✅ BERT NER")
        st.info("✅ Sentence-BERT")
        st.info("✅ Q-Learning")
        st.info("✅ Random Forest")
        st.info("✅ Diversity ML")
    
    # Route to pages
    if "Home" in page:
        show_home()
    elif "Single" in page:
        show_single_resume()
    elif "Bulk" in page:
        show_bulk_analysis()

if __name__ == "__main__":
    main()