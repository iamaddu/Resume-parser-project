import importlib

# Dynamically load a PDF reader to avoid unresolved import errors in some editors/linters.
# Prefer the maintained 'pypdf' package and fall back to PyPDF2 (or None if neither is available).
PdfReader = None
try:
    if importlib.util.find_spec("pypdf") is not None:
        pypdf = importlib.import_module("pypdf")
        PdfReader = getattr(pypdf, "PdfReader", None)
    elif importlib.util.find_spec("PyPDF2") is not None:
        pyPdf2 = importlib.import_module("PyPDF2")
        # PyPDF2 historically exposed PdfReader or PdfFileReader depending on version
        PdfReader = getattr(pyPdf2, "PdfReader", getattr(pyPdf2, "PdfFileReader", None))
    else:
        PdfReader = None
except Exception:
    PdfReader = None

import spacy
import re

# Load spaCy English model (make sure to run: python -m spacy download en_core_web_sm)
nlp = spacy.load("en_core_web_sm")

# Example skill and degree keywords (expand as needed)
SKILL_KEYWORDS = {
    "python", "java", "c++", "sql", "machine learning", "deep learning", "nlp", "pandas",
    "numpy", "tensorflow", "scikit-learn", "docker", "kubernetes", "aws", "azure", "linux",
    "git", "excel", "tableau", "powerbi", "spark", "hadoop", "flask", "django", "javascript"
}
DEGREE_KEYWORDS = [
    "bachelor", "master", "phd", "b.sc", "m.sc", "bs", "ms", "b.tech", "m.tech", "mba", "bca", "mca"
]

# Skill synonyms and related skills mapping
SKILL_SYNONYMS = {
    "python": ["py", "python3", "python2"],
    "machine learning": ["ml", "ai", "artificial intelligence", "deep learning"],
    "sql": ["mysql", "postgresql", "sqlite", "database", "db"],
    "excel": ["spreadsheet", "ms excel"],
    "javascript": ["js", "nodejs", "node.js"],
    "c++": ["cpp"],
    "git": ["version control", "github", "gitlab"],
    "aws": ["amazon web services", "cloud", "ec2", "s3"],
    # Add more as needed
}

def normalize_skill(skill):
    skill = skill.lower().strip()
    for canonical, synonyms in SKILL_SYNONYMS.items():
        if skill == canonical or skill in synonyms:
            return canonical
    return skill

def extract_skills_with_synonyms(text):
    doc = nlp(text)
    found_skills = set()
    tokens = [token.text.lower() for token in doc]
    for skill in SKILL_KEYWORDS:
        if skill in tokens:
            found_skills.add(skill)
        # Check synonyms
        if skill in SKILL_SYNONYMS:
            for synonym in SKILL_SYNONYMS[skill]:
                if synonym in tokens:
                    found_skills.add(skill)
    return list(found_skills)

def extract_text_from_pdf(pdf_path):
    text = ""
    if PdfReader is None:
        raise ImportError("pypdf or PyPDF2 is required to extract PDF text. Install 'pypdf' (pip install pypdf).")
    with open(pdf_path, "rb") as f:
        # PdfReader was set above to the appropriate class from pypdf or PyPDF2
        reader = PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def parse_resume(text):
    doc = nlp(text)
    # Skills extraction (with synonyms)
    skills = set(extract_skills_with_synonyms(text))
    # Experience extraction (look for years)
    experience = ""
    exp_match = re.search(r'(\d+)\s+years?', text, re.IGNORECASE)
    if exp_match:
        experience = exp_match.group(1)
    # Education extraction (NER + keyword)
    education = []
    for sent in doc.sents:
        for deg in DEGREE_KEYWORDS:
            if deg in sent.text.lower():
                education.append(sent.text.strip())
    # Name extraction (prefer PERSON entity, fallback to first non-empty line)
    name = ""
    for ent in doc.ents:
        if ent.label_ == "PERSON" and len(ent.text.split()) <= 4:
            name = ent.text
            break
    if not name:
        for line in text.split('\n'):
            if line.strip() and not any(x in line.lower() for x in ["skill", "education", "experience", "email", "@"]):
                name = line.strip()
                break
    # Email extraction
    email_match = re.search(r'[\w\.-]+@[\w\.-]+', text)
    email = email_match.group(0) if email_match else ""
    # Phone extraction
    phone_match = re.search(r'(\+?\d{1,3}[-.\s]?)?(\d{10})', text)
    phone = phone_match.group(0) if phone_match else ""
    return {
        "name": name,
        "email": email,
        "phone": phone,
        "skills": list(skills),
        "experience": experience,
        "education": education
    }

def detect_ai_generated(text):
    # Simple heuristic: flag if text contains typical AI phrases or is too generic
    ai_phrases = [
        "As an AI language model", "I am a highly motivated individual",
        "I have a proven track record", "I am passionate about", "I am excited to apply"
    ]
    generic_count = sum(phrase.lower() in text.lower() for phrase in ai_phrases)
    # Flag if generic_count is high or text is very long and impersonal
    if generic_count > 0 or len(text.split()) > 1000:
        return True
    return False

def is_resume_pdf(text: str) -> bool:
    """
    Improved heuristic to check if a PDF text is likely a resume.
    Looks for common resume sections, patterns, and avoids assignment/letter formats.
    """
    if not text or len(text) < 100:
        return False
    resume_sections = [
        "education", "experience", "skills", "projects", "certifications",
        "summary", "objective", "work history", "employment", "contact", "profile"
    ]
    assignment_keywords = [
        "assignment", "lab", "experiment", "report", "question", "answer", "submitted by", "teacher", "student id"
    ]
    found_sections = sum(1 for section in resume_sections if section in text.lower())
    found_assignment = any(word in text.lower() for word in assignment_keywords)
    # At least 2 resume sections and not an assignment
    return found_sections >= 2 and not found_assignment