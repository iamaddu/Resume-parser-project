import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# For demonstration, you can expand these as needed
EDUCATION_LEVELS = ["phd", "master", "bachelor", "mba", "m.tech", "b.tech"]

def create_feature_vector(resume_data, job_keywords):
    """
    resume_data: dict from parse_resume()
    job_keywords: list of keywords (strings)
    Returns: numpy array feature vector
    """
    # Binary features for each job keyword in skills
    skill_features = [1 if kw.lower() in [s.lower() for s in resume_data["skills"]] else 0 for kw in job_keywords]
    # Years of experience (as float, 0 if missing)
    try:
        exp = float(resume_data["experience"])
    except (ValueError, TypeError):
        exp = 0.0
    # Binary features for education levels
    edu_text = " ".join(resume_data["education"]).lower()
    edu_features = [1 if level in edu_text else 0 for level in EDUCATION_LEVELS]
    # Combine all features
    feature_vector = np.array(skill_features + [exp] + edu_features)
    return feature_vector

# Rule-based scoring function (for production use)
def rule_based_score(resume_data, job_keywords):
    skill_matches = sum([1 for kw in job_keywords if kw.lower() in [s.lower() for s in resume_data["skills"]]])
    total_keywords = len(job_keywords) if job_keywords else 1
    skill_score = skill_matches / total_keywords
    try:
        exp_score = min(float(resume_data["experience"]) / 10, 1.0)  # Cap at 10 years
    except (ValueError, TypeError):
        exp_score = 0.0
    edu_score = 0.2 if resume_data["education"] else 0.0
    final_score = skill_score * 0.6 + exp_score * 0.2 + edu_score * 0.2
    return round(final_score * 100, 2)  # Return as percentage

def train_dummy_model():
    """
    Trains a RandomForestClassifier on a synthetic dataset for demonstration.
    Returns the trained model and the list of job keywords used for features.
    """
    # Synthetic job keywords and data
    job_keywords = ["python", "machine learning", "sql", "aws", "excel"]
    X = []
    y = []
    # Create 20 synthetic resumes (10 relevant, 10 not relevant)
    for i in range(10):
        # Relevant: has most keywords, 5+ years exp, master's degree
        resume = {
            "skills": ["python", "machine learning", "sql"],
            "experience": str(5 + i),
            "education": ["Master of Science"]
        }
        X.append(create_feature_vector(resume, job_keywords))
        y.append(1)
    for i in range(10):
        # Not relevant: few/no keywords, <3 years exp, no master's
        resume = {
            "skills": ["excel"],
            "experience": str(i % 3),
            "education": ["Bachelor of Arts"]
        }
        X.append(create_feature_vector(resume, job_keywords))
        y.append(0)
    X = np.array(X)
    y = np.array(y)
    # Grid search for best RandomForest parameters
    param_grid = {
        "n_estimators": [10, 50],
        "max_depth": [2, 4, None]
    }
    clf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3)
    clf.fit(X, y)
    return clf.best_estimator_, job_keywords