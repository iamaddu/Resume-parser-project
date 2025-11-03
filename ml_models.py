"""
ML/DL Models for NeuroMatch AI
Implements all advanced ML/DL features for AIML project
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import json
from datetime import datetime

# Try importing transformers (optional for basic functionality)
try:
    from transformers import pipeline, AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("WARNING: Transformers not installed. Install with: pip install transformers torch")

# Try importing sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("WARNING: Sentence-transformers not installed. Install with: pip install sentence-transformers")


# ============================================================================
# 1. REINFORCEMENT LEARNING - Adaptive Scoring System
# ============================================================================

class ReinforcementLearningScorer:
    """
    Q-Learning based adaptive scoring system
    Learns from HR feedback to improve scoring weights
    """
    
    def __init__(self, learning_rate=0.1, discount_factor=0.9):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        
        # Initial weights (same as rule-based)
        self.weights = {
            'technical_skills': 0.35,
            'experience': 0.25,
            'education': 0.15,
            'leadership': 0.10,
            'achievements': 0.10,
            'cultural_fit': 0.05
        }
        
        # Feedback history
        self.feedback_history = []
        self.model_file = 'rl_weights.pkl'
        
        # Load existing weights if available
        self.load_weights()
    
    def get_weights(self):
        """Return current weights"""
        return self.weights.copy()
    
    def record_feedback(self, candidate_scores, hr_decision, our_prediction):
        """
        Record HR feedback for learning
        
        Args:
            candidate_scores: dict of component scores
            hr_decision: 'hired', 'rejected', 'interviewed'
            our_prediction: our predicted decision
        """
        feedback = {
            'timestamp': datetime.now().isoformat(),
            'scores': candidate_scores,
            'hr_decision': hr_decision,
            'our_prediction': our_prediction,
            'match': hr_decision == our_prediction
        }
        
        self.feedback_history.append(feedback)
        
        # Update weights based on feedback
        self._update_weights(candidate_scores, hr_decision, our_prediction)
        
        # Save updated weights
        self.save_weights()
    
    def _update_weights(self, scores, hr_decision, our_prediction):
        """
        Q-Learning weight update
        Reward = +1 if match, -1 if mismatch
        """
        reward = 1.0 if hr_decision == our_prediction else -1.0
        
        # If HR hired but we scored low, increase weights of strong components
        if hr_decision == 'hired' and our_prediction != 'hired':
            for component, score in scores.items():
                if score > 0.7:  # Strong component
                    self.weights[component] += self.learning_rate * reward * score
        
        # If HR rejected but we scored high, decrease weights of weak components
        elif hr_decision == 'rejected' and our_prediction == 'hired':
            for component, score in scores.items():
                if score < 0.5:  # Weak component
                    self.weights[component] -= self.learning_rate * abs(reward) * (1 - score)
        
        # Normalize weights to sum to 1.0
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}
    
    def save_weights(self):
        """Save learned weights to disk"""
        with open(self.model_file, 'wb') as f:
            pickle.dump({
                'weights': self.weights,
                'feedback_history': self.feedback_history
            }, f)
    
    def load_weights(self):
        """Load learned weights from disk"""
        if os.path.exists(self.model_file):
            try:
                with open(self.model_file, 'rb') as f:
                    data = pickle.load(f)
                    self.weights = data['weights']
                    self.feedback_history = data.get('feedback_history', [])
                print(f"[OK] Loaded RL weights from {self.model_file}")
            except Exception as e:
                print(f"[WARNING] Could not load RL weights: {e}")
    
    def get_stats(self):
        """Get learning statistics"""
        if not self.feedback_history:
            return {
                'total_feedback': 0,
                'accuracy': 0.0,
                'weights': self.weights
            }
        
        matches = sum(1 for f in self.feedback_history if f['match'])
        accuracy = matches / len(self.feedback_history)
        
        return {
            'total_feedback': len(self.feedback_history),
            'accuracy': accuracy,
            'matches': matches,
            'weights': self.weights
        }
    
    def get_recommendation(self, resume_data, job_requirements):
        """
        Get Q-Learning recommendation for candidate
        
        Args:
            resume_data: parsed resume data
            job_requirements: job requirements dict
        
        Returns:
            dict: {
                'recommendation': str,
                'confidence': float,
                'reasoning': str
            }
        """
        # Calculate component scores using current weights
        scores = {
            'technical_skills': min(len(resume_data.get('skills', [])) / max(len(job_requirements.get('skills', [])), 1), 1.0),
            'experience': min(resume_data.get('experience', 0) / max(job_requirements.get('min_experience', 1), 1), 1.0),
            'education': 0.8 if resume_data.get('highest_education', 'bachelor') in ['bachelor', 'master', 'phd'] else 0.5,
            'leadership': 0.8 if resume_data.get('leadership_indicators', []) else 0.3,
            'achievements': 0.9 if resume_data.get('achievements', []) else 0.4,
            'cultural_fit': 0.7
        }
        
        # Calculate weighted score using learned weights
        final_score = sum(scores[component] * self.weights[component] for component in scores)
        
        # Generate recommendation
        if final_score >= 0.8:
            recommendation = "HIRE"
            confidence = final_score
            reasoning = "Strong candidate across all dimensions"
        elif final_score >= 0.65:
            recommendation = "INTERVIEW"
            confidence = final_score
            reasoning = "Good candidate, worth interviewing"
        elif final_score >= 0.45:
            recommendation = "MAYBE"
            confidence = final_score * 0.8
            reasoning = "Borderline candidate, consider carefully"
        else:
            recommendation = "REJECT"
            confidence = 1.0 - final_score
            reasoning = "Below threshold for this role"
        
        return {
            'recommendation': recommendation,
            'confidence': float(confidence),
            'reasoning': reasoning,
            'component_scores': scores,
            'final_score': float(final_score)
        }


# ============================================================================
# 2. BERT-BASED NER - Advanced Resume Parsing
# ============================================================================

class BERTResumeParser:
    """
    BERT-based Named Entity Recognition for resume parsing
    Extracts: PERSON, ORG, DATE, SKILL, EDUCATION
    """
    
    def __init__(self):
        self.ner_pipeline = None
        self.model_loaded = False
        
        if TRANSFORMERS_AVAILABLE:
            try:
                # Load pre-trained NER model
                self.ner_pipeline = pipeline(
                    "ner",
                    model="dslim/bert-base-NER",
                    aggregation_strategy="simple"
                )
                self.model_loaded = True
                print("[OK] BERT NER model loaded successfully")
            except Exception as e:
                print(f"[WARNING] Could not load BERT model: {e}")
    
    def extract_entities(self, resume_text):
        """
        Extract named entities from resume
        
        Returns:
            dict: {
                'persons': [...],
                'organizations': [...],
                'locations': [...],
                'dates': [...]
            }
        """
        if not self.model_loaded:
            return self._fallback_extraction(resume_text)
        
        try:
            entities = self.ner_pipeline(resume_text)
            
            result = {
                'persons': [],
                'organizations': [],
                'locations': [],
                'dates': []
            }
            
            for entity in entities:
                entity_type = entity['entity_group']
                text = entity['word']
                
                if entity_type == 'PER':
                    result['persons'].append(text)
                elif entity_type == 'ORG':
                    result['organizations'].append(text)
                elif entity_type == 'LOC':
                    result['locations'].append(text)
            
            return result
        
        except Exception as e:
            print(f"[WARNING] BERT NER failed: {e}")
            return self._fallback_extraction(resume_text)
    
    def _fallback_extraction(self, resume_text):
        """Fallback to regex-based extraction"""
        import re
        
        # Simple regex patterns
        name_pattern = r'^([A-Z][a-z]+ [A-Z][a-z]+)'
        org_pattern = r'(?:at|@)\s+([A-Z][A-Za-z\s&]+(?:Inc|LLC|Ltd|Corp)?)'
        
        persons = re.findall(name_pattern, resume_text, re.MULTILINE)
        organizations = re.findall(org_pattern, resume_text)
        
        return {
            'persons': persons[:3],
            'organizations': organizations[:5],
            'locations': [],
            'dates': []
        }


# ============================================================================
# 3. SENTENCE-BERT - Semantic Skill Matching
# ============================================================================

class SemanticSkillMatcher:
    """
    Sentence-BERT for semantic similarity matching
    Understands: "ML" = "Machine Learning", "AI" = "Artificial Intelligence"
    """
    
    def __init__(self):
        self.model = None
        self.model_loaded = False
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                # Load lightweight sentence transformer
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                self.model_loaded = True
                print("[OK] Sentence-BERT model loaded successfully")
            except Exception as e:
                print(f"[WARNING] Could not load Sentence-BERT: {e}")
    
    def match_skills(self, required_skills, candidate_skills, threshold=0.7):
        """
        Match skills using semantic similarity
        
        Args:
            required_skills: list of required skill strings
            candidate_skills: list of candidate skill strings
            threshold: similarity threshold (0-1)
        
        Returns:
            dict: {
                'matched': [...],
                'missing': [...],
                'similarity_scores': {...}
            }
        """
        if not self.model_loaded:
            return self._fallback_matching(required_skills, candidate_skills)
        
        try:
            # Encode skills
            required_embeddings = self.model.encode(required_skills)
            candidate_embeddings = self.model.encode(candidate_skills)
            
            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(required_embeddings, candidate_embeddings)
            
            matched = []
            missing = []
            similarity_scores = {}
            
            for i, req_skill in enumerate(required_skills):
                max_similarity = similarity_matrix[i].max()
                best_match_idx = similarity_matrix[i].argmax()
                
                if max_similarity >= threshold:
                    matched_skill = candidate_skills[best_match_idx]
                    matched.append(req_skill)
                    similarity_scores[req_skill] = {
                        'matched_with': matched_skill,
                        'similarity': float(max_similarity)
                    }
                else:
                    missing.append(req_skill)
                    similarity_scores[req_skill] = {
                        'matched_with': None,
                        'similarity': float(max_similarity)
                    }
            
            return {
                'matched': matched,
                'missing': missing,
                'similarity_scores': similarity_scores
            }
        
        except Exception as e:
            print(f"[WARNING] Semantic matching failed: {e}")
            return self._fallback_matching(required_skills, candidate_skills)
    
    def _fallback_matching(self, required_skills, candidate_skills):
        """Fallback to exact string matching"""
        required_lower = [s.lower() for s in required_skills]
        candidate_lower = [s.lower() for s in candidate_skills]
        
        matched = [s for s in required_skills if s.lower() in candidate_lower]
        missing = [s for s in required_skills if s.lower() not in candidate_lower]
        
        return {
            'matched': matched,
            'missing': missing,
            'similarity_scores': {}
        }
    
    def calculate_similarity(self, resume_text, job_description):
        """
        Calculate semantic similarity between resume and job description
        
        Args:
            resume_text: full resume text
            job_description: job requirements text
        
        Returns:
            float: similarity score (0-1)
        """
        if not self.model_loaded:
            # Fallback to keyword matching
            resume_words = set(resume_text.lower().split())
            job_words = set(job_description.lower().split())
            intersection = len(resume_words.intersection(job_words))
            union = len(resume_words.union(job_words))
            return intersection / union if union > 0 else 0.0
        
        try:
            # Encode both texts
            resume_embedding = self.model.encode([resume_text])
            job_embedding = self.model.encode([job_description])
            
            # Calculate cosine similarity
            similarity = cosine_similarity(resume_embedding, job_embedding)[0][0]
            return float(similarity)
        
        except Exception as e:
            print(f"[WARNING] Similarity calculation failed: {e}")
            # Fallback
            resume_words = set(resume_text.lower().split())
            job_words = set(job_description.lower().split())
            intersection = len(resume_words.intersection(job_words))
            union = len(resume_words.union(job_words))
            return intersection / union if union > 0 else 0.0


# ============================================================================
# 4. RANDOM FOREST - Attrition Prediction
# ============================================================================

class AttritionPredictor:
    """
    Random Forest model to predict candidate attrition risk
    Features: job_hopping, salary_gap, experience, education, skills_match
    """
    
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.model_file = 'attrition_model.pkl'
        
        # Load existing model if available
        self.load_model()
    
    def extract_features(self, resume_data, job_requirements, match_score):
        """
        Extract features for attrition prediction
        
        Returns:
            numpy array of features
        """
        # Feature engineering
        job_hopping_score = self._calculate_job_hopping(resume_data)
        salary_gap = self._calculate_salary_gap(resume_data, job_requirements)
        experience_years = resume_data.get('experience', 0)
        education_level = self._encode_education(resume_data.get('highest_education', 'bachelor'))
        skills_match_pct = match_score
        
        # Additional features
        has_leadership = 1 if resume_data.get('leadership_indicators', []) else 0
        has_achievements = 1 if resume_data.get('achievements', []) else 0
        skills_count = len(resume_data.get('skills', []))
        
        features = np.array([
            job_hopping_score,
            salary_gap,
            experience_years,
            education_level,
            skills_match_pct,
            has_leadership,
            has_achievements,
            skills_count
        ])
        
        return features.reshape(1, -1)
    
    def _calculate_job_hopping(self, resume_data):
        """Calculate job hopping score (0-1, higher = more hopping)"""
        # Simplified: count short-term jobs
        # In real implementation, parse job durations
        return 0.3  # Placeholder
    
    def _calculate_salary_gap(self, resume_data, job_requirements):
        """Calculate salary expectation gap"""
        # Simplified: assume 0.5 (moderate gap)
        return 0.5
    
    def _encode_education(self, education):
        """Encode education level as number"""
        education_map = {
            'high school': 1,
            'diploma': 2,
            'bachelor': 3,
            'master': 4,
            'phd': 5
        }
        return education_map.get(education.lower(), 3)
    
    def predict_attrition_risk(self, resume_data, job_requirements, match_score):
        """
        Predict attrition risk
        
        Returns:
            dict: {
                'risk_score': 0.0-1.0,
                'risk_level': 'low'/'medium'/'high',
                'recommendation': str
            }
        """
        features = self.extract_features(resume_data, job_requirements, match_score)
        
        if not self.is_trained:
            # Use heuristic if model not trained
            return self._heuristic_prediction(features[0])
        
        try:
            # Predict probability of attrition
            features_scaled = self.scaler.transform(features)
            risk_score = self.model.predict_proba(features_scaled)[0][1]
            
            if risk_score < 0.3:
                risk_level = 'low'
                recommendation = "Low attrition risk - Likely to stay long-term"
            elif risk_score < 0.7:
                risk_level = 'medium'
                recommendation = "Moderate risk - Monitor engagement closely"
            else:
                risk_level = 'high'
                recommendation = "High attrition risk - Consider retention strategies"
            
            return {
                'risk_score': float(risk_score),
                'risk_level': risk_level,
                'recommendation': recommendation
            }
        
        except Exception as e:
            print(f"[WARNING] Attrition prediction failed: {e}")
            return self._heuristic_prediction(features[0])
    
    def _heuristic_prediction(self, features):
        """Fallback heuristic prediction"""
        job_hopping, salary_gap, experience, education, skills_match = features[:5]
        
        # Simple heuristic
        risk_score = (job_hopping * 0.4 + salary_gap * 0.3 + (1 - skills_match) * 0.3)
        
        if risk_score < 0.3:
            risk_level = 'low'
        elif risk_score < 0.7:
            risk_level = 'medium'
        else:
            risk_level = 'high'
        
        return {
            'risk_score': float(risk_score),
            'risk_level': risk_level,
            'recommendation': f"Estimated risk based on heuristics"
        }
    
    def train_model(self, X, y):
        """Train the model with labeled data"""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        self.save_model()
    
    def save_model(self):
        """Save trained model"""
        with open(self.model_file, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'is_trained': self.is_trained
            }, f)
    
    def load_model(self):
        """Load trained model"""
        if os.path.exists(self.model_file):
            try:
                with open(self.model_file, 'rb') as f:
                    data = pickle.load(f)  # Fixed: was pickle.dump
                    self.model = data['model']
                    self.scaler = data['scaler']
                    self.is_trained = data['is_trained']
                print(f"[OK] Loaded attrition model from {self.model_file}")
            except Exception as e:
                print(f"[WARNING] Could not load attrition model: {e}")
    
    def predict_score(self, features):
        """
        Predict ML score for candidate
        
        Args:
            features: list of numerical features
        
        Returns:
            float: ML prediction score (0-1)
        """
        try:
            # Convert to numpy array
            features_array = np.array(features).reshape(1, -1)
            
            if self.is_trained:
                # Use trained model
                features_scaled = self.scaler.transform(features_array)
                score = self.model.predict_proba(features_scaled)[0][1]
            else:
                # Use heuristic scoring
                # Normalize features and calculate weighted average
                normalized_features = np.clip(features_array[0], 0, 1)
                score = np.mean(normalized_features)
            
            return float(score)
        
        except Exception as e:
            print(f"[WARNING] ML score prediction failed: {e}")
            # Fallback to simple average
            return float(np.mean(np.clip(features, 0, 1)))


# ============================================================================
# 5. DIVERSITY ANALYTICS - ML-Based Diversity Tracking
# ============================================================================

class DiversityAnalyzer:
    """
    ML-based diversity analytics
    Tracks: education diversity, experience diversity, skill diversity
    """
    
    def analyze_diversity(self, candidates_data):
        """
        Analyze diversity metrics across candidates
        
        Args:
            candidates_data: list of resume_data dicts
        
        Returns:
            dict: diversity metrics
        """
        if not candidates_data:
            return {}
        
        # Education diversity
        education_levels = [c.get('highest_education', 'bachelor') for c in candidates_data]
        education_diversity = len(set(education_levels)) / len(education_levels)
        
        # Experience diversity (variance)
        experience_years = [c.get('experience', 0) for c in candidates_data]
        experience_std = np.std(experience_years) if experience_years else 0
        experience_diversity = min(experience_std / 5.0, 1.0)  # Normalize
        
        # Skill diversity
        all_skills = []
        for c in candidates_data:
            all_skills.extend(c.get('skills', []))
        unique_skills = len(set(all_skills))
        total_skills = len(all_skills)
        skill_diversity = unique_skills / total_skills if total_skills > 0 else 0
        
        # University diversity
        universities = []
        for c in candidates_data:
            # Extract university from education field (simplified)
            universities.append(c.get('university', 'Unknown'))
        university_diversity = len(set(universities)) / len(universities)
        
        return {
            'education_diversity': float(education_diversity),
            'experience_diversity': float(experience_diversity),
            'skill_diversity': float(skill_diversity),
            'university_diversity': float(university_diversity),
            'overall_diversity_score': float(np.mean([
                education_diversity,
                experience_diversity,
                skill_diversity,
                university_diversity
            ])),
            'total_candidates': len(candidates_data),
            'unique_skills': unique_skills,
            'unique_universities': len(set(universities))
        }


# ============================================================================
# Initialize Global Models
# ============================================================================

# Singleton instances
rl_scorer = ReinforcementLearningScorer()
bert_parser = BERTResumeParser()
semantic_matcher = SemanticSkillMatcher()
attrition_predictor = AttritionPredictor()
diversity_analyzer = DiversityAnalyzer()


# ============================================================================
# Utility Functions
# ============================================================================

def get_ml_models_status():
    """Get status of all ML models"""
    return {
        'reinforcement_learning': {
            'loaded': True,
            'feedback_count': len(rl_scorer.feedback_history),
            'accuracy': rl_scorer.get_stats().get('accuracy', 0.0)
        },
        'bert_ner': {
            'loaded': bert_parser.model_loaded,
            'model': 'dslim/bert-base-NER' if bert_parser.model_loaded else 'Not loaded'
        },
        'semantic_matching': {
            'loaded': semantic_matcher.model_loaded,
            'model': 'all-MiniLM-L6-v2' if semantic_matcher.model_loaded else 'Not loaded'
        },
        'attrition_prediction': {
            'loaded': True,
            'trained': attrition_predictor.is_trained,
            'model': 'Random Forest (100 trees)'
        },
        'diversity_analytics': {
            'loaded': True,
            'model': 'Statistical Analysis'
        }
    }


if __name__ == "__main__":
    # Test models
    print("=" * 60)
    print("ML/DL Models Status")
    print("=" * 60)
    
    status = get_ml_models_status()
    for model_name, model_status in status.items():
        print(f"\n{model_name.upper()}:")
        for key, value in model_status.items():
            print(f"  {key}: {value}")
