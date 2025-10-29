"""
NeuroMatch AI - Ensemble Matching AI with Explainable AI
Multi-model fusion system with SHAP/LIME integration
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
import shap
import lime
import lime.lime_tabular
import joblib
import logging
from typing import Dict, List, Tuple, Optional, Any
import os
from datetime import datetime
from dataclasses import dataclass

# Import our custom modules
from .cognitive_analyzer import CognitiveAnalyzer
from .growth_predictor import GrowthPredictor
from .innovation_detector import InnovationDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MatchingResult:
    """Comprehensive matching result with explanations"""
    overall_score: float
    cognitive_compatibility: float
    growth_alignment: float
    innovation_match: float
    technical_fit: float
    cultural_fit: float
    explanation: Dict[str, Any]
    confidence: float
    recommendations: List[str]

class EnsembleMatcher:
    """Advanced ensemble matching system with explainable AI"""
    
    def __init__(self, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize component analyzers
        self.cognitive_analyzer = CognitiveAnalyzer(device=self.device)
        self.growth_predictor = GrowthPredictor(device=self.device)
        self.innovation_detector = InnovationDetector()
        
        # Ensemble models
        self.models = {}
        self.model_weights = {}
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Explainability components
        self.shap_explainer = None
        self.lime_explainer = None
        
        logger.info("Initialized EnsembleMatcher with explainable AI components")
    
    def initialize_models(self):
        """Initialize ensemble of ML models"""
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42
            ),
            'svm': SVC(
                C=1.0,
                kernel='rbf',
                probability=True,
                random_state=42
            )
        }
        
        # Initial equal weights
        self.model_weights = {name: 1.0 for name in self.models.keys()}
    
    def extract_comprehensive_features(self, resume_data: Dict, 
                                     job_requirements: Dict) -> np.ndarray:
        """Extract comprehensive features for matching"""
        features = []
        
        # Basic resume features
        experience_years = float(resume_data.get('experience', 0))
        skills_count = len(resume_data.get('skills', []))
        education_level = self._encode_education_level(resume_data.get('education', []))
        
        features.extend([experience_years, skills_count, education_level])
        
        # Cognitive pattern features
        resume_text = self._create_resume_text(resume_data)
        cognitive_scores = self.cognitive_analyzer.predict_cognitive_pattern(resume_text)
        features.extend(list(cognitive_scores.values()))
        
        # Growth prediction features
        growth_analysis = self.growth_predictor.predict_growth_potential(resume_data)
        features.extend([
            growth_analysis['overall_growth_potential'],
            growth_analysis['technical_growth'],
            growth_analysis['leadership_growth'],
            growth_analysis['innovation_potential']
        ])
        
        # Innovation features
        innovation_metrics = self.innovation_detector.calculate_innovation_score(resume_text)
        features.extend([
            innovation_metrics.novelty_score,
            innovation_metrics.complexity_score,
            innovation_metrics.impact_potential,
            innovation_metrics.technical_depth
        ])
        
        # Job matching features
        job_match_features = self._calculate_job_match_features(resume_data, job_requirements)
        features.extend(job_match_features)
        
        return np.array(features, dtype=np.float32)
    
    def _create_resume_text(self, resume_data: Dict) -> str:
        """Create comprehensive text from resume data"""
        text_parts = []
        
        if resume_data.get('name'):
            text_parts.append(resume_data['name'])
        
        if resume_data.get('skills'):
            text_parts.append(' '.join(resume_data['skills']))
        
        if resume_data.get('education'):
            text_parts.append(' '.join(resume_data['education']))
        
        if resume_data.get('current_role'):
            text_parts.append(resume_data['current_role'])
        
        return ' '.join(text_parts)
    
    def _encode_education_level(self, education_list: List[str]) -> float:
        """Encode education level numerically"""
        education_text = ' '.join(education_list).lower()
        
        if any(degree in education_text for degree in ['phd', 'doctorate']):
            return 4.0
        elif any(degree in education_text for degree in ['master', 'mba']):
            return 3.0
        elif any(degree in education_text for degree in ['bachelor', 'b.tech']):
            return 2.0
        else:
            return 1.0
    
    def _calculate_job_match_features(self, resume_data: Dict, 
                                    job_requirements: Dict) -> List[float]:
        """Calculate job-specific matching features"""
        features = []
        
        # Required skills match
        required_skills = job_requirements.get('required_skills', [])
        candidate_skills = [skill.lower() for skill in resume_data.get('skills', [])]
        
        if required_skills:
            skills_match_ratio = sum(1 for skill in required_skills 
                                   if skill.lower() in candidate_skills) / len(required_skills)
        else:
            skills_match_ratio = 0.0
        
        features.append(skills_match_ratio)
        
        # Experience match
        required_experience = job_requirements.get('min_experience', 0)
        candidate_experience = float(resume_data.get('experience', 0))
        experience_match = min(candidate_experience / max(required_experience, 1), 2.0)  # Cap at 2x
        features.append(experience_match)
        
        # Education match
        required_education = job_requirements.get('education_level', 'bachelor')
        candidate_education_level = self._encode_education_level(resume_data.get('education', []))
        
        education_requirements = {'high_school': 1, 'bachelor': 2, 'master': 3, 'phd': 4}
        required_level = education_requirements.get(required_education.lower(), 2)
        education_match = min(candidate_education_level / required_level, 2.0)
        features.append(education_match)
        
        # Industry/domain match
        job_domain = job_requirements.get('domain', '').lower()
        resume_text = self._create_resume_text(resume_data).lower()
        domain_match = 1.0 if job_domain in resume_text else 0.0
        features.append(domain_match)
        
        # Role level match
        required_level = job_requirements.get('role_level', 'mid')
        candidate_experience_years = float(resume_data.get('experience', 0))
        
        level_mapping = {'junior': 2, 'mid': 5, 'senior': 8, 'lead': 12}
        required_years = level_mapping.get(required_level, 5)
        level_match = min(candidate_experience_years / required_years, 2.0)
        features.append(level_match)
        
        return features
    
    def train_ensemble(self, training_data: List[Tuple[Dict, Dict, float]], 
                      validation_split: float = 0.2):
        """Train ensemble of models with hyperparameter optimization"""
        logger.info("Starting ensemble model training...")
        
        # Prepare training data
        X_features = []
        y_targets = []
        
        for resume_data, job_requirements, match_score in training_data:
            features = self.extract_comprehensive_features(resume_data, job_requirements)
            X_features.append(features)
            y_targets.append(1 if match_score >= 0.7 else 0)  # Binary classification
        
        X = np.array(X_features)
        y = np.array(y_targets)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize models
        self.initialize_models()
        
        # Train each model with hyperparameter tuning
        trained_models = {}
        model_scores = {}
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            
            # Define parameter grids for each model
            param_grids = self._get_param_grids()
            
            if name in param_grids:
                # Grid search for optimal parameters
                grid_search = GridSearchCV(
                    model, param_grids[name], 
                    cv=5, scoring='roc_auc', n_jobs=-1
                )
                grid_search.fit(X_scaled, y)
                trained_models[name] = grid_search.best_estimator_
                
                # Cross-validation score
                cv_scores = cross_val_score(
                    grid_search.best_estimator_, X_scaled, y, 
                    cv=5, scoring='roc_auc'
                )
                model_scores[name] = np.mean(cv_scores)
                
                logger.info(f"{name} - Best params: {grid_search.best_params_}")
                logger.info(f"{name} - CV Score: {model_scores[name]:.4f}")
            else:
                # Train with default parameters
                model.fit(X_scaled, y)
                trained_models[name] = model
                cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='roc_auc')
                model_scores[name] = np.mean(cv_scores)
        
        self.models = trained_models
        
        # Calculate optimal weights based on performance
        self._calculate_optimal_weights(model_scores)
        
        # Initialize explainability components
        self._initialize_explainers(X_scaled, y)
        
        self.is_trained = True
        logger.info("Ensemble training completed successfully!")
    
    def _get_param_grids(self) -> Dict[str, Dict]:
        """Get parameter grids for hyperparameter tuning"""
        return {
            'random_forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10]
            },
            'gradient_boosting': {
                'n_estimators': [100, 150, 200],
                'learning_rate': [0.05, 0.1, 0.15],
                'max_depth': [3, 5, 7]
            },
            'logistic_regression': {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        }
    
    def _calculate_optimal_weights(self, model_scores: Dict[str, float]):
        """Calculate optimal ensemble weights based on model performance"""
        # Normalize scores to weights
        total_score = sum(model_scores.values())
        if total_score > 0:
            self.model_weights = {
                name: score / total_score 
                for name, score in model_scores.items()
            }
        else:
            # Equal weights if all scores are zero
            num_models = len(model_scores)
            self.model_weights = {name: 1.0 / num_models for name in model_scores.keys()}
        
        logger.info(f"Optimal model weights: {self.model_weights}")
    
    def _initialize_explainers(self, X_scaled: np.ndarray, y: np.ndarray):
        """Initialize SHAP and LIME explainers"""
        try:
            # Use the best performing model for explanations
            best_model_name = max(self.model_weights, key=self.model_weights.get)
            best_model = self.models[best_model_name]
            
            # Initialize SHAP explainer
            if hasattr(best_model, 'predict_proba'):
                self.shap_explainer = shap.Explainer(best_model, X_scaled[:100])  # Sample for efficiency
            
            # Initialize LIME explainer
            feature_names = self._get_feature_names()
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                X_scaled,
                feature_names=feature_names,
                class_names=['No Match', 'Match'],
                mode='classification'
            )
            
            logger.info("Explainability components initialized successfully")
            
        except Exception as e:
            logger.warning(f"Failed to initialize explainers: {e}")
            self.shap_explainer = None
            self.lime_explainer = None
    
    def _get_feature_names(self) -> List[str]:
        """Get feature names for explainability"""
        base_features = ['experience_years', 'skills_count', 'education_level']
        
        # Cognitive pattern features
        cognitive_features = [
            'analytical_thinker', 'creative_innovator', 'strategic_planner',
            'collaborative_leader', 'detail_perfectionist', 'adaptive_solver',
            'results_executor', 'empathetic_communicator'
        ]
        
        # Growth features
        growth_features = [
            'overall_growth', 'technical_growth', 'leadership_growth', 'innovation_growth'
        ]
        
        # Innovation features
        innovation_features = [
            'novelty_score', 'complexity_score', 'impact_potential', 'technical_depth'
        ]
        
        # Job matching features
        job_features = [
            'skills_match_ratio', 'experience_match', 'education_match', 
            'domain_match', 'level_match'
        ]
        
        return base_features + cognitive_features + growth_features + innovation_features + job_features
    
    def predict_match(self, resume_data: Dict, job_requirements: Dict) -> MatchingResult:
        """Predict comprehensive matching score with explanations"""
        if not self.is_trained:
            return self._rule_based_matching(resume_data, job_requirements)
        
        # Extract features
        features = self.extract_comprehensive_features(resume_data, job_requirements)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Ensemble prediction
        ensemble_proba = 0.0
        model_predictions = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(features_scaled)[0][1]  # Probability of match
            else:
                proba = model.decision_function(features_scaled)[0]
                proba = 1 / (1 + np.exp(-proba))  # Sigmoid transformation
            
            model_predictions[name] = proba
            ensemble_proba += proba * self.model_weights[name]
        
        # Generate detailed analysis
        detailed_scores = self._calculate_detailed_scores(resume_data, job_requirements)
        
        # Generate explanations
        explanations = self._generate_explanations(features_scaled, detailed_scores)
        
        # Calculate confidence
        prediction_variance = np.var(list(model_predictions.values()))
        confidence = max(0.1, 1.0 - prediction_variance)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(detailed_scores, job_requirements)
        
        return MatchingResult(
            overall_score=float(ensemble_proba),
            cognitive_compatibility=detailed_scores['cognitive_compatibility'],
            growth_alignment=detailed_scores['growth_alignment'],
            innovation_match=detailed_scores['innovation_match'],
            technical_fit=detailed_scores['technical_fit'],
            cultural_fit=detailed_scores['cultural_fit'],
            explanation=explanations,
            confidence=float(confidence),
            recommendations=recommendations
        )
    
    def _calculate_detailed_scores(self, resume_data: Dict, job_requirements: Dict) -> Dict[str, float]:
        """Calculate detailed matching scores for different aspects"""
        resume_text = self._create_resume_text(resume_data)
        
        # Cognitive compatibility
        cognitive_scores = self.cognitive_analyzer.predict_cognitive_pattern(resume_text)
        job_cognitive_requirements = job_requirements.get('cognitive_requirements', {})
        
        if job_cognitive_requirements:
            cognitive_compatibility = sum(
                cognitive_scores.get(pattern, 0) * weight
                for pattern, weight in job_cognitive_requirements.items()
            ) / sum(job_cognitive_requirements.values())
        else:
            cognitive_compatibility = 0.7  # Default moderate compatibility
        
        # Growth alignment
        growth_analysis = self.growth_predictor.predict_growth_potential(resume_data)
        growth_alignment = growth_analysis['overall_growth_potential']
        
        # Innovation match
        innovation_metrics = self.innovation_detector.calculate_innovation_score(resume_text)
        innovation_requirements = job_requirements.get('innovation_level', 0.5)
        innovation_match = min(innovation_metrics.novelty_score / innovation_requirements, 1.0)
        
        # Technical fit
        required_skills = job_requirements.get('required_skills', [])
        candidate_skills = [skill.lower() for skill in resume_data.get('skills', [])]
        
        if required_skills:
            technical_fit = sum(1 for skill in required_skills 
                              if skill.lower() in candidate_skills) / len(required_skills)
        else:
            technical_fit = 0.5
        
        # Cultural fit (based on cognitive patterns and soft skills)
        cultural_indicators = ['communication', 'teamwork', 'leadership', 'collaboration']
        cultural_score = sum(1 for indicator in cultural_indicators 
                           if indicator in resume_text.lower()) / len(cultural_indicators)
        cultural_fit = min(cultural_score + cognitive_compatibility * 0.3, 1.0)
        
        return {
            'cognitive_compatibility': cognitive_compatibility,
            'growth_alignment': growth_alignment,
            'innovation_match': innovation_match,
            'technical_fit': technical_fit,
            'cultural_fit': cultural_fit
        }
    
    def _generate_explanations(self, features_scaled: np.ndarray, 
                             detailed_scores: Dict[str, float]) -> Dict[str, Any]:
        """Generate comprehensive explanations using SHAP and LIME"""
        explanations = {
            'detailed_scores': detailed_scores,
            'feature_importance': {},
            'shap_values': None,
            'lime_explanation': None,
            'key_factors': []
        }
        
        try:
            # SHAP explanations
            if self.shap_explainer is not None:
                shap_values = self.shap_explainer(features_scaled)
                explanations['shap_values'] = shap_values.values[0].tolist()
                
                # Top contributing features
                feature_names = self._get_feature_names()
                feature_importance = dict(zip(feature_names, shap_values.values[0]))
                explanations['feature_importance'] = feature_importance
                
                # Key positive and negative factors
                sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
                explanations['key_factors'] = sorted_features[:5]
            
            # LIME explanations
            if self.lime_explainer is not None:
                best_model_name = max(self.model_weights, key=self.model_weights.get)
                best_model = self.models[best_model_name]
                
                lime_exp = self.lime_explainer.explain_instance(
                    features_scaled[0], 
                    best_model.predict_proba,
                    num_features=10
                )
                explanations['lime_explanation'] = lime_exp.as_list()
        
        except Exception as e:
            logger.warning(f"Failed to generate explanations: {e}")
        
        return explanations
    
    def _generate_recommendations(self, detailed_scores: Dict[str, float], 
                                job_requirements: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Technical fit recommendations
        if detailed_scores['technical_fit'] < 0.7:
            missing_skills = job_requirements.get('required_skills', [])
            recommendations.append(
                f"Consider developing skills in: {', '.join(missing_skills[:3])}"
            )
        
        # Growth alignment recommendations
        if detailed_scores['growth_alignment'] < 0.6:
            recommendations.append(
                "Focus on demonstrating leadership experience and continuous learning"
            )
        
        # Innovation match recommendations
        if detailed_scores['innovation_match'] < 0.5:
            recommendations.append(
                "Highlight innovative projects and creative problem-solving experiences"
            )
        
        # Cultural fit recommendations
        if detailed_scores['cultural_fit'] < 0.6:
            recommendations.append(
                "Emphasize teamwork, communication, and collaborative achievements"
            )
        
        # Overall recommendations
        if detailed_scores['cognitive_compatibility'] < 0.6:
            recommendations.append(
                "Consider roles that better align with your cognitive strengths"
            )
        
        return recommendations[:3]  # Limit to top 3 recommendations
    
    def _rule_based_matching(self, resume_data: Dict, job_requirements: Dict) -> MatchingResult:
        """Fallback rule-based matching when models aren't trained"""
        detailed_scores = self._calculate_detailed_scores(resume_data, job_requirements)
        
        # Simple weighted average
        overall_score = (
            detailed_scores['technical_fit'] * 0.3 +
            detailed_scores['cognitive_compatibility'] * 0.2 +
            detailed_scores['growth_alignment'] * 0.2 +
            detailed_scores['innovation_match'] * 0.15 +
            detailed_scores['cultural_fit'] * 0.15
        )
        
        recommendations = self._generate_recommendations(detailed_scores, job_requirements)
        
        return MatchingResult(
            overall_score=overall_score,
            cognitive_compatibility=detailed_scores['cognitive_compatibility'],
            growth_alignment=detailed_scores['growth_alignment'],
            innovation_match=detailed_scores['innovation_match'],
            technical_fit=detailed_scores['technical_fit'],
            cultural_fit=detailed_scores['cultural_fit'],
            explanation={'detailed_scores': detailed_scores, 'method': 'rule_based'},
            confidence=0.7,
            recommendations=recommendations
        )
    
    def save_ensemble(self, model_path: str):
        """Save trained ensemble and all components"""
        if not self.is_trained:
            raise ValueError("No trained ensemble to save")
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save ensemble components
        ensemble_data = {
            'models': self.models,
            'model_weights': self.model_weights,
            'scaler': self.scaler,
            'is_trained': self.is_trained,
            'feature_names': self._get_feature_names(),
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(ensemble_data, model_path)
        logger.info(f"Ensemble saved to {model_path}")
    
    def load_ensemble(self, model_path: str):
        """Load trained ensemble"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Ensemble file not found: {model_path}")
        
        ensemble_data = joblib.load(model_path)
        
        self.models = ensemble_data['models']
        self.model_weights = ensemble_data['model_weights']
        self.scaler = ensemble_data['scaler']
        self.is_trained = ensemble_data['is_trained']
        
        logger.info(f"Ensemble loaded from {model_path}")

# Example usage
if __name__ == "__main__":
    matcher = EnsembleMatcher()
    
    # Example data
    resume_data = {
        'name': 'John Doe',
        'experience': '5',
        'skills': ['python', 'machine learning', 'sql', 'aws'],
        'education': ['Master of Science in Computer Science'],
        'current_role': 'Senior Data Scientist'
    }
    
    job_requirements = {
        'required_skills': ['python', 'machine learning', 'sql'],
        'min_experience': 3,
        'education_level': 'bachelor',
        'role_level': 'senior',
        'cognitive_requirements': {
            'Analytical Thinker': 0.8,
            'Strategic Planner': 0.6
        },
        'innovation_level': 0.7
    }
    
    # Predict match
    result = matcher.predict_match(resume_data, job_requirements)
    
    print(f"Overall Match Score: {result.overall_score:.3f}")
    print(f"Technical Fit: {result.technical_fit:.3f}")
    print(f"Cognitive Compatibility: {result.cognitive_compatibility:.3f}")
    print(f"Growth Alignment: {result.growth_alignment:.3f}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Recommendations: {result.recommendations}")
