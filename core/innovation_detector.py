"""
NeuroMatch AI - Innovation Detection Engine
Novelty detection algorithms and project complexity analysis
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
import re
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class InnovationMetrics:
    """Innovation assessment metrics"""
    novelty_score: float
    complexity_score: float
    impact_potential: float
    technical_depth: float
    creativity_index: float
    market_relevance: float

class InnovationDetector:
    """Main innovation detection and analysis engine"""
    
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.is_fitted = False
        
        # Innovation keywords by category
        self.innovation_keywords = {
            'technical': ['algorithm', 'ai', 'ml', 'blockchain', 'quantum', 'neural', 'deep learning'],
            'process': ['automation', 'optimization', 'efficiency', 'streamline', 'workflow'],
            'product': ['prototype', 'mvp', 'patent', 'invention', 'design', 'user experience'],
            'research': ['research', 'experiment', 'hypothesis', 'analysis', 'discovery'],
            'creative': ['creative', 'innovative', 'novel', 'unique', 'original', 'breakthrough']
        }
        
    def extract_innovation_features(self, text: str, project_data: Optional[Dict] = None) -> np.ndarray:
        """Extract innovation-related features from text and project data"""
        features = []
        
        # Text-based innovation indicators
        text_lower = text.lower()
        
        # Innovation keyword density by category
        for category, keywords in self.innovation_keywords.items():
            density = sum(1 for keyword in keywords if keyword in text_lower) / len(keywords)
            features.append(density)
        
        # Technical complexity indicators
        technical_terms = ['system', 'architecture', 'framework', 'platform', 'infrastructure']
        tech_complexity = sum(1 for term in technical_terms if term in text_lower) / len(technical_terms)
        features.append(tech_complexity)
        
        # Novelty indicators
        novelty_terms = ['first', 'new', 'novel', 'innovative', 'pioneering', 'groundbreaking']
        novelty_score = sum(1 for term in novelty_terms if term in text_lower) / len(novelty_terms)
        features.append(novelty_score)
        
        # Impact indicators
        impact_terms = ['impact', 'transform', 'revolutionize', 'improve', 'enhance', 'solve']
        impact_score = sum(1 for term in impact_terms if term in text_lower) / len(impact_terms)
        features.append(impact_score)
        
        # Project-specific features
        if project_data:
            features.extend(self._extract_project_features(project_data))
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])  # Default project features
        
        return np.array(features)
    
    def _extract_project_features(self, project_data: Dict) -> List[float]:
        """Extract features from project metadata"""
        features = []
        
        # Project duration complexity
        duration = project_data.get('duration_months', 6)
        duration_complexity = min(duration / 24, 1.0)  # Normalize to 2 years max
        features.append(duration_complexity)
        
        # Team size complexity
        team_size = project_data.get('team_size', 1)
        team_complexity = min(team_size / 20, 1.0)  # Normalize to 20 people max
        features.append(team_complexity)
        
        # Technology stack diversity
        technologies = project_data.get('technologies', [])
        tech_diversity = min(len(technologies) / 10, 1.0)  # Normalize to 10 techs max
        features.append(tech_diversity)
        
        # Budget/scale indicator
        budget = project_data.get('budget', 0)
        budget_scale = min(budget / 1000000, 1.0) if budget > 0 else 0.0  # Normalize to 1M max
        features.append(budget_scale)
        
        return features
    
    def calculate_innovation_score(self, resume_text: str, project_history: Optional[List[Dict]] = None) -> InnovationMetrics:
        """Calculate comprehensive innovation metrics"""
        
        # Extract base features
        features = self.extract_innovation_features(resume_text)
        
        # Calculate individual metrics
        novelty_score = self._calculate_novelty_score(resume_text, project_history)
        complexity_score = self._calculate_complexity_score(resume_text, project_history)
        impact_potential = self._calculate_impact_potential(resume_text)
        technical_depth = self._calculate_technical_depth(resume_text)
        creativity_index = self._calculate_creativity_index(resume_text)
        market_relevance = self._calculate_market_relevance(resume_text)
        
        return InnovationMetrics(
            novelty_score=novelty_score,
            complexity_score=complexity_score,
            impact_potential=impact_potential,
            technical_depth=technical_depth,
            creativity_index=creativity_index,
            market_relevance=market_relevance
        )
    
    def _calculate_novelty_score(self, text: str, project_history: Optional[List[Dict]] = None) -> float:
        """Calculate novelty/uniqueness score"""
        text_lower = text.lower()
        
        # Base novelty from keywords
        novelty_keywords = ['first', 'novel', 'unique', 'original', 'pioneering', 'breakthrough', 'innovative']
        base_novelty = sum(1 for keyword in novelty_keywords if keyword in text_lower) / len(novelty_keywords)
        
        # Project novelty assessment
        project_novelty = 0.0
        if project_history:
            for project in project_history:
                project_text = project.get('description', '').lower()
                if any(keyword in project_text for keyword in novelty_keywords):
                    project_novelty += 0.2
        
        project_novelty = min(project_novelty, 1.0)
        
        # Combined novelty score
        return (base_novelty * 0.6 + project_novelty * 0.4)
    
    def _calculate_complexity_score(self, text: str, project_history: Optional[List[Dict]] = None) -> float:
        """Calculate technical and project complexity"""
        text_lower = text.lower()
        
        # Technical complexity indicators
        complex_terms = [
            'architecture', 'system design', 'scalability', 'distributed', 'microservices',
            'algorithm', 'optimization', 'machine learning', 'ai', 'data pipeline'
        ]
        tech_complexity = sum(1 for term in complex_terms if term in text_lower) / len(complex_terms)
        
        # Project complexity from history
        project_complexity = 0.0
        if project_history:
            for project in project_history:
                # Complexity factors
                duration = project.get('duration_months', 0)
                team_size = project.get('team_size', 1)
                tech_count = len(project.get('technologies', []))
                
                # Normalize and combine
                complexity_factor = (
                    min(duration / 24, 1.0) * 0.4 +
                    min(team_size / 15, 1.0) * 0.3 +
                    min(tech_count / 8, 1.0) * 0.3
                )
                project_complexity += complexity_factor * 0.2
        
        project_complexity = min(project_complexity, 1.0)
        
        return (tech_complexity * 0.7 + project_complexity * 0.3)
    
    def _calculate_impact_potential(self, text: str) -> float:
        """Calculate potential impact score"""
        text_lower = text.lower()
        
        # Impact keywords
        impact_keywords = [
            'impact', 'transform', 'improve', 'solve', 'benefit', 'value',
            'efficiency', 'cost saving', 'revenue', 'user experience', 'performance'
        ]
        
        impact_score = sum(1 for keyword in impact_keywords if keyword in text_lower) / len(impact_keywords)
        
        # Scale indicators
        scale_keywords = ['enterprise', 'global', 'million', 'thousand', 'large scale', 'widespread']
        scale_score = sum(1 for keyword in scale_keywords if keyword in text_lower) / len(scale_keywords)
        
        return (impact_score * 0.7 + scale_score * 0.3)
    
    def _calculate_technical_depth(self, text: str) -> float:
        """Calculate technical depth and sophistication"""
        text_lower = text.lower()
        
        # Advanced technical terms
        advanced_terms = [
            'neural network', 'deep learning', 'reinforcement learning', 'computer vision',
            'natural language processing', 'blockchain', 'quantum computing', 'edge computing',
            'kubernetes', 'docker', 'tensorflow', 'pytorch', 'distributed systems'
        ]
        
        advanced_score = sum(1 for term in advanced_terms if term in text_lower) / len(advanced_terms)
        
        # Programming and tools depth
        tech_tools = [
            'python', 'java', 'c++', 'rust', 'go', 'scala', 'r',
            'aws', 'gcp', 'azure', 'spark', 'hadoop', 'kafka'
        ]
        
        tools_score = sum(1 for tool in tech_tools if tool in text_lower) / len(tech_tools)
        
        return (advanced_score * 0.6 + tools_score * 0.4)
    
    def _calculate_creativity_index(self, text: str) -> float:
        """Calculate creativity and out-of-box thinking indicators"""
        text_lower = text.lower()
        
        # Creative process indicators
        creative_terms = [
            'creative', 'design thinking', 'brainstorm', 'ideation', 'prototype',
            'experiment', 'explore', 'artistic', 'visual', 'user-centered'
        ]
        
        creative_score = sum(1 for term in creative_terms if term in text_lower) / len(creative_terms)
        
        # Problem-solving approach indicators
        problem_solving = [
            'solution', 'approach', 'methodology', 'framework', 'strategy',
            'alternative', 'innovative approach', 'creative solution'
        ]
        
        solving_score = sum(1 for term in problem_solving if term in text_lower) / len(problem_solving)
        
        return (creative_score * 0.6 + solving_score * 0.4)
    
    def _calculate_market_relevance(self, text: str) -> float:
        """Calculate market relevance and commercial viability"""
        text_lower = text.lower()
        
        # Market-oriented terms
        market_terms = [
            'market', 'customer', 'user', 'business', 'commercial', 'product',
            'startup', 'entrepreneur', 'revenue', 'roi', 'value proposition'
        ]
        
        market_score = sum(1 for term in market_terms if term in text_lower) / len(market_terms)
        
        # Industry relevance
        industry_terms = [
            'fintech', 'healthtech', 'edtech', 'saas', 'e-commerce', 'mobile',
            'web', 'cloud', 'iot', 'ar', 'vr', 'gaming', 'social media'
        ]
        
        industry_score = sum(1 for term in industry_terms if term in text_lower) / len(industry_terms)
        
        return (market_score * 0.7 + industry_score * 0.3)
    
    def detect_innovation_patterns(self, resume_texts: List[str]) -> List[Dict[str, float]]:
        """Detect innovation patterns across multiple resumes"""
        if not self.is_fitted:
            self._fit_detectors(resume_texts)
        
        results = []
        for text in resume_texts:
            features = self.extract_innovation_features(text)
            
            # Anomaly detection for innovation
            anomaly_score = self.isolation_forest.decision_function([features])[0]
            is_innovative = self.isolation_forest.predict([features])[0] == -1
            
            # Calculate innovation metrics
            metrics = self.calculate_innovation_score(text)
            
            result = {
                'is_innovative_outlier': is_innovative,
                'anomaly_score': float(anomaly_score),
                'novelty_score': metrics.novelty_score,
                'complexity_score': metrics.complexity_score,
                'impact_potential': metrics.impact_potential,
                'technical_depth': metrics.technical_depth,
                'creativity_index': metrics.creativity_index,
                'market_relevance': metrics.market_relevance,
                'overall_innovation_score': self._calculate_overall_score(metrics)
            }
            
            results.append(result)
        
        return results
    
    def _fit_detectors(self, resume_texts: List[str]):
        """Fit anomaly detection models on resume corpus"""
        features_list = []
        for text in resume_texts:
            features = self.extract_innovation_features(text)
            features_list.append(features)
        
        X = np.array(features_list)
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit isolation forest for anomaly detection
        self.isolation_forest.fit(X_scaled)
        self.is_fitted = True
        
        logger.info("Innovation detection models fitted successfully")
    
    def _calculate_overall_score(self, metrics: InnovationMetrics) -> float:
        """Calculate weighted overall innovation score"""
        weights = {
            'novelty': 0.25,
            'complexity': 0.20,
            'impact': 0.20,
            'technical': 0.15,
            'creativity': 0.15,
            'market': 0.05
        }
        
        overall_score = (
            metrics.novelty_score * weights['novelty'] +
            metrics.complexity_score * weights['complexity'] +
            metrics.impact_potential * weights['impact'] +
            metrics.technical_depth * weights['technical'] +
            metrics.creativity_index * weights['creativity'] +
            metrics.market_relevance * weights['market']
        )
        
        return min(overall_score, 1.0)
    
    def generate_innovation_insights(self, innovation_results: Dict[str, float]) -> Dict[str, str]:
        """Generate human-readable insights from innovation analysis"""
        insights = {}
        
        overall_score = innovation_results['overall_innovation_score']
        
        # Overall assessment
        if overall_score >= 0.8:
            insights['overall'] = "Exceptional innovator with groundbreaking potential"
        elif overall_score >= 0.6:
            insights['overall'] = "Strong innovation capabilities with significant potential"
        elif overall_score >= 0.4:
            insights['overall'] = "Moderate innovation skills with room for growth"
        else:
            insights['overall'] = "Limited innovation indicators, focus on creative development"
        
        # Specific strengths
        strengths = []
        if innovation_results['technical_depth'] >= 0.7:
            strengths.append("Advanced technical expertise")
        if innovation_results['creativity_index'] >= 0.7:
            strengths.append("Strong creative problem-solving")
        if innovation_results['complexity_score'] >= 0.7:
            strengths.append("Handles complex projects effectively")
        if innovation_results['impact_potential'] >= 0.7:
            strengths.append("High potential for meaningful impact")
        
        insights['strengths'] = ", ".join(strengths) if strengths else "Developing innovation capabilities"
        
        # Development areas
        development = []
        if innovation_results['novelty_score'] < 0.5:
            development.append("Explore more novel approaches")
        if innovation_results['market_relevance'] < 0.5:
            development.append("Increase market awareness")
        if innovation_results['creativity_index'] < 0.5:
            development.append("Enhance creative thinking")
        
        insights['development'] = ", ".join(development) if development else "Continue building on strong foundation"
        
        return insights

# Example usage
if __name__ == "__main__":
    detector = InnovationDetector()
    
    sample_text = """
    Led development of novel machine learning algorithm for real-time fraud detection.
    Created innovative blockchain-based solution that reduced transaction costs by 40%.
    Pioneered use of computer vision in autonomous vehicle navigation system.
    Built scalable microservices architecture serving millions of users globally.
    """
    
    metrics = detector.calculate_innovation_score(sample_text)
    print(f"Innovation Metrics:")
    print(f"  Novelty: {metrics.novelty_score:.3f}")
    print(f"  Complexity: {metrics.complexity_score:.3f}")
    print(f"  Impact: {metrics.impact_potential:.3f}")
    print(f"  Technical Depth: {metrics.technical_depth:.3f}")
    print(f"  Creativity: {metrics.creativity_index:.3f}")
    print(f"  Market Relevance: {metrics.market_relevance:.3f}")
