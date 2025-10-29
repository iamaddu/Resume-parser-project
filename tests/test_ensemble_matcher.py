"""
Test suite for Ensemble Matcher
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ensemble_matcher import EnsembleMatcher, MatchingResult

class TestEnsembleMatcher:
    
    @pytest.fixture
    def matcher(self):
        """Create matcher instance for testing"""
        return EnsembleMatcher()
    
    @pytest.fixture
    def sample_resume_data(self):
        """Sample resume data for testing"""
        return {
            'name': 'John Doe',
            'experience': '5',
            'skills': ['python', 'machine learning', 'sql', 'aws'],
            'education': ['Master of Science in Computer Science'],
            'current_role': 'Senior Data Scientist'
        }
    
    @pytest.fixture
    def sample_job_requirements(self):
        """Sample job requirements for testing"""
        return {
            'role': 'Senior Data Scientist',
            'required_skills': ['python', 'machine learning', 'sql'],
            'min_experience': 3,
            'education_level': 'master',
            'role_level': 'senior',
            'cognitive_requirements': {
                'Analytical Thinker': 0.8,
                'Strategic Planner': 0.6
            },
            'innovation_level': 0.7,
            'domain': 'fintech'
        }
    
    def test_matcher_initialization(self, matcher):
        """Test matcher initialization"""
        assert matcher is not None
        assert hasattr(matcher, 'cognitive_analyzer')
        assert hasattr(matcher, 'growth_predictor')
        assert hasattr(matcher, 'innovation_detector')
        assert matcher.is_trained == False
    
    def test_initialize_models(self, matcher):
        """Test model initialization"""
        matcher.initialize_models()
        
        assert 'random_forest' in matcher.models
        assert 'gradient_boosting' in matcher.models
        assert 'logistic_regression' in matcher.models
        assert 'svm' in matcher.models
        
        # Check model weights
        assert len(matcher.model_weights) == 4
        for weight in matcher.model_weights.values():
            assert weight == 1.0  # Initial equal weights
    
    def test_extract_comprehensive_features(self, matcher, sample_resume_data, sample_job_requirements):
        """Test comprehensive feature extraction"""
        features = matcher.extract_comprehensive_features(sample_resume_data, sample_job_requirements)
        
        assert isinstance(features, np.ndarray)
        assert len(features) > 10  # Should have multiple feature categories
        assert all(isinstance(f, (int, float, np.number)) for f in features)
    
    def test_encode_education_level(self, matcher):
        """Test education level encoding"""
        # Test different education levels
        phd_level = matcher._encode_education_level(['Ph.D. in Computer Science'])
        master_level = matcher._encode_education_level(['Master of Science'])
        bachelor_level = matcher._encode_education_level(['Bachelor of Engineering'])
        
        assert phd_level == 4.0
        assert master_level == 3.0
        assert bachelor_level == 2.0
        
        # Test empty education
        empty_level = matcher._encode_education_level([])
        assert empty_level == 0.0
    
    def test_calculate_job_match_features(self, matcher, sample_resume_data, sample_job_requirements):
        """Test job-specific matching features calculation"""
        features = matcher._calculate_job_match_features(sample_resume_data, sample_job_requirements)
        
        assert isinstance(features, list)
        assert len(features) == 5  # Expected number of job match features
        
        # Skills match should be high (3/3 skills match)
        skills_match = features[0]
        assert skills_match == 1.0  # Perfect skill match
    
    def test_predict_match_untrained(self, matcher, sample_resume_data, sample_job_requirements):
        """Test matching prediction without trained models (rule-based fallback)"""
        result = matcher.predict_match(sample_resume_data, sample_job_requirements)
        
        assert isinstance(result, MatchingResult)
        assert 0 <= result.overall_score <= 1
        assert 0 <= result.technical_fit <= 1
        assert 0 <= result.cognitive_compatibility <= 1
        assert 0 <= result.confidence <= 1
        assert isinstance(result.recommendations, list)
    
    def test_calculate_detailed_scores(self, matcher, sample_resume_data, sample_job_requirements):
        """Test detailed score calculation"""
        scores = matcher._calculate_detailed_scores(sample_resume_data, sample_job_requirements)
        
        expected_keys = [
            'cognitive_compatibility',
            'growth_alignment', 
            'innovation_match',
            'technical_fit',
            'cultural_fit'
        ]
        
        for key in expected_keys:
            assert key in scores
            assert 0 <= scores[key] <= 1
    
    def test_generate_recommendations(self, matcher, sample_job_requirements):
        """Test recommendation generation"""
        # Test with low scores to trigger recommendations
        low_scores = {
            'technical_fit': 0.3,
            'growth_alignment': 0.4,
            'innovation_match': 0.2,
            'cultural_fit': 0.5,
            'cognitive_compatibility': 0.4
        }
        
        recommendations = matcher._generate_recommendations(low_scores, sample_job_requirements)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 3  # Limited to top 3
        assert all(isinstance(rec, str) for rec in recommendations)
    
    def test_feature_names_generation(self, matcher):
        """Test feature names generation"""
        feature_names = matcher._get_feature_names()
        
        assert isinstance(feature_names, list)
        assert len(feature_names) > 20  # Should have many features
        
        # Check for expected feature categories
        feature_str = ' '.join(feature_names)
        assert 'experience_years' in feature_str
        assert 'skills_count' in feature_str
        assert 'education_level' in feature_str

class TestMatchingResult:
    
    def test_matching_result_creation(self):
        """Test MatchingResult dataclass creation"""
        result = MatchingResult(
            overall_score=0.85,
            cognitive_compatibility=0.8,
            growth_alignment=0.9,
            innovation_match=0.7,
            technical_fit=0.95,
            cultural_fit=0.8,
            explanation={'method': 'test'},
            confidence=0.9,
            recommendations=['Test recommendation']
        )
        
        assert result.overall_score == 0.85
        assert result.cognitive_compatibility == 0.8
        assert result.confidence == 0.9
        assert len(result.recommendations) == 1

class TestModelWeights:
    
    def test_calculate_optimal_weights(self):
        """Test optimal weight calculation"""
        matcher = EnsembleMatcher()
        
        # Mock model scores
        model_scores = {
            'random_forest': 0.9,
            'gradient_boosting': 0.85,
            'logistic_regression': 0.8,
            'svm': 0.75
        }
        
        matcher._calculate_optimal_weights(model_scores)
        
        # Check weights sum to approximately 1
        total_weight = sum(matcher.model_weights.values())
        assert abs(total_weight - 1.0) < 0.01
        
        # Check best model has highest weight
        best_model = max(model_scores, key=model_scores.get)
        assert matcher.model_weights[best_model] == max(matcher.model_weights.values())
    
    def test_calculate_weights_zero_scores(self):
        """Test weight calculation with zero scores"""
        matcher = EnsembleMatcher()
        
        model_scores = {
            'random_forest': 0.0,
            'gradient_boosting': 0.0,
            'logistic_regression': 0.0,
            'svm': 0.0
        }
        
        matcher._calculate_optimal_weights(model_scores)
        
        # Should default to equal weights
        for weight in matcher.model_weights.values():
            assert weight == 0.25

class TestIntegration:
    
    def test_end_to_end_matching(self):
        """Test complete end-to-end matching process"""
        matcher = EnsembleMatcher()
        
        resume_data = {
            'name': 'Jane Smith',
            'experience': '7',
            'skills': ['python', 'tensorflow', 'aws', 'sql', 'machine learning'],
            'education': ['Master of Science in Data Science'],
            'current_role': 'Senior ML Engineer'
        }
        
        job_requirements = {
            'role': 'Principal ML Engineer',
            'required_skills': ['python', 'tensorflow', 'machine learning'],
            'min_experience': 5,
            'education_level': 'master',
            'role_level': 'senior',
            'cognitive_requirements': {
                'Analytical Thinker': 0.9,
                'Creative Innovator': 0.7
            },
            'innovation_level': 0.8
        }
        
        # Run complete matching
        result = matcher.predict_match(resume_data, job_requirements)
        
        # Verify results
        assert isinstance(result, MatchingResult)
        assert 0 <= result.overall_score <= 1
        assert result.technical_fit > 0.5  # Should be high due to skill overlap
        assert len(result.recommendations) >= 0
        
    def test_batch_feature_extraction(self):
        """Test feature extraction for multiple candidates"""
        matcher = EnsembleMatcher()
        
        resumes = [
            {
                'name': 'Candidate 1',
                'experience': '3',
                'skills': ['python', 'sql'],
                'education': ['Bachelor of Science'],
                'current_role': 'Data Analyst'
            },
            {
                'name': 'Candidate 2', 
                'experience': '8',
                'skills': ['python', 'machine learning', 'aws'],
                'education': ['Master of Science'],
                'current_role': 'Senior Data Scientist'
            }
        ]
        
        job_req = {
            'required_skills': ['python', 'machine learning'],
            'min_experience': 3,
            'education_level': 'bachelor'
        }
        
        # Extract features for all candidates
        features_list = []
        for resume in resumes:
            features = matcher.extract_comprehensive_features(resume, job_req)
            features_list.append(features)
        
        assert len(features_list) == 2
        assert all(isinstance(f, np.ndarray) for f in features_list)
        assert features_list[0].shape == features_list[1].shape

if __name__ == "__main__":
    pytest.main([__file__])
