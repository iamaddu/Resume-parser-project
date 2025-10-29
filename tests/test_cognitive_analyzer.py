"""
Test suite for Cognitive Analyzer
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.cognitive_analyzer import CognitiveAnalyzer, CognitivePatternClassifier

class TestCognitiveAnalyzer:
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance for testing"""
        return CognitiveAnalyzer()
    
    @pytest.fixture
    def sample_resume_text(self):
        """Sample resume text for testing"""
        return """
        Experienced data scientist with 5 years in machine learning and analytics.
        Led cross-functional teams to deliver innovative solutions. Strong background
        in statistical analysis, algorithm development, and strategic planning.
        Passionate about mentoring junior developers and fostering collaborative
        team environments. Achieved 95% accuracy in predictive models.
        """
    
    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization"""
        assert analyzer is not None
        assert analyzer.model_name == 'distilbert-base-uncased'
        assert analyzer.device in ['cpu', 'cuda']
        assert len(analyzer.COGNITIVE_PATTERNS) == 8
    
    def test_cognitive_patterns_defined(self, analyzer):
        """Test that all cognitive patterns are properly defined"""
        expected_patterns = [
            "Analytical Thinker",
            "Creative Innovator", 
            "Strategic Planner",
            "Collaborative Leader",
            "Detail Perfectionist",
            "Adaptive Problem-Solver",
            "Results-Driven Executor",
            "Empathetic Communicator"
        ]
        
        for pattern in expected_patterns:
            assert pattern in analyzer.COGNITIVE_PATTERNS.values()
    
    def test_extract_cognitive_features(self, analyzer, sample_resume_text):
        """Test cognitive feature extraction"""
        features = analyzer.extract_cognitive_features(sample_resume_text)
        
        assert isinstance(features, dict)
        assert len(features) == 8  # 8 cognitive features
        
        # Check all scores are between 0 and 1
        for score in features.values():
            assert 0 <= score <= 1
    
    def test_predict_cognitive_pattern_untrained(self, analyzer, sample_resume_text):
        """Test prediction without trained model (feature-based fallback)"""
        pattern_scores = analyzer.predict_cognitive_pattern(sample_resume_text)
        
        assert isinstance(pattern_scores, dict)
        assert len(pattern_scores) == 8
        
        # Check all patterns have scores
        for pattern_name in analyzer.COGNITIVE_PATTERNS.values():
            assert pattern_name in pattern_scores
            assert 0 <= pattern_scores[pattern_name] <= 1
        
        # Check scores sum to approximately 1 (normalized)
        total_score = sum(pattern_scores.values())
        assert abs(total_score - 1.0) < 0.1
    
    def test_get_dominant_pattern(self, analyzer, sample_resume_text):
        """Test dominant pattern identification"""
        dominant_pattern, confidence = analyzer.get_dominant_pattern(sample_resume_text)
        
        assert isinstance(dominant_pattern, str)
        assert dominant_pattern in analyzer.COGNITIVE_PATTERNS.values()
        assert 0 <= confidence <= 1
    
    def test_preprocess_text(self, analyzer):
        """Test text preprocessing"""
        test_text = "This is a test resume with various skills and experience."
        
        processed = analyzer.preprocess_text(test_text)
        
        assert 'input_ids' in processed
        assert 'attention_mask' in processed
        assert processed['input_ids'].shape[1] <= 512  # Max length check
    
    @patch('torch.cuda.is_available')
    def test_device_selection_cuda_available(self, mock_cuda, analyzer):
        """Test device selection when CUDA is available"""
        mock_cuda.return_value = True
        new_analyzer = CognitiveAnalyzer()
        # Note: Actual device might still be CPU if no GPU present
        assert new_analyzer.device in ['cpu', 'cuda']
    
    @patch('torch.cuda.is_available')
    def test_device_selection_cuda_unavailable(self, mock_cuda, analyzer):
        """Test device selection when CUDA is unavailable"""
        mock_cuda.return_value = False
        new_analyzer = CognitiveAnalyzer()
        assert new_analyzer.device == 'cpu'

class TestCognitivePatternClassifier:
    
    def test_model_initialization(self):
        """Test model initialization"""
        model = CognitivePatternClassifier()
        
        assert model is not None
        assert model.num_patterns == 8
        assert hasattr(model, 'bert')
        assert hasattr(model, 'classifier')
    
    def test_forward_pass(self):
        """Test forward pass through model"""
        model = CognitivePatternClassifier()
        
        # Create dummy input
        batch_size = 2
        seq_length = 128
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
        
        assert outputs.shape == (batch_size, 8)  # 8 cognitive patterns

class TestCognitiveCompatibility:
    
    def test_analyze_cognitive_compatibility(self):
        """Test cognitive compatibility analysis"""
        from core.cognitive_analyzer import analyze_cognitive_compatibility
        
        candidate_pattern = {
            "Analytical Thinker": 0.8,
            "Strategic Planner": 0.6,
            "Creative Innovator": 0.3
        }
        
        job_requirements = {
            "Analytical Thinker": 0.9,
            "Strategic Planner": 0.7,
            "Creative Innovator": 0.2
        }
        
        compatibility = analyze_cognitive_compatibility(candidate_pattern, job_requirements)
        
        assert 0 <= compatibility <= 1
        assert isinstance(compatibility, float)
    
    def test_compatibility_perfect_match(self):
        """Test compatibility with perfect match"""
        from core.cognitive_analyzer import analyze_cognitive_compatibility
        
        pattern = {"Analytical Thinker": 1.0}
        requirements = {"Analytical Thinker": 1.0}
        
        compatibility = analyze_cognitive_compatibility(pattern, requirements)
        assert compatibility == 1.0
    
    def test_compatibility_no_match(self):
        """Test compatibility with no matching patterns"""
        from core.cognitive_analyzer import analyze_cognitive_compatibility
        
        pattern = {"Analytical Thinker": 1.0}
        requirements = {"Creative Innovator": 1.0}
        
        compatibility = analyze_cognitive_compatibility(pattern, requirements)
        assert compatibility == 0.0

class TestIntegration:
    
    def test_end_to_end_analysis(self):
        """Test complete end-to-end cognitive analysis"""
        analyzer = CognitiveAnalyzer()
        
        resume_text = """
        Senior Software Engineer with expertise in machine learning and data analysis.
        Led multiple teams to deliver innovative products. Strong analytical thinking
        and strategic planning skills. Passionate about solving complex problems
        and mentoring team members.
        """
        
        # Run complete analysis
        pattern_scores = analyzer.predict_cognitive_pattern(resume_text)
        dominant_pattern, confidence = analyzer.get_dominant_pattern(resume_text)
        features = analyzer.extract_cognitive_features(resume_text)
        
        # Verify results
        assert len(pattern_scores) == 8
        assert dominant_pattern in analyzer.COGNITIVE_PATTERNS.values()
        assert 0 <= confidence <= 1
        assert len(features) == 8
        
        # Check that analytical thinking is prominent (given the text)
        assert pattern_scores["Analytical Thinker"] > 0.1  # Should detect analytical keywords

if __name__ == "__main__":
    pytest.main([__file__])
