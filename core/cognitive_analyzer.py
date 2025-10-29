"""
NeuroMatch AI - Cognitive Pattern Analysis Engine
BERT-based cognitive style classification with 8 cognitive patterns
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import logging
from typing import Dict, List, Tuple, Optional
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CognitivePatternClassifier(nn.Module):
    """BERT-based neural network for cognitive pattern classification"""
    
    def __init__(self, model_name='distilbert-base-uncased', num_patterns=8, dropout_rate=0.3):
        super(CognitivePatternClassifier, self).__init__()
        self.num_patterns = num_patterns
        
        # Load pre-trained BERT model
        self.bert = AutoModel.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        
        # Freeze BERT parameters for transfer learning
        for param in self.bert.parameters():
            param.requires_grad = False
            
        # Unfreeze last 2 layers for fine-tuning
        for param in self.bert.encoder.layer[-2:].parameters():
            param.requires_grad = True
            
        # Classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_patterns)
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

class CognitiveAnalyzer:
    """Main cognitive pattern analysis engine"""
    
    # 8 Cognitive Patterns for Professional Assessment
    COGNITIVE_PATTERNS = {
        0: "Analytical Thinker",      # Data-driven, logical reasoning
        1: "Creative Innovator",      # Out-of-box thinking, artistic
        2: "Strategic Planner",       # Long-term vision, systematic
        3: "Collaborative Leader",    # Team-oriented, communication
        4: "Detail Perfectionist",    # Precision, quality-focused
        5: "Adaptive Problem-Solver", # Flexible, quick learning
        6: "Results-Driven Executor", # Goal-oriented, performance
        7: "Empathetic Communicator"  # People-focused, emotional intelligence
    }
    
    def __init__(self, model_name='distilbert-base-uncased', device=None):
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.is_trained = False
        
        logger.info(f"Initialized CognitiveAnalyzer with device: {self.device}")
        
    def preprocess_text(self, text: str, max_length: int = 512) -> Dict:
        """Tokenize and preprocess text for BERT input"""
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].to(self.device),
            'attention_mask': encoding['attention_mask'].to(self.device)
        }
    
    def extract_cognitive_features(self, resume_text: str) -> Dict[str, float]:
        """Extract cognitive indicators from resume text"""
        text = resume_text.lower()
        
        # Feature extraction based on linguistic patterns
        features = {
            'analytical_score': 0.0,
            'creative_score': 0.0,
            'strategic_score': 0.0,
            'collaborative_score': 0.0,
            'detail_score': 0.0,
            'adaptive_score': 0.0,
            'results_score': 0.0,
            'empathetic_score': 0.0
        }
        
        # Analytical indicators
        analytical_keywords = ['data', 'analysis', 'research', 'statistics', 'metrics', 'algorithm', 'model', 'quantitative']
        features['analytical_score'] = sum(1 for word in analytical_keywords if word in text) / len(analytical_keywords)
        
        # Creative indicators
        creative_keywords = ['design', 'creative', 'innovative', 'artistic', 'visual', 'concept', 'ideation', 'brainstorm']
        features['creative_score'] = sum(1 for word in creative_keywords if word in text) / len(creative_keywords)
        
        # Strategic indicators
        strategic_keywords = ['strategy', 'planning', 'vision', 'roadmap', 'framework', 'architecture', 'systematic']
        features['strategic_score'] = sum(1 for word in strategic_keywords if word in text) / len(strategic_keywords)
        
        # Collaborative indicators
        collaborative_keywords = ['team', 'collaboration', 'communication', 'leadership', 'mentoring', 'coordination']
        features['collaborative_score'] = sum(1 for word in collaborative_keywords if word in text) / len(collaborative_keywords)
        
        # Detail-oriented indicators
        detail_keywords = ['quality', 'precision', 'accuracy', 'detailed', 'thorough', 'meticulous', 'documentation']
        features['detail_score'] = sum(1 for word in detail_keywords if word in text) / len(detail_keywords)
        
        # Adaptive indicators
        adaptive_keywords = ['learning', 'adaptable', 'flexible', 'agile', 'quick', 'versatile', 'dynamic']
        features['adaptive_score'] = sum(1 for word in adaptive_keywords if word in text) / len(adaptive_keywords)
        
        # Results-driven indicators
        results_keywords = ['results', 'achievement', 'performance', 'goal', 'target', 'success', 'delivered']
        features['results_score'] = sum(1 for word in results_keywords if word in text) / len(results_keywords)
        
        # Empathetic indicators
        empathetic_keywords = ['customer', 'user', 'stakeholder', 'relationship', 'support', 'service', 'empathy']
        features['empathetic_score'] = sum(1 for word in empathetic_keywords if word in text) / len(empathetic_keywords)
        
        return features
    
    def train_model(self, training_data: List[Tuple[str, int]], validation_split=0.2, 
                   epochs=10, batch_size=16, learning_rate=2e-5):
        """Train the cognitive pattern classifier"""
        logger.info("Starting cognitive pattern model training...")
        
        # Prepare data
        texts, labels = zip(*training_data)
        X_train, X_val, y_train, y_val = train_test_split(
            texts, labels, test_size=validation_split, random_state=42, stratify=labels
        )
        
        # Initialize model
        self.model = CognitivePatternClassifier().to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            correct_predictions = 0
            total_predictions = 0
            
            # Process in batches
            for i in range(0, len(X_train), batch_size):
                batch_texts = X_train[i:i+batch_size]
                batch_labels = torch.tensor(y_train[i:i+batch_size]).to(self.device)
                
                # Tokenize batch
                batch_encodings = [self.preprocess_text(text) for text in batch_texts]
                input_ids = torch.cat([enc['input_ids'] for enc in batch_encodings])
                attention_mask = torch.cat([enc['attention_mask'] for enc in batch_encodings])
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, batch_labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                predictions = torch.argmax(outputs, dim=1)
                correct_predictions += (predictions == batch_labels).sum().item()
                total_predictions += len(batch_labels)
            
            # Validation
            val_accuracy = self._validate_model(X_val, y_val, batch_size)
            train_accuracy = correct_predictions / total_predictions
            
            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}, "
                       f"Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")
        
        self.is_trained = True
        logger.info("Model training completed successfully!")
        
    def _validate_model(self, X_val, y_val, batch_size):
        """Validate model performance"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i in range(0, len(X_val), batch_size):
                batch_texts = X_val[i:i+batch_size]
                batch_labels = torch.tensor(y_val[i:i+batch_size]).to(self.device)
                
                batch_encodings = [self.preprocess_text(text) for text in batch_texts]
                input_ids = torch.cat([enc['input_ids'] for enc in batch_encodings])
                attention_mask = torch.cat([enc['attention_mask'] for enc in batch_encodings])
                
                outputs = self.model(input_ids, attention_mask)
                predictions = torch.argmax(outputs, dim=1)
                correct += (predictions == batch_labels).sum().item()
                total += len(batch_labels)
        
        self.model.train()
        return correct / total
    
    def predict_cognitive_pattern(self, resume_text: str) -> Dict[str, float]:
        """Predict cognitive pattern for a resume"""
        if not self.is_trained or self.model is None:
            logger.warning("Model not trained. Using feature-based prediction.")
            return self._feature_based_prediction(resume_text)
        
        self.model.eval()
        with torch.no_grad():
            encoding = self.preprocess_text(resume_text)
            outputs = self.model(encoding['input_ids'], encoding['attention_mask'])
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            
        # Convert to pattern scores
        pattern_scores = {}
        for idx, pattern_name in self.COGNITIVE_PATTERNS.items():
            pattern_scores[pattern_name] = float(probabilities[idx])
            
        return pattern_scores
    
    def _feature_based_prediction(self, resume_text: str) -> Dict[str, float]:
        """Fallback feature-based prediction when model is not trained"""
        features = self.extract_cognitive_features(resume_text)
        
        # Map features to cognitive patterns
        pattern_scores = {
            "Analytical Thinker": features['analytical_score'],
            "Creative Innovator": features['creative_score'],
            "Strategic Planner": features['strategic_score'],
            "Collaborative Leader": features['collaborative_score'],
            "Detail Perfectionist": features['detail_score'],
            "Adaptive Problem-Solver": features['adaptive_score'],
            "Results-Driven Executor": features['results_score'],
            "Empathetic Communicator": features['empathetic_score']
        }
        
        # Normalize scores
        total_score = sum(pattern_scores.values())
        if total_score > 0:
            pattern_scores = {k: v/total_score for k, v in pattern_scores.items()}
        else:
            # Default uniform distribution if no patterns detected
            pattern_scores = {k: 1.0/len(pattern_scores) for k in pattern_scores}
            
        return pattern_scores
    
    def get_dominant_pattern(self, resume_text: str) -> Tuple[str, float]:
        """Get the dominant cognitive pattern and its confidence"""
        pattern_scores = self.predict_cognitive_pattern(resume_text)
        dominant_pattern = max(pattern_scores, key=pattern_scores.get)
        confidence = pattern_scores[dominant_pattern]
        return dominant_pattern, confidence
    
    def save_model(self, model_path: str):
        """Save trained model"""
        if self.model is None:
            raise ValueError("No model to save")
            
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'cognitive_patterns': self.COGNITIVE_PATTERNS,
            'is_trained': self.is_trained,
            'timestamp': datetime.now().isoformat()
        }, model_path)
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str):
        """Load trained model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = CognitivePatternClassifier().to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.is_trained = checkpoint['is_trained']
        logger.info(f"Model loaded from {model_path}")

def analyze_cognitive_compatibility(candidate_pattern: Dict[str, float], 
                                  job_requirements: Dict[str, float]) -> float:
    """Calculate cognitive compatibility between candidate and job requirements"""
    compatibility_score = 0.0
    
    for pattern, job_weight in job_requirements.items():
        if pattern in candidate_pattern:
            compatibility_score += candidate_pattern[pattern] * job_weight
    
    return min(compatibility_score, 1.0)  # Cap at 1.0

# Example usage and testing
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = CognitiveAnalyzer()
    
    # Example resume text
    sample_resume = """
    Experienced data scientist with 5 years in machine learning and analytics.
    Led cross-functional teams to deliver innovative solutions. Strong background
    in statistical analysis, algorithm development, and strategic planning.
    Passionate about mentoring junior developers and fostering collaborative
    team environments. Achieved 95% accuracy in predictive models.
    """
    
    # Analyze cognitive pattern
    pattern_scores = analyzer.predict_cognitive_pattern(sample_resume)
    dominant_pattern, confidence = analyzer.get_dominant_pattern(sample_resume)
    
    print("Cognitive Pattern Analysis:")
    for pattern, score in pattern_scores.items():
        print(f"  {pattern}: {score:.3f}")
    print(f"\nDominant Pattern: {dominant_pattern} (confidence: {confidence:.3f})")
