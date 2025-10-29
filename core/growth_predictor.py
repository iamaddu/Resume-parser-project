"""
NeuroMatch AI - Growth Prediction System
LSTM-based career trajectory analysis and future role prediction
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import logging
from typing import Dict, List, Tuple, Optional, Any
import os
from datetime import datetime, timedelta
import re
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CareerTrajectory:
    """Data class for career trajectory information"""
    positions: List[str]
    durations: List[float]  # in years
    skill_progression: List[List[str]]
    salary_progression: List[float]
    industry_changes: List[str]
    education_timeline: List[Tuple[str, str]]  # (degree, year)

class CareerLSTM(nn.Module):
    """LSTM neural network for career growth prediction"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3, output_size=1):
        super(CareerLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use last output for prediction
        last_output = attn_out[:, -1, :]
        last_output = self.dropout(last_output)
        
        # Final prediction
        output = self.fc(last_output)
        return output

class GrowthPredictor:
    """Main growth prediction engine"""
    
    # Career levels and progression mapping
    CAREER_LEVELS = {
        'intern': 0, 'junior': 1, 'mid': 2, 'senior': 3, 
        'lead': 4, 'principal': 5, 'manager': 6, 'director': 7, 
        'vp': 8, 'c-level': 9
    }
    
    ROLE_CATEGORIES = {
        'technical': ['engineer', 'developer', 'scientist', 'analyst', 'architect'],
        'management': ['manager', 'director', 'lead', 'supervisor', 'head'],
        'product': ['product', 'pm', 'owner', 'strategy'],
        'design': ['designer', 'ux', 'ui', 'creative'],
        'sales': ['sales', 'account', 'business development'],
        'marketing': ['marketing', 'brand', 'content', 'digital'],
        'operations': ['operations', 'ops', 'logistics', 'supply chain'],
        'finance': ['finance', 'accounting', 'controller', 'cfo']
    }
    
    def __init__(self, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.feature_names = []
        
        logger.info(f"Initialized GrowthPredictor with device: {self.device}")
    
    def extract_career_features(self, resume_data: Dict, career_history: Optional[List[Dict]] = None) -> np.ndarray:
        """Extract numerical features for career growth prediction"""
        features = []
        
        # Basic metrics
        experience_years = float(resume_data.get('experience', 0))
        features.append(experience_years)
        
        # Education level encoding
        education_level = self._encode_education_level(resume_data.get('education', []))
        features.append(education_level)
        
        # Skills diversity and technical depth
        skills = resume_data.get('skills', [])
        skills_count = len(skills)
        technical_skills_ratio = self._calculate_technical_skills_ratio(skills)
        features.extend([skills_count, technical_skills_ratio])
        
        # Career trajectory features (if available)
        if career_history:
            trajectory_features = self._extract_trajectory_features(career_history)
            features.extend(trajectory_features)
        else:
            # Default trajectory features
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0])  # 5 default trajectory features
        
        # Industry and role category features
        current_role_category = self._categorize_role(resume_data.get('current_role', ''))
        features.append(current_role_category)
        
        # Learning velocity indicators
        learning_velocity = self._estimate_learning_velocity(resume_data)
        features.append(learning_velocity)
        
        # Leadership indicators
        leadership_score = self._calculate_leadership_score(resume_data)
        features.append(leadership_score)
        
        # Innovation indicators
        innovation_score = self._calculate_innovation_score(resume_data)
        features.append(innovation_score)
        
        # Adaptability score
        adaptability_score = self._calculate_adaptability_score(resume_data)
        features.append(adaptability_score)
        
        return np.array(features, dtype=np.float32)
    
    def _encode_education_level(self, education_list: List[str]) -> float:
        """Encode education level as numerical value"""
        education_text = ' '.join(education_list).lower()
        
        if any(degree in education_text for degree in ['phd', 'doctorate']):
            return 4.0
        elif any(degree in education_text for degree in ['master', 'mba', 'm.tech', 'm.sc']):
            return 3.0
        elif any(degree in education_text for degree in ['bachelor', 'b.tech', 'b.sc', 'bs']):
            return 2.0
        elif any(degree in education_text for degree in ['diploma', 'certificate']):
            return 1.0
        else:
            return 0.0
    
    def _calculate_technical_skills_ratio(self, skills: List[str]) -> float:
        """Calculate ratio of technical to total skills"""
        if not skills:
            return 0.0
            
        technical_keywords = [
            'python', 'java', 'javascript', 'sql', 'machine learning', 'ai',
            'data science', 'cloud', 'aws', 'docker', 'kubernetes', 'react',
            'node.js', 'tensorflow', 'pytorch', 'git', 'linux'
        ]
        
        technical_count = sum(1 for skill in skills 
                            if any(tech in skill.lower() for tech in technical_keywords))
        
        return technical_count / len(skills)
    
    def _extract_trajectory_features(self, career_history: List[Dict]) -> List[float]:
        """Extract features from career trajectory"""
        if not career_history:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Career progression rate
        progression_rate = len(career_history) / max(1, career_history[-1].get('years_experience', 1))
        
        # Average role duration
        durations = [role.get('duration', 1.0) for role in career_history]
        avg_duration = np.mean(durations)
        
        # Role level progression
        levels = [self._get_role_level(role.get('title', '')) for role in career_history]
        level_progression = (levels[-1] - levels[0]) / max(1, len(levels) - 1) if len(levels) > 1 else 0
        
        # Industry diversity
        industries = [role.get('industry', 'unknown') for role in career_history]
        industry_diversity = len(set(industries)) / len(industries) if industries else 0
        
        # Salary growth rate (if available)
        salaries = [role.get('salary', 0) for role in career_history if role.get('salary', 0) > 0]
        salary_growth = 0.0
        if len(salaries) > 1:
            salary_growth = (salaries[-1] - salaries[0]) / salaries[0]
        
        return [progression_rate, avg_duration, level_progression, industry_diversity, salary_growth]
    
    def _get_role_level(self, title: str) -> int:
        """Get numerical level for a job title"""
        title_lower = title.lower()
        
        for level_name, level_value in self.CAREER_LEVELS.items():
            if level_name in title_lower:
                return level_value
        
        # Default based on common patterns
        if any(word in title_lower for word in ['intern', 'trainee']):
            return 0
        elif any(word in title_lower for word in ['junior', 'associate']):
            return 1
        elif any(word in title_lower for word in ['senior', 'sr']):
            return 3
        elif any(word in title_lower for word in ['lead', 'principal']):
            return 4
        else:
            return 2  # Default to mid-level
    
    def _categorize_role(self, role_title: str) -> float:
        """Categorize role and return numerical encoding"""
        title_lower = role_title.lower()
        
        for category, keywords in self.ROLE_CATEGORIES.items():
            if any(keyword in title_lower for keyword in keywords):
                return hash(category) % 10  # Simple hash-based encoding
        
        return 0.0  # Default category
    
    def _estimate_learning_velocity(self, resume_data: Dict) -> float:
        """Estimate learning velocity based on skill acquisition"""
        skills = resume_data.get('skills', [])
        experience_years = float(resume_data.get('experience', 1))
        
        # Skills per year as a proxy for learning velocity
        return len(skills) / max(1, experience_years)
    
    def _calculate_leadership_score(self, resume_data: Dict) -> float:
        """Calculate leadership potential score"""
        text = ' '.join([
            str(resume_data.get('current_role', '')),
            ' '.join(resume_data.get('skills', [])),
            ' '.join(resume_data.get('education', []))
        ]).lower()
        
        leadership_keywords = [
            'lead', 'manage', 'supervise', 'mentor', 'coach', 'team',
            'project management', 'leadership', 'coordination', 'delegation'
        ]
        
        score = sum(1 for keyword in leadership_keywords if keyword in text)
        return min(score / len(leadership_keywords), 1.0)
    
    def _calculate_innovation_score(self, resume_data: Dict) -> float:
        """Calculate innovation potential score"""
        text = ' '.join([
            str(resume_data.get('current_role', '')),
            ' '.join(resume_data.get('skills', [])),
            ' '.join(resume_data.get('education', []))
        ]).lower()
        
        innovation_keywords = [
            'innovation', 'creative', 'research', 'development', 'patent',
            'prototype', 'design', 'algorithm', 'optimization', 'improvement'
        ]
        
        score = sum(1 for keyword in innovation_keywords if keyword in text)
        return min(score / len(innovation_keywords), 1.0)
    
    def _calculate_adaptability_score(self, resume_data: Dict) -> float:
        """Calculate adaptability score"""
        skills = resume_data.get('skills', [])
        
        # Diversity of skill categories as adaptability indicator
        skill_categories = set()
        for skill in skills:
            skill_lower = skill.lower()
            for category, keywords in self.ROLE_CATEGORIES.items():
                if any(keyword in skill_lower for keyword in keywords):
                    skill_categories.add(category)
        
        return len(skill_categories) / len(self.ROLE_CATEGORIES)
    
    def prepare_sequence_data(self, career_trajectories: List[CareerTrajectory], 
                            sequence_length: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequential data for LSTM training"""
        X, y = [], []
        
        for trajectory in career_trajectories:
            if len(trajectory.positions) < sequence_length + 1:
                continue
                
            # Create sequences
            for i in range(len(trajectory.positions) - sequence_length):
                # Input sequence
                sequence_features = []
                for j in range(i, i + sequence_length):
                    features = [
                        self._get_role_level(trajectory.positions[j]),
                        trajectory.durations[j] if j < len(trajectory.durations) else 1.0,
                        len(trajectory.skill_progression[j]) if j < len(trajectory.skill_progression) else 0,
                        trajectory.salary_progression[j] if j < len(trajectory.salary_progression) else 50000
                    ]
                    sequence_features.append(features)
                
                X.append(sequence_features)
                
                # Target (next career level)
                next_level = self._get_role_level(trajectory.positions[i + sequence_length])
                y.append(next_level)
        
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    
    def train_model(self, training_data: List[Tuple[Dict, float]], 
                   validation_split: float = 0.2, epochs: int = 100, 
                   batch_size: int = 32, learning_rate: float = 0.001):
        """Train the growth prediction model"""
        logger.info("Starting growth prediction model training...")
        
        # Prepare features and targets
        X_features = []
        y_targets = []
        
        for resume_data, growth_score in training_data:
            features = self.extract_career_features(resume_data)
            X_features.append(features)
            y_targets.append(growth_score)
        
        X = np.array(X_features)
        y = np.array(y_targets).reshape(-1, 1)
        
        # Store feature names for later use
        self.feature_names = [
            'experience_years', 'education_level', 'skills_count', 'technical_ratio',
            'progression_rate', 'avg_duration', 'level_progression', 'industry_diversity',
            'salary_growth', 'role_category', 'learning_velocity', 'leadership_score',
            'innovation_score', 'adaptability_score'
        ]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=validation_split, random_state=42
        )
        
        # Reshape for LSTM (add sequence dimension)
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).to(self.device)
        
        # Initialize model
        input_size = X_train.shape[2]
        self.model = CareerLSTM(input_size=input_size).to(self.device)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for i in range(0, len(X_train_tensor), batch_size):
                batch_X = X_train_tensor[i:i+batch_size]
                batch_y = y_train_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 20:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        self.is_trained = True
        logger.info("Growth prediction model training completed!")
    
    def predict_growth_potential(self, resume_data: Dict, 
                               career_history: Optional[List[Dict]] = None) -> Dict[str, float]:
        """Predict career growth potential"""
        if not self.is_trained or self.model is None:
            return self._rule_based_growth_prediction(resume_data)
        
        # Extract and scale features
        features = self.extract_career_features(resume_data, career_history)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        features_tensor = torch.FloatTensor(features_scaled.reshape(1, 1, -1)).to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            growth_score = self.model(features_tensor).cpu().numpy()[0, 0]
        
        # Generate comprehensive growth analysis
        return self._generate_growth_analysis(resume_data, growth_score, features)
    
    def _rule_based_growth_prediction(self, resume_data: Dict) -> Dict[str, float]:
        """Fallback rule-based growth prediction"""
        experience = float(resume_data.get('experience', 0))
        education_level = self._encode_education_level(resume_data.get('education', []))
        skills_count = len(resume_data.get('skills', []))
        
        # Simple growth score calculation
        growth_score = (
            min(experience / 10, 1.0) * 0.4 +  # Experience factor
            education_level / 4.0 * 0.3 +      # Education factor
            min(skills_count / 20, 1.0) * 0.3   # Skills factor
        )
        
        return self._generate_growth_analysis(resume_data, growth_score, None)
    
    def _generate_growth_analysis(self, resume_data: Dict, growth_score: float, 
                                features: Optional[np.ndarray]) -> Dict[str, float]:
        """Generate comprehensive growth analysis"""
        analysis = {
            'overall_growth_potential': float(growth_score),
            'technical_growth': 0.0,
            'leadership_growth': 0.0,
            'innovation_potential': 0.0,
            'adaptability_score': 0.0,
            'learning_velocity': 0.0,
            'career_trajectory_score': 0.0,
            'next_role_probability': 0.0
        }
        
        if features is not None:
            # Extract specific scores from features
            analysis['technical_growth'] = float(features[3])  # technical_ratio
            analysis['leadership_growth'] = float(features[11])  # leadership_score
            analysis['innovation_potential'] = float(features[12])  # innovation_score
            analysis['adaptability_score'] = float(features[13])  # adaptability_score
            analysis['learning_velocity'] = float(features[10])  # learning_velocity
            analysis['career_trajectory_score'] = float(features[6])  # level_progression
        else:
            # Calculate from resume data directly
            analysis['technical_growth'] = self._calculate_technical_skills_ratio(resume_data.get('skills', []))
            analysis['leadership_growth'] = self._calculate_leadership_score(resume_data)
            analysis['innovation_potential'] = self._calculate_innovation_score(resume_data)
            analysis['adaptability_score'] = self._calculate_adaptability_score(resume_data)
            analysis['learning_velocity'] = self._estimate_learning_velocity(resume_data)
        
        # Calculate next role probability based on growth score
        analysis['next_role_probability'] = min(growth_score * 1.2, 1.0)
        
        return analysis
    
    def predict_future_roles(self, resume_data: Dict, years_ahead: int = 5) -> List[Dict[str, Any]]:
        """Predict potential future roles"""
        growth_analysis = self.predict_growth_potential(resume_data)
        current_experience = float(resume_data.get('experience', 0))
        
        future_roles = []
        
        for year in range(1, years_ahead + 1):
            projected_experience = current_experience + year
            growth_factor = growth_analysis['overall_growth_potential']
            
            # Predict role level progression
            current_level = self._get_role_level(resume_data.get('current_role', ''))
            projected_level = min(current_level + int(year * growth_factor * 2), 9)
            
            # Generate role prediction
            role_prediction = {
                'year': year,
                'projected_experience': projected_experience,
                'estimated_level': projected_level,
                'level_name': self._get_level_name(projected_level),
                'probability': max(0.1, 1.0 - (year * 0.15)),  # Decreasing certainty
                'key_skills_needed': self._predict_required_skills(projected_level),
                'estimated_salary_range': self._estimate_salary_range(projected_level, projected_experience)
            }
            
            future_roles.append(role_prediction)
        
        return future_roles
    
    def _get_level_name(self, level: int) -> str:
        """Get level name from numerical level"""
        level_names = {v: k for k, v in self.CAREER_LEVELS.items()}
        return level_names.get(level, 'mid')
    
    def _predict_required_skills(self, level: int) -> List[str]:
        """Predict skills needed for future role level"""
        skill_recommendations = {
            0: ['basic programming', 'communication', 'learning'],
            1: ['programming languages', 'frameworks', 'testing'],
            2: ['system design', 'databases', 'project management'],
            3: ['architecture', 'mentoring', 'advanced algorithms'],
            4: ['team leadership', 'strategic thinking', 'cross-functional collaboration'],
            5: ['technical vision', 'innovation', 'industry expertise'],
            6: ['people management', 'budgeting', 'stakeholder management'],
            7: ['organizational leadership', 'business strategy', 'executive presence'],
            8: ['corporate strategy', 'board relations', 'market analysis'],
            9: ['visionary leadership', 'industry transformation', 'global perspective']
        }
        
        return skill_recommendations.get(level, ['continuous learning', 'adaptation'])
    
    def _estimate_salary_range(self, level: int, experience: float) -> Tuple[int, int]:
        """Estimate salary range for given level and experience"""
        base_salaries = {
            0: (40000, 60000),   # Intern/Entry
            1: (60000, 80000),   # Junior
            2: (80000, 120000),  # Mid
            3: (120000, 160000), # Senior
            4: (160000, 220000), # Lead/Principal
            5: (220000, 300000), # Principal/Staff
            6: (200000, 350000), # Manager
            7: (300000, 500000), # Director
            8: (500000, 800000), # VP
            9: (800000, 2000000) # C-Level
        }
        
        base_min, base_max = base_salaries.get(level, (80000, 120000))
        
        # Adjust for experience
        experience_multiplier = 1 + (experience * 0.05)
        
        return (
            int(base_min * experience_multiplier),
            int(base_max * experience_multiplier)
        )
    
    def save_model(self, model_path: str):
        """Save trained model and preprocessing objects"""
        if self.model is None:
            raise ValueError("No model to save")
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model and metadata
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained,
            'career_levels': self.CAREER_LEVELS,
            'role_categories': self.ROLE_CATEGORIES,
            'timestamp': datetime.now().isoformat()
        }, model_path)
        
        logger.info(f"Growth prediction model saved to {model_path}")
    
    def load_model(self, model_path: str):
        """Load trained model and preprocessing objects"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Load model architecture (need to know input size)
        if 'feature_names' in checkpoint:
            input_size = len(checkpoint['feature_names'])
            self.model = CareerLSTM(input_size=input_size).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.scaler = checkpoint['scaler']
        self.feature_names = checkpoint.get('feature_names', [])
        self.is_trained = checkpoint['is_trained']
        
        logger.info(f"Growth prediction model loaded from {model_path}")

# Example usage and testing
if __name__ == "__main__":
    # Initialize predictor
    predictor = GrowthPredictor()
    
    # Example resume data
    sample_resume = {
        'experience': '5',
        'education': ['Master of Science in Computer Science'],
        'skills': ['python', 'machine learning', 'sql', 'aws', 'leadership'],
        'current_role': 'Senior Data Scientist'
    }
    
    # Predict growth potential
    growth_analysis = predictor.predict_growth_potential(sample_resume)
    future_roles = predictor.predict_future_roles(sample_resume)
    
    print("Growth Analysis:")
    for metric, score in growth_analysis.items():
        print(f"  {metric}: {score:.3f}")
    
    print("\nFuture Role Predictions:")
    for role in future_roles:
        print(f"  Year {role['year']}: {role['level_name']} (Level {role['estimated_level']})")
        print(f"    Probability: {role['probability']:.2f}")
        print(f"    Salary Range: ${role['estimated_salary_range'][0]:,} - ${role['estimated_salary_range'][1]:,}")
