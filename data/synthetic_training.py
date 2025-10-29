"""
NeuroMatch AI - Synthetic Training Data Generator
Generate 10,000+ realistic training profiles with cognitive patterns and growth trajectories
"""

import numpy as np
import pandas as pd
import random
from faker import Faker
from typing import Dict, List, Tuple, Optional
import json
import os
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SyntheticDataGenerator:
    """Generate realistic synthetic resume and job matching data"""
    
    def __init__(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        self.fake = Faker()
        Faker.seed(seed)
        
        # Define skill categories and levels
        self.skill_categories = {
            'programming': {
                'beginner': ['html', 'css', 'javascript', 'python', 'sql'],
                'intermediate': ['react', 'node.js', 'java', 'c++', 'mongodb'],
                'advanced': ['tensorflow', 'pytorch', 'kubernetes', 'microservices', 'system design']
            },
            'data_science': {
                'beginner': ['excel', 'statistics', 'data analysis', 'pandas', 'numpy'],
                'intermediate': ['machine learning', 'scikit-learn', 'tableau', 'r', 'spark'],
                'advanced': ['deep learning', 'nlp', 'computer vision', 'mlops', 'big data']
            },
            'cloud': {
                'beginner': ['aws basics', 'cloud computing', 'linux', 'git'],
                'intermediate': ['aws', 'docker', 'ci/cd', 'terraform'],
                'advanced': ['kubernetes', 'serverless', 'cloud architecture', 'devops']
            },
            'business': {
                'beginner': ['communication', 'teamwork', 'project management'],
                'intermediate': ['leadership', 'strategy', 'product management'],
                'advanced': ['executive leadership', 'business strategy', 'stakeholder management']
            }
        }
        
        # Job roles and their requirements
        self.job_roles = {
            'Software Engineer': {
                'skills': ['programming'],
                'experience_range': (0, 15),
                'education_levels': ['bachelor', 'master'],
                'cognitive_patterns': ['Analytical Thinker', 'Detail Perfectionist', 'Adaptive Problem-Solver']
            },
            'Data Scientist': {
                'skills': ['data_science', 'programming'],
                'experience_range': (2, 12),
                'education_levels': ['master', 'phd'],
                'cognitive_patterns': ['Analytical Thinker', 'Creative Innovator', 'Strategic Planner']
            },
            'Product Manager': {
                'skills': ['business', 'programming'],
                'experience_range': (3, 15),
                'education_levels': ['bachelor', 'master', 'mba'],
                'cognitive_patterns': ['Strategic Planner', 'Collaborative Leader', 'Results-Driven Executor']
            },
            'DevOps Engineer': {
                'skills': ['cloud', 'programming'],
                'experience_range': (2, 12),
                'education_levels': ['bachelor', 'master'],
                'cognitive_patterns': ['Adaptive Problem-Solver', 'Detail Perfectionist', 'Results-Driven Executor']
            },
            'ML Engineer': {
                'skills': ['data_science', 'programming', 'cloud'],
                'experience_range': (3, 10),
                'education_levels': ['master', 'phd'],
                'cognitive_patterns': ['Analytical Thinker', 'Creative Innovator', 'Detail Perfectionist']
            }
        }
        
        # Cognitive patterns with characteristics
        self.cognitive_patterns = {
            'Analytical Thinker': {
                'keywords': ['analysis', 'data', 'research', 'metrics', 'statistics'],
                'skills_preference': ['data_science', 'programming'],
                'growth_rate': 1.2
            },
            'Creative Innovator': {
                'keywords': ['creative', 'design', 'innovative', 'prototype', 'ideation'],
                'skills_preference': ['programming', 'business'],
                'growth_rate': 1.4
            },
            'Strategic Planner': {
                'keywords': ['strategy', 'planning', 'vision', 'roadmap', 'architecture'],
                'skills_preference': ['business', 'cloud'],
                'growth_rate': 1.3
            },
            'Collaborative Leader': {
                'keywords': ['team', 'leadership', 'communication', 'mentoring', 'coordination'],
                'skills_preference': ['business'],
                'growth_rate': 1.5
            },
            'Detail Perfectionist': {
                'keywords': ['quality', 'precision', 'thorough', 'documentation', 'testing'],
                'skills_preference': ['programming', 'cloud'],
                'growth_rate': 1.1
            },
            'Adaptive Problem-Solver': {
                'keywords': ['flexible', 'agile', 'learning', 'versatile', 'problem-solving'],
                'skills_preference': ['programming', 'cloud'],
                'growth_rate': 1.3
            },
            'Results-Driven Executor': {
                'keywords': ['results', 'achievement', 'performance', 'delivery', 'execution'],
                'skills_preference': ['business', 'cloud'],
                'growth_rate': 1.4
            },
            'Empathetic Communicator': {
                'keywords': ['customer', 'user', 'stakeholder', 'communication', 'support'],
                'skills_preference': ['business'],
                'growth_rate': 1.2
            }
        }
        
        # Education levels
        self.education_levels = {
            'bachelor': ['Bachelor of Science', 'Bachelor of Engineering', 'Bachelor of Technology'],
            'master': ['Master of Science', 'Master of Engineering', 'Master of Technology'],
            'mba': ['Master of Business Administration'],
            'phd': ['Doctor of Philosophy', 'Ph.D. in Computer Science']
        }
        
        # Companies by tier
        self.companies = {
            'tier1': ['Google', 'Microsoft', 'Amazon', 'Apple', 'Meta', 'Netflix'],
            'tier2': ['Uber', 'Airbnb', 'Spotify', 'Adobe', 'Salesforce', 'Twitter'],
            'tier3': ['Startup Inc', 'TechCorp', 'DataSoft', 'CloudTech', 'InnovateLab']
        }
    
    def generate_resume_profile(self, target_role: Optional[str] = None) -> Dict:
        """Generate a single realistic resume profile"""
        
        # Select role (random if not specified)
        if target_role is None:
            role = random.choice(list(self.job_roles.keys()))
        else:
            role = target_role
        
        role_info = self.job_roles[role]
        
        # Generate basic info
        name = self.fake.name()
        email = f"{name.lower().replace(' ', '.')}@{self.fake.domain_name()}"
        phone = self.fake.phone_number()
        
        # Generate experience
        min_exp, max_exp = role_info['experience_range']
        experience_years = random.randint(min_exp, max_exp)
        
        # Select dominant cognitive pattern
        dominant_pattern = random.choice(role_info['cognitive_patterns'])
        pattern_info = self.cognitive_patterns[dominant_pattern]
        
        # Generate skills based on cognitive pattern and role
        skills = self._generate_skills(role_info['skills'], pattern_info['skills_preference'], experience_years)
        
        # Generate education
        education_level = random.choice(role_info['education_levels'])
        education = [random.choice(self.education_levels[education_level])]
        
        # Generate career history
        career_history = self._generate_career_history(role, experience_years, skills)
        
        # Generate projects
        projects = self._generate_projects(skills, dominant_pattern, experience_years)
        
        # Create resume text
        resume_text = self._create_resume_text(name, skills, education, career_history, projects, dominant_pattern)
        
        return {
            'name': name,
            'email': email,
            'phone': phone,
            'experience': str(experience_years),
            'skills': skills,
            'education': education,
            'current_role': career_history[-1]['title'] if career_history else role,
            'career_history': career_history,
            'projects': projects,
            'dominant_cognitive_pattern': dominant_pattern,
            'resume_text': resume_text,
            'target_role': role
        }
    
    def _generate_skills(self, role_skill_categories: List[str], 
                        pattern_preferences: List[str], experience_years: int) -> List[str]:
        """Generate realistic skill set based on role and experience"""
        skills = []
        
        # Determine skill level based on experience
        if experience_years <= 2:
            primary_level = 'beginner'
            secondary_level = 'beginner'
        elif experience_years <= 5:
            primary_level = 'intermediate'
            secondary_level = 'beginner'
        elif experience_years <= 8:
            primary_level = 'intermediate'
            secondary_level = 'intermediate'
        else:
            primary_level = 'advanced'
            secondary_level = 'intermediate'
        
        # Add skills from role requirements
        for category in role_skill_categories:
            if category in self.skill_categories:
                # Primary skills
                category_skills = self.skill_categories[category]
                skills.extend(random.sample(category_skills[primary_level], 
                                          min(3, len(category_skills[primary_level]))))
                
                # Secondary skills
                if secondary_level in category_skills:
                    skills.extend(random.sample(category_skills[secondary_level], 
                                              min(2, len(category_skills[secondary_level]))))
        
        # Add skills based on cognitive pattern preferences
        for pref_category in pattern_preferences:
            if pref_category in self.skill_categories and pref_category not in role_skill_categories:
                category_skills = self.skill_categories[pref_category]
                skills.extend(random.sample(category_skills['beginner'], 
                                          min(2, len(category_skills['beginner']))))
        
        # Remove duplicates and return
        return list(set(skills))
    
    def _generate_career_history(self, target_role: str, total_experience: int, skills: List[str]) -> List[Dict]:
        """Generate realistic career progression"""
        if total_experience == 0:
            return []
        
        career_history = []
        current_experience = 0
        
        # Entry level position
        if total_experience > 0:
            entry_duration = min(2, total_experience)
            entry_role = f"Junior {target_role}" if total_experience > 2 else target_role
            
            career_history.append({
                'title': entry_role,
                'company': random.choice(self.companies['tier3']),
                'duration_years': entry_duration,
                'start_year': 2024 - total_experience,
                'end_year': 2024 - total_experience + entry_duration,
                'technologies': skills[:3],
                'achievements': self._generate_achievements(entry_role, entry_duration)
            })
            
            current_experience += entry_duration
        
        # Mid-level positions
        while current_experience < total_experience:
            remaining_years = total_experience - current_experience
            duration = min(random.randint(2, 4), remaining_years)
            
            if remaining_years <= 2:
                # Current role
                role_title = f"Senior {target_role}" if total_experience > 5 else target_role
                company_tier = 'tier1' if total_experience > 8 else 'tier2'
            else:
                role_title = target_role
                company_tier = 'tier2' if total_experience > 5 else 'tier3'
            
            career_history.append({
                'title': role_title,
                'company': random.choice(self.companies[company_tier]),
                'duration_years': duration,
                'start_year': 2024 - total_experience + current_experience,
                'end_year': 2024 - total_experience + current_experience + duration,
                'technologies': random.sample(skills, min(5, len(skills))),
                'achievements': self._generate_achievements(role_title, duration)
            })
            
            current_experience += duration
        
        return career_history
    
    def _generate_achievements(self, role: str, duration: int) -> List[str]:
        """Generate role-appropriate achievements"""
        achievement_templates = {
            'Software Engineer': [
                "Developed {feature} that improved {metric} by {percentage}%",
                "Led team of {team_size} engineers to deliver {project}",
                "Optimized {system} resulting in {improvement}",
                "Implemented {technology} reducing {problem} by {percentage}%"
            ],
            'Data Scientist': [
                "Built ML model with {accuracy}% accuracy for {use_case}",
                "Analyzed {data_size} of data to identify {insight}",
                "Developed predictive model that increased {metric} by {percentage}%",
                "Led data science initiative resulting in ${savings}K cost savings"
            ],
            'Product Manager': [
                "Launched {product} that acquired {users}K users in {timeframe}",
                "Increased user engagement by {percentage}% through {strategy}",
                "Led cross-functional team of {team_size} to deliver {milestone}",
                "Defined product roadmap resulting in {revenue}% revenue growth"
            ]
        }
        
        # Find matching templates
        templates = []
        for key, template_list in achievement_templates.items():
            if key.lower() in role.lower():
                templates = template_list
                break
        
        if not templates:
            templates = achievement_templates['Software Engineer']  # Default
        
        # Generate 2-4 achievements
        num_achievements = min(duration + 1, 4)
        achievements = []
        
        for _ in range(num_achievements):
            template = random.choice(templates)
            
            # Fill in template variables
            achievement = template.format(
                feature=random.choice(['API', 'dashboard', 'algorithm', 'system']),
                metric=random.choice(['performance', 'efficiency', 'accuracy', 'speed']),
                percentage=random.randint(10, 50),
                team_size=random.randint(3, 12),
                project=random.choice(['mobile app', 'web platform', 'data pipeline', 'ML system']),
                system=random.choice(['database', 'API', 'frontend', 'backend']),
                improvement=random.choice(['40% faster response time', '60% cost reduction', '30% better accuracy']),
                technology=random.choice(['microservices', 'cloud architecture', 'ML pipeline', 'automation']),
                problem=random.choice(['latency', 'errors', 'costs', 'manual work']),
                accuracy=random.randint(85, 98),
                use_case=random.choice(['fraud detection', 'recommendation system', 'demand forecasting']),
                data_size=random.choice(['1TB', '500GB', '2PB', '100GB']),
                insight=random.choice(['customer behavior patterns', 'market trends', 'optimization opportunities']),
                savings=random.randint(50, 500),
                product=random.choice(['mobile app', 'web platform', 'API service', 'analytics tool']),
                users=random.randint(10, 1000),
                timeframe=random.choice(['6 months', '1 year', '3 months']),
                strategy=random.choice(['UX improvements', 'feature enhancements', 'personalization']),
                milestone=random.choice(['product launch', 'system migration', 'feature rollout']),
                revenue=random.randint(15, 40)
            )
            
            achievements.append(achievement)
        
        return achievements
    
    def _generate_projects(self, skills: List[str], cognitive_pattern: str, experience: int) -> List[Dict]:
        """Generate relevant project portfolio"""
        projects = []
        num_projects = min(experience // 2 + 1, 5)
        
        project_templates = {
            'Analytical Thinker': [
                'Data Analysis Dashboard for Business Intelligence',
                'Predictive Analytics Model for Customer Behavior',
                'Statistical Analysis Tool for Market Research'
            ],
            'Creative Innovator': [
                'AI-Powered Creative Content Generator',
                'Innovative Mobile App with Unique UX',
                'Prototype for Next-Generation User Interface'
            ],
            'Strategic Planner': [
                'Enterprise Architecture Migration Strategy',
                'Long-term Technology Roadmap Implementation',
                'Strategic Data Platform Design'
            ]
        }
        
        # Get pattern-specific templates or use default
        templates = project_templates.get(cognitive_pattern, project_templates['Analytical Thinker'])
        
        for i in range(num_projects):
            project_name = random.choice(templates) if templates else f"Project {i+1}"
            
            # Select relevant technologies
            project_skills = random.sample(skills, min(4, len(skills)))
            
            projects.append({
                'name': project_name,
                'description': f"Developed {project_name.lower()} using {', '.join(project_skills[:2])}",
                'technologies': project_skills,
                'duration_months': random.randint(2, 12),
                'impact': self._generate_project_impact(),
                'complexity_score': random.uniform(0.3, 0.9)
            })
        
        return projects
    
    def _generate_project_impact(self) -> str:
        """Generate project impact description"""
        impacts = [
            "Improved system performance by 40%",
            "Reduced processing time by 60%",
            "Increased user engagement by 25%",
            "Saved $50K in operational costs",
            "Processed 1M+ data points daily",
            "Served 100K+ active users",
            "Achieved 95% accuracy in predictions",
            "Reduced manual work by 80%"
        ]
        return random.choice(impacts)
    
    def _create_resume_text(self, name: str, skills: List[str], education: List[str], 
                           career_history: List[Dict], projects: List[Dict], 
                           cognitive_pattern: str) -> str:
        """Create comprehensive resume text"""
        
        # Add cognitive pattern keywords naturally
        pattern_keywords = self.cognitive_patterns[cognitive_pattern]['keywords']
        
        text_parts = [
            f"Name: {name}",
            f"Skills: {', '.join(skills)}",
            f"Education: {', '.join(education)}"
        ]
        
        # Add career summary with cognitive keywords
        if career_history:
            summary_keywords = random.sample(pattern_keywords, min(3, len(pattern_keywords)))
            summary = f"Experienced professional with strong {summary_keywords[0]} and {summary_keywords[1]} skills. "
            summary += f"Demonstrated expertise in {summary_keywords[2]} and delivering high-quality solutions."
            text_parts.append(f"Summary: {summary}")
        
        # Add work experience
        if career_history:
            text_parts.append("Work Experience:")
            for job in career_history:
                job_text = f"{job['title']} at {job['company']} ({job['start_year']}-{job['end_year']}). "
                job_text += f"Technologies: {', '.join(job['technologies'])}. "
                job_text += f"Achievements: {'. '.join(job['achievements'][:2])}."
                text_parts.append(job_text)
        
        # Add projects
        if projects:
            text_parts.append("Projects:")
            for project in projects:
                project_text = f"{project['name']}: {project['description']}. "
                project_text += f"Impact: {project['impact']}."
                text_parts.append(project_text)
        
        return "\n".join(text_parts)
    
    def generate_job_requirements(self, role: str) -> Dict:
        """Generate realistic job requirements for a role"""
        role_info = self.job_roles.get(role, self.job_roles['Software Engineer'])
        
        # Required skills
        required_skills = []
        for skill_category in role_info['skills']:
            if skill_category in self.skill_categories:
                category_skills = self.skill_categories[skill_category]
                required_skills.extend(random.sample(
                    category_skills['beginner'] + category_skills['intermediate'], 
                    min(4, len(category_skills['beginner']) + len(category_skills['intermediate']))
                ))
        
        # Experience requirements
        min_exp, max_exp = role_info['experience_range']
        required_experience = random.randint(min_exp, min(max_exp, min_exp + 5))
        
        # Cognitive requirements
        cognitive_requirements = {}
        for pattern in role_info['cognitive_patterns']:
            cognitive_requirements[pattern] = random.uniform(0.6, 1.0)
        
        return {
            'role': role,
            'required_skills': required_skills[:5],  # Limit to 5 skills
            'min_experience': required_experience,
            'education_level': random.choice(role_info['education_levels']),
            'role_level': self._determine_role_level(required_experience),
            'cognitive_requirements': cognitive_requirements,
            'innovation_level': random.uniform(0.3, 0.8),
            'domain': random.choice(['fintech', 'healthcare', 'e-commerce', 'saas', 'gaming'])
        }
    
    def _determine_role_level(self, experience: int) -> str:
        """Determine role level based on experience"""
        if experience <= 2:
            return 'junior'
        elif experience <= 5:
            return 'mid'
        elif experience <= 8:
            return 'senior'
        else:
            return 'lead'
    
    def generate_training_dataset(self, num_samples: int = 10000) -> Tuple[List[Dict], List[Dict], List[float]]:
        """Generate complete training dataset with resume-job pairs and match scores"""
        logger.info(f"Generating {num_samples} training samples...")
        
        resumes = []
        job_requirements = []
        match_scores = []
        
        roles = list(self.job_roles.keys())
        
        for i in range(num_samples):
            if i % 1000 == 0:
                logger.info(f"Generated {i}/{num_samples} samples")
            
            # Generate resume
            target_role = random.choice(roles)
            resume = self.generate_resume_profile(target_role)
            
            # Generate job requirement (70% matching role, 30% different role)
            if random.random() < 0.7:
                job_role = target_role  # Matching role
                base_match_score = random.uniform(0.6, 0.95)
            else:
                job_role = random.choice([r for r in roles if r != target_role])  # Different role
                base_match_score = random.uniform(0.2, 0.6)
            
            job_req = self.generate_job_requirements(job_role)
            
            # Calculate realistic match score
            match_score = self._calculate_realistic_match_score(resume, job_req, base_match_score)
            
            resumes.append(resume)
            job_requirements.append(job_req)
            match_scores.append(match_score)
        
        logger.info(f"Generated {num_samples} training samples successfully!")
        return resumes, job_requirements, match_scores
    
    def _calculate_realistic_match_score(self, resume: Dict, job_req: Dict, base_score: float) -> float:
        """Calculate realistic match score with some noise"""
        
        # Skill match factor
        resume_skills = [skill.lower() for skill in resume['skills']]
        required_skills = [skill.lower() for skill in job_req['required_skills']]
        
        skill_matches = sum(1 for skill in required_skills if skill in resume_skills)
        skill_factor = skill_matches / len(required_skills) if required_skills else 0.5
        
        # Experience factor
        resume_exp = int(resume['experience'])
        required_exp = job_req['min_experience']
        exp_factor = min(resume_exp / max(required_exp, 1), 2.0) / 2.0  # Normalize to 0-1
        
        # Cognitive pattern alignment
        resume_pattern = resume['dominant_cognitive_pattern']
        cognitive_factor = job_req['cognitive_requirements'].get(resume_pattern, 0.5)
        
        # Weighted combination
        calculated_score = (
            skill_factor * 0.4 +
            exp_factor * 0.3 +
            cognitive_factor * 0.2 +
            base_score * 0.1
        )
        
        # Add some realistic noise
        noise = random.uniform(-0.1, 0.1)
        final_score = max(0.0, min(1.0, calculated_score + noise))
        
        return final_score
    
    def save_dataset(self, resumes: List[Dict], job_requirements: List[Dict], 
                    match_scores: List[float], output_dir: str = "data"):
        """Save generated dataset to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as JSON
        dataset = {
            'resumes': resumes,
            'job_requirements': job_requirements,
            'match_scores': match_scores,
            'metadata': {
                'num_samples': len(resumes),
                'generated_at': datetime.now().isoformat(),
                'cognitive_patterns': list(self.cognitive_patterns.keys()),
                'job_roles': list(self.job_roles.keys())
            }
        }
        
        with open(os.path.join(output_dir, 'synthetic_training_data.json'), 'w') as f:
            json.dump(dataset, f, indent=2)
        
        # Save as CSV for easy analysis
        df_data = []
        for resume, job_req, score in zip(resumes, job_requirements, match_scores):
            row = {
                'name': resume['name'],
                'experience_years': resume['experience'],
                'skills_count': len(resume['skills']),
                'education_level': resume['education'][0] if resume['education'] else '',
                'cognitive_pattern': resume['dominant_cognitive_pattern'],
                'target_role': resume['target_role'],
                'job_role': job_req['role'],
                'required_experience': job_req['min_experience'],
                'match_score': score,
                'is_match': 1 if score >= 0.7 else 0
            }
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        df.to_csv(os.path.join(output_dir, 'synthetic_training_data.csv'), index=False)
        
        logger.info(f"Dataset saved to {output_dir}")
        logger.info(f"Match rate: {(df['is_match'].sum() / len(df) * 100):.1f}%")

# Example usage and testing
if __name__ == "__main__":
    generator = SyntheticDataGenerator()
    
    # Generate small sample for testing
    resumes, job_reqs, scores = generator.generate_training_dataset(100)
    
    # Save dataset
    generator.save_dataset(resumes, job_reqs, scores)
    
    # Print sample
    print("Sample Resume:")
    print(f"Name: {resumes[0]['name']}")
    print(f"Role: {resumes[0]['target_role']}")
    print(f"Skills: {resumes[0]['skills']}")
    print(f"Pattern: {resumes[0]['dominant_cognitive_pattern']}")
    print(f"Match Score: {scores[0]:.3f}")
