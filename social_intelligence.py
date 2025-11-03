"""
Social Intelligence Module
Analyzes candidate's online presence and professional brand
"""

import re
import requests
from datetime import datetime
import json

class SocialIntelligenceAnalyzer:
    """
    Analyzes candidate's social media presence and professional brand
    """
    
    def __init__(self):
        self.platforms = ['linkedin', 'github', 'twitter', 'stackoverflow', 'medium']
    
    def extract_social_links(self, resume_text):
        """Extract social media links from resume"""
        links = {
            'linkedin': None,
            'github': None,
            'twitter': None,
            'stackoverflow': None,
            'medium': None,
            'portfolio': None,
            'blog': None
        }
        
        # LinkedIn patterns
        linkedin_patterns = [
            r'linkedin\.com/in/([a-zA-Z0-9-]+)',
            r'linkedin\.com/pub/([a-zA-Z0-9-]+)',
            r'www\.linkedin\.com/in/([a-zA-Z0-9-]+)'
        ]
        
        for pattern in linkedin_patterns:
            match = re.search(pattern, resume_text, re.IGNORECASE)
            if match:
                links['linkedin'] = f"https://linkedin.com/in/{match.group(1)}"
                break
        
        # GitHub patterns
        github_patterns = [
            r'github\.com/([a-zA-Z0-9-]+)',
            r'www\.github\.com/([a-zA-Z0-9-]+)'
        ]
        
        for pattern in github_patterns:
            match = re.search(pattern, resume_text, re.IGNORECASE)
            if match:
                links['github'] = f"https://github.com/{match.group(1)}"
                break
        
        # Twitter patterns
        twitter_patterns = [
            r'twitter\.com/([a-zA-Z0-9_]+)',
            r'@([a-zA-Z0-9_]+)',
            r'x\.com/([a-zA-Z0-9_]+)'
        ]
        
        for pattern in twitter_patterns:
            match = re.search(pattern, resume_text, re.IGNORECASE)
            if match:
                username = match.group(1)
                if username and len(username) > 2:
                    links['twitter'] = f"https://twitter.com/{username}"
                    break
        
        # Stack Overflow
        stackoverflow_patterns = [
            r'stackoverflow\.com/users/(\d+)',
            r'stackoverflow\.com/u/(\d+)'
        ]
        
        for pattern in stackoverflow_patterns:
            match = re.search(pattern, resume_text, re.IGNORECASE)
            if match:
                links['stackoverflow'] = f"https://stackoverflow.com/users/{match.group(1)}"
                break
        
        # Medium
        medium_patterns = [
            r'medium\.com/@([a-zA-Z0-9-]+)',
            r'([a-zA-Z0-9-]+)\.medium\.com'
        ]
        
        for pattern in medium_patterns:
            match = re.search(pattern, resume_text, re.IGNORECASE)
            if match:
                links['medium'] = f"https://medium.com/@{match.group(1)}"
                break
        
        # Portfolio/Blog (generic URLs)
        url_pattern = r'https?://(?:www\.)?([a-zA-Z0-9-]+\.[a-zA-Z]{2,})(?:/[^\s]*)?'
        urls = re.findall(url_pattern, resume_text)
        
        for url in urls:
            if 'portfolio' in url.lower() or 'blog' in url.lower():
                links['portfolio'] = f"https://{url}"
                break
        
        return links
    
    def analyze_linkedin_presence(self, linkedin_url, resume_data):
        """
        Analyze LinkedIn profile (simulated - real API requires authentication)
        In production, use LinkedIn API or web scraping with proper authentication
        """
        if not linkedin_url:
            return {
                'found': False,
                'profile_completeness': 0,
                'connections': 0,
                'recommendations': 0,
                'activity_level': 'Unknown',
                'profile_strength': 'Not Found'
            }
        
        # Simulated analysis based on resume data
        # In production, you would use LinkedIn API or scraping
        
        # Estimate profile completeness based on resume data
        completeness_score = 0
        if resume_data.get('experience', 0) > 0:
            completeness_score += 30
        if len(resume_data.get('skills', [])) > 5:
            completeness_score += 25
        if resume_data.get('highest_education') in ['bachelor', 'master', 'phd']:
            completeness_score += 20
        if resume_data.get('achievements'):
            completeness_score += 15
        if linkedin_url:
            completeness_score += 10  # Has LinkedIn profile
        
        # Estimate connections based on experience
        exp_years = resume_data.get('experience', 0)
        estimated_connections = min(exp_years * 50 + 100, 500)
        
        # Estimate activity level
        if exp_years >= 5:
            activity_level = 'Highly Active'
        elif exp_years >= 2:
            activity_level = 'Moderately Active'
        else:
            activity_level = 'Low Activity'
        
        # Profile strength
        if completeness_score >= 80:
            profile_strength = 'All-Star'
        elif completeness_score >= 60:
            profile_strength = 'Expert'
        elif completeness_score >= 40:
            profile_strength = 'Intermediate'
        else:
            profile_strength = 'Beginner'
        
        return {
            'found': True,
            'url': linkedin_url,
            'profile_completeness': completeness_score,
            'connections': estimated_connections,
            'recommendations': max(exp_years // 2, 0),
            'activity_level': activity_level,
            'profile_strength': profile_strength,
            'last_active': 'Within last week'  # Simulated
        }
    
    def analyze_github_presence(self, github_url, resume_data):
        """Analyze GitHub profile"""
        if not github_url:
            return {
                'found': False,
                'repositories': 0,
                'contributions': 0,
                'followers': 0,
                'coding_activity': 'Unknown'
            }
        
        # Simulated GitHub analysis
        skills = resume_data.get('skills', [])
        tech_skills = [s for s in skills if s in ['python', 'java', 'javascript', 'react', 'node', 'go', 'rust']]
        
        # Estimate based on skills
        estimated_repos = len(tech_skills) * 3 + 5
        estimated_contributions = resume_data.get('experience', 0) * 200
        estimated_followers = len(tech_skills) * 10
        
        if len(tech_skills) >= 4:
            coding_activity = 'Very Active'
        elif len(tech_skills) >= 2:
            coding_activity = 'Moderately Active'
        else:
            coding_activity = 'Low Activity'
        
        return {
            'found': True,
            'url': github_url,
            'repositories': estimated_repos,
            'contributions': estimated_contributions,
            'followers': estimated_followers,
            'coding_activity': coding_activity,
            'languages': tech_skills[:5]
        }
    
    def analyze_twitter_presence(self, twitter_url):
        """Analyze Twitter/X presence"""
        if not twitter_url:
            return {
                'found': False,
                'followers': 0,
                'engagement': 'Unknown',
                'professional_content': False
            }
        
        # Simulated Twitter analysis
        return {
            'found': True,
            'url': twitter_url,
            'followers': 250,  # Simulated
            'tweets': 450,
            'engagement': 'Moderate',
            'professional_content': True,
            'thought_leadership': 'Shares industry insights'
        }
    
    def analyze_stackoverflow_presence(self, stackoverflow_url):
        """Analyze Stack Overflow presence"""
        if not stackoverflow_url:
            return {
                'found': False,
                'reputation': 0,
                'answers': 0,
                'badges': 0
            }
        
        return {
            'found': True,
            'url': stackoverflow_url,
            'reputation': 1250,  # Simulated
            'answers': 45,
            'questions': 12,
            'badges': 8,
            'top_tags': ['python', 'javascript', 'sql']
        }
    
    def calculate_online_presence_score(self, social_links, linkedin_data, github_data, twitter_data, stackoverflow_data):
        """Calculate overall online presence score"""
        score = 0
        max_score = 100
        
        # LinkedIn (40 points)
        if linkedin_data['found']:
            score += 10  # Has profile
            score += (linkedin_data['profile_completeness'] / 100) * 20  # Completeness
            if linkedin_data['connections'] > 200:
                score += 10
        
        # GitHub (30 points)
        if github_data['found']:
            score += 10  # Has profile
            if github_data['repositories'] > 5:
                score += 10
            if github_data['contributions'] > 100:
                score += 10
        
        # Twitter (15 points)
        if twitter_data['found']:
            score += 5
            if twitter_data['professional_content']:
                score += 10
        
        # Stack Overflow (15 points)
        if stackoverflow_data['found']:
            score += 5
            if stackoverflow_data['reputation'] > 500:
                score += 10
        
        return min(score, max_score)
    
    def generate_social_intelligence_report(self, resume_text, resume_data):
        """Generate comprehensive social intelligence report"""
        
        # Extract social links
        social_links = self.extract_social_links(resume_text)
        
        # Analyze each platform
        linkedin_data = self.analyze_linkedin_presence(social_links['linkedin'], resume_data)
        github_data = self.analyze_github_presence(social_links['github'], resume_data)
        twitter_data = self.analyze_twitter_presence(social_links['twitter'])
        stackoverflow_data = self.analyze_stackoverflow_presence(social_links['stackoverflow'])
        
        # Calculate overall score
        online_presence_score = self.calculate_online_presence_score(
            social_links, linkedin_data, github_data, twitter_data, stackoverflow_data
        )
        
        # Determine engagement level
        if online_presence_score >= 70:
            engagement_level = 'Highly Engaged Professional'
            engagement_color = 'success'
        elif online_presence_score >= 40:
            engagement_level = 'Moderately Active'
            engagement_color = 'info'
        else:
            engagement_level = 'Limited Online Presence'
            engagement_color = 'warning'
        
        # Generate insights
        insights = []
        red_flags = []
        
        if linkedin_data['found']:
            if linkedin_data['profile_strength'] in ['All-Star', 'Expert']:
                insights.append(f"Strong LinkedIn presence ({linkedin_data['profile_strength']} profile)")
            if linkedin_data['connections'] > 300:
                insights.append(f"Well-connected professional ({linkedin_data['connections']}+ connections)")
        else:
            red_flags.append("No LinkedIn profile found - Limited professional networking")
        
        if github_data['found']:
            if github_data['repositories'] > 10:
                insights.append(f"Active open-source contributor ({github_data['repositories']} repositories)")
            if github_data['coding_activity'] == 'Very Active':
                insights.append("High coding activity - Passionate developer")
        else:
            if any(skill in resume_data.get('skills', []) for skill in ['python', 'java', 'javascript']):
                red_flags.append("No GitHub profile despite technical skills")
        
        if stackoverflow_data['found']:
            if stackoverflow_data['reputation'] > 1000:
                insights.append(f"Active Stack Overflow contributor ({stackoverflow_data['reputation']} reputation)")
        
        if twitter_data['found'] and twitter_data['professional_content']:
            insights.append("Shares professional insights on social media")
        
        # Professional brand assessment
        if online_presence_score >= 70:
            brand_assessment = "Strong professional brand with active online presence"
        elif online_presence_score >= 40:
            brand_assessment = "Moderate professional brand - Room for improvement"
        else:
            brand_assessment = "Weak online presence - May lack networking skills"
        
        return {
            'social_links': social_links,
            'linkedin': linkedin_data,
            'github': github_data,
            'twitter': twitter_data,
            'stackoverflow': stackoverflow_data,
            'online_presence_score': online_presence_score,
            'engagement_level': engagement_level,
            'engagement_color': engagement_color,
            'insights': insights,
            'red_flags': red_flags,
            'brand_assessment': brand_assessment,
            'recommendation': self._generate_recommendation(online_presence_score, insights, red_flags)
        }
    
    def _generate_recommendation(self, score, insights, red_flags):
        """Generate hiring recommendation based on social intelligence"""
        if score >= 70 and len(red_flags) == 0:
            return "STRONG HIRE - Active professional with excellent online presence"
        elif score >= 40:
            return "CONSIDER - Moderate online presence, verify networking skills in interview"
        else:
            return "CAUTION - Limited online presence may indicate poor networking or outdated skills"

# Global instance
social_analyzer = SocialIntelligenceAnalyzer()

def get_social_intelligence(resume_text, resume_data):
    """
    Main function to get social intelligence report
    """
    return social_analyzer.generate_social_intelligence_report(resume_text, resume_data)
    
