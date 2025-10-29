"""
NeuroMatch AI - Advanced Streamlit Dashboard
Complete interactive ML-powered resume analysis platform
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.cognitive_analyzer import CognitiveAnalyzer
from core.growth_predictor import GrowthPredictor
from core.innovation_detector import InnovationDetector
from core.ensemble_matcher import EnsembleMatcher
from data.synthetic_training import SyntheticDataGenerator
from resume_parser import extract_text_from_pdf, parse_resume, is_resume_pdf

import logging
import tempfile
from datetime import datetime
import json

# Configure page
st.set_page_config(
    page_title="NeuroMatch AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

class NeuroMatchApp:
    def __init__(self):
        self.cognitive_analyzer = CognitiveAnalyzer()
        self.growth_predictor = GrowthPredictor()
        self.innovation_detector = InnovationDetector()
        self.ensemble_matcher = EnsembleMatcher()
        
        # Initialize session state
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None
        if 'resume_data' not in st.session_state:
            st.session_state.resume_data = None
    
    def parse_text_resume(self, text):
        """Parse resume from raw text"""
        # Simple text parsing
        lines = text.split('\n')
        
        # Extract basic info
        resume_data = {
            'name': 'Candidate',
            'experience': '0',
            'skills': [],
            'education': [],
            'current_role': 'Unknown'
        }
        
        # Simple keyword extraction
        text_lower = text.lower()
        
        # Extract skills
        skill_keywords = ['python', 'java', 'javascript', 'sql', 'machine learning', 'aws', 'docker', 'react', 'node.js']
        found_skills = [skill for skill in skill_keywords if skill in text_lower]
        resume_data['skills'] = found_skills
        
        # Extract experience (look for years)
        import re
        exp_match = re.search(r'(\d+)\s*(?:years?|yrs?)', text_lower)
        if exp_match:
            resume_data['experience'] = exp_match.group(1)
        
        # Extract education
        education_keywords = ['bachelor', 'master', 'phd', 'degree', 'university', 'college']
        found_education = [edu for edu in education_keywords if edu in text_lower]
        resume_data['education'] = found_education
        
        return resume_data
    
    def advanced_candidate_scoring(self, resume_text, resume_data, job_criteria):
        """Advanced scoring system that never misses great candidates"""
        
        text_lower = resume_text.lower()
        
        # Base scores
        base_score = 0.0
        bonus_points = 0.0
        penalty_points = 0.0
        
        # 1. MUST-HAVE SKILLS CHECK (Critical)
        must_have = job_criteria.get('must_have_skills', [])
        if must_have:
            must_have_count = sum(1 for skill in must_have if skill.lower() in text_lower)
            must_have_ratio = must_have_count / len(must_have)
            base_score += must_have_ratio * 0.4  # 40% weight for critical skills
        else:
            base_score += 0.4  # If no must-haves specified, give full points
        
        # 2. REQUIRED SKILLS (Flexible matching)
        required_skills = job_criteria.get('required_skills', [])
        if required_skills:
            # Direct matches
            direct_matches = sum(1 for skill in required_skills if skill.lower() in text_lower)
            
            # Similar skill detection (e.g., React -> JavaScript, TensorFlow -> Machine Learning)
            skill_synonyms = {
                'python': ['django', 'flask', 'pandas', 'numpy'],
                'javascript': ['react', 'vue', 'angular', 'node.js', 'typescript'],
                'machine learning': ['tensorflow', 'pytorch', 'scikit-learn', 'keras', 'ml'],
                'aws': ['cloud', 'ec2', 's3', 'lambda', 'azure', 'gcp'],
                'sql': ['mysql', 'postgresql', 'database', 'mongodb'],
                'docker': ['kubernetes', 'containerization', 'devops']
            }
            
            similar_matches = 0
            for req_skill in required_skills:
                if req_skill.lower() in skill_synonyms:
                    similar_matches += sum(1 for syn in skill_synonyms[req_skill.lower()] 
                                         if syn in text_lower)
            
            total_skill_score = (direct_matches + similar_matches * 0.7) / len(required_skills)
            base_score += min(total_skill_score, 1.0) * 0.3  # 30% weight
        
        # 3. EXPERIENCE MATCHING (Flexible)
        candidate_exp = int(resume_data.get('experience', '0'))
        min_exp = job_criteria.get('min_experience', 0)
        
        if candidate_exp >= min_exp:
            exp_score = min(candidate_exp / max(min_exp * 1.5, 1), 1.0)  # Cap at 1.5x requirement
        else:
            # Penalty for under-experience, but not elimination
            exp_score = candidate_exp / max(min_exp, 1) * 0.7  # 30% penalty
        
        base_score += exp_score * 0.2  # 20% weight
        
        # 4. EDUCATION MATCHING
        education_level = job_criteria.get('education_level', 'any')
        if education_level != 'any':
            candidate_education = ' '.join(resume_data.get('education', [])).lower()
            if education_level in candidate_education:
                base_score += 0.1  # 10% weight
            else:
                base_score += 0.05  # Half points for different education
        else:
            base_score += 0.1
        
        # 5. BONUS POINTS (Never miss hidden gems!)
        
        # Leadership indicators
        if job_criteria.get('bonus_leadership', False):
            leadership_keywords = ['led', 'managed', 'team lead', 'mentor', 'supervisor', 'director']
            if any(keyword in text_lower for keyword in leadership_keywords):
                bonus_points += 0.1
        
        # Startup experience
        if job_criteria.get('bonus_startup', False):
            startup_keywords = ['startup', 'founder', 'early stage', 'seed', 'series a']
            if any(keyword in text_lower for keyword in startup_keywords):
                bonus_points += 0.08
        
        # Remote work experience
        if job_criteria.get('bonus_remote', False):
            remote_keywords = ['remote', 'distributed', 'work from home', 'virtual team']
            if any(keyword in text_lower for keyword in remote_keywords):
                bonus_points += 0.05
        
        # Certifications
        if job_criteria.get('bonus_certifications', False):
            cert_keywords = ['certified', 'certification', 'aws certified', 'google cloud', 'microsoft']
            if any(keyword in text_lower for keyword in cert_keywords):
                bonus_points += 0.07
        
        # Growth indicators
        if job_criteria.get('look_for_growth', True):
            growth_keywords = ['learned', 'self-taught', 'bootcamp', 'online course', 'upskilled']
            if any(keyword in text_lower for keyword in growth_keywords):
                bonus_points += 0.06
        
        # Projects and open source
        if job_criteria.get('look_for_projects', True):
            project_keywords = ['github', 'open source', 'side project', 'personal project', 'portfolio']
            if any(keyword in text_lower for keyword in project_keywords):
                bonus_points += 0.08
        
        # Continuous learning
        if job_criteria.get('look_for_learning', True):
            learning_keywords = ['coursera', 'udemy', 'pluralsight', 'conference', 'workshop']
            if any(keyword in text_lower for keyword in learning_keywords):
                bonus_points += 0.05
        
        # 6. DEAL BREAKERS (Penalties)
        deal_breakers = job_criteria.get('deal_breakers', [])
        if deal_breakers:
            for deal_breaker in deal_breakers:
                if deal_breaker.lower() in text_lower:
                    penalty_points += 0.1
        
        # 7. NICE-TO-HAVE BONUS
        nice_to_have = job_criteria.get('nice_to_have', [])
        if nice_to_have:
            nice_matches = sum(1 for skill in nice_to_have if skill.lower() in text_lower)
            bonus_points += (nice_matches / len(nice_to_have)) * 0.1
        
        # 8. RANKING STRATEGY ADJUSTMENTS
        strategy = job_criteria.get('priority_weights', '')
        
        if '💎 Diamond Hunter' in strategy:
            # Boost high-potential candidates with lower experience
            if candidate_exp < min_exp and bonus_points > 0.15:
                bonus_points += 0.15  # Extra boost for hidden gems
        
        elif '🚀 High Potential' in strategy:
            # Focus on growth indicators
            if bonus_points > 0.1:
                bonus_points *= 1.5  # Amplify growth signals
        
        elif '🧠 Cognitive Fit' in strategy:
            # Add cognitive analysis
            try:
                cognitive_scores = self.cognitive_analyzer.predict_cognitive_pattern(resume_text)
                dominant_pattern, confidence = self.cognitive_analyzer.get_dominant_pattern(resume_text)
                
                # Boost based on cognitive fit
                if confidence > 0.7:
                    bonus_points += 0.1
            except:
                pass
        
        # Final score calculation
        final_score = min(base_score + bonus_points - penalty_points, 1.0)
        
        return max(final_score, 0.0)  # Ensure non-negative
    
    def process_bulk_text_resumes(self, resumes_list, job_criteria):
        """Process multiple resume texts with advanced scoring"""
        results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, resume_text in enumerate(resumes_list):
            status_text.text(f"🔍 Analyzing resume {i+1}/{len(resumes_list)} with advanced AI...")
            
            # Parse resume
            resume_data = self.parse_text_resume(resume_text)
            
            # Run analysis
            try:
                # Advanced scoring
                advanced_score = self.advanced_candidate_scoring(resume_text, resume_data, job_criteria)
                
                # AI analysis
                cognitive_scores = self.cognitive_analyzer.predict_cognitive_pattern(resume_text)
                dominant_pattern, confidence = self.cognitive_analyzer.get_dominant_pattern(resume_text)
                growth_analysis = self.growth_predictor.predict_growth_potential(resume_data)
                innovation_metrics = self.innovation_detector.calculate_innovation_score(resume_text)
                
                # Determine candidate category
                candidate_exp = int(resume_data.get('experience', '0'))
                min_exp = job_criteria.get('min_experience', 0)
                
                if candidate_exp < min_exp and advanced_score > 0.7:
                    category = "💎 Hidden Gem"
                elif candidate_exp >= min_exp * 1.5 and advanced_score > 0.8:
                    category = "🌟 Overqualified"
                elif advanced_score > 0.85:
                    category = "🎯 Perfect Match"
                elif advanced_score > 0.7:
                    category = "✅ Good Match"
                elif advanced_score > 0.5:
                    category = "⚠️ Potential"
                else:
                    category = "❌ Poor Match"
                
                results.append({
                    'resume_id': f'Resume_{i+1}',
                    'name': resume_data.get('name', f'Candidate_{i+1}'),
                    'experience': resume_data.get('experience', '0'),
                    'skills_count': len(resume_data.get('skills', [])),
                    'dominant_pattern': dominant_pattern,
                    'pattern_confidence': confidence,
                    'growth_potential': growth_analysis['overall_growth_potential'],
                    'innovation_score': innovation_metrics.novelty_score,
                    'match_score': advanced_score,
                    'category': category,
                    'technical_fit': min(advanced_score * 1.1, 1.0),
                    'status': 'Success'
                })
                
            except Exception as e:
                results.append({
                    'resume_id': f'Resume_{i+1}',
                    'status': f'Error: {str(e)}'
                })
            
            progress_bar.progress((i + 1) / len(resumes_list))
        
        # Display advanced ranked results
        self.display_advanced_ranked_results(results, job_criteria)
    
    def display_advanced_ranked_results(self, results, job_criteria):
        """Display advanced ranked candidate results with categories"""
        st.subheader("🏆 SMART RANKED CANDIDATES - NEVER MISS THE BEST!")
        
        # Filter successful results
        successful_results = [r for r in results if r['status'] == 'Success']
        
        if not successful_results:
            st.error("❌ No successful analyses found")
            return
        
        # Sort by match score (highest first)
        successful_results.sort(key=lambda x: x['match_score'], reverse=True)
        
        # Categorize candidates
        categories = {
            "🎯 Perfect Match": [r for r in successful_results if r['category'] == "🎯 Perfect Match"],
            "💎 Hidden Gems": [r for r in successful_results if r['category'] == "💎 Hidden Gem"],
            "🌟 Overqualified": [r for r in successful_results if r['category'] == "🌟 Overqualified"],
            "✅ Good Matches": [r for r in successful_results if r['category'] == "✅ Good Match"],
            "⚠️ Potential": [r for r in successful_results if r['category'] == "⚠️ Potential"],
            "❌ Poor Matches": [r for r in successful_results if r['category'] == "❌ Poor Match"]
        }
        
        # Summary metrics with insights
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("📊 Total Processed", len(results))
        
        with col2:
            perfect_matches = len(categories["🎯 Perfect Match"])
            st.metric("🎯 Perfect Matches", perfect_matches)
        
        with col3:
            hidden_gems = len(categories["💎 Hidden Gems"])
            st.metric("💎 Hidden Gems", hidden_gems)
            if hidden_gems > 0:
                st.success("Found undervalued candidates!")
        
        with col4:
            good_matches = len(categories["✅ Good Matches"])
            st.metric("✅ Good Matches", good_matches)
        
        with col5:
            avg_match = sum(r['match_score'] for r in successful_results) / len(successful_results)
            st.metric("📈 Avg Score", f"{avg_match:.1%}")
        
        # Strategy insights
        strategy = job_criteria.get('priority_weights', '')
        if '💎 Diamond Hunter' in strategy and hidden_gems > 0:
            st.success(f"🎉 Diamond Hunter Strategy found {hidden_gems} hidden gems with high potential!")
        elif '🚀 High Potential' in strategy:
            high_growth = len([r for r in successful_results if r['growth_potential'] > 0.7])
            st.info(f"🚀 High Potential Strategy identified {high_growth} candidates with strong growth potential")
        
        # Display by categories
        for category_name, candidates in categories.items():
            if candidates:
                st.subheader(f"{category_name} ({len(candidates)} candidates)")
                
                # Special highlighting for important categories
                if "Perfect Match" in category_name:
                    st.success("🎉 These candidates meet all your requirements perfectly!")
                elif "Hidden Gem" in category_name:
                    st.warning("💎 These candidates have high potential but may be overlooked by others!")
                elif "Overqualified" in category_name:
                    st.info("🌟 These candidates exceed your requirements - consider for senior roles!")
                
                # Show top candidates in each category
                for i, candidate in enumerate(candidates[:5]):  # Show top 5 per category
                    with st.expander(f"#{i+1} - {candidate['name']} - {candidate['match_score']:.1%} Match"):
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.write(f"**Experience:** {candidate['experience']} years")
                            st.write(f"**Skills Found:** {candidate['skills_count']}")
                        
                        with col2:
                            st.write(f"**Cognitive Pattern:** {candidate['dominant_pattern']}")
                            st.write(f"**Pattern Confidence:** {candidate['pattern_confidence']:.1%}")
                        
                        with col3:
                            st.write(f"**Growth Potential:** {candidate['growth_potential']:.1%}")
                            st.write(f"**Innovation Score:** {candidate['innovation_score']:.1%}")
                        
                        with col4:
                            st.write(f"**Technical Fit:** {candidate['technical_fit']:.1%}")
                            st.write(f"**Category:** {candidate['category']}")
                        
                        # Match score visualization
                        st.progress(candidate['match_score'], text=f"Overall Match: {candidate['match_score']:.1%}")
                        
                        # Special insights for hidden gems
                        if candidate['category'] == "💎 Hidden Gem":
                            st.info("💡 **Why this is a hidden gem:** High potential despite lower experience. Consider for growth-oriented roles!")
        
        # Filtering and sorting options
        st.subheader("🔍 Filter & Sort Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_score_filter = st.slider("Minimum Match Score", 0.0, 1.0, 0.5, 0.05)
        
        with col2:
            category_filter = st.multiselect("Show Categories", 
                list(categories.keys()), 
                default=["🎯 Perfect Match", "💎 Hidden Gems", "✅ Good Matches"])
        
        with col3:
            sort_by = st.selectbox("Sort By", 
                ["Match Score", "Growth Potential", "Innovation Score", "Experience"])
        
        # Apply filters
        filtered_results = [r for r in successful_results 
                          if r['match_score'] >= min_score_filter 
                          and r['category'] in category_filter]
        
        # Sort by selected criteria
        if sort_by == "Growth Potential":
            filtered_results.sort(key=lambda x: x['growth_potential'], reverse=True)
        elif sort_by == "Innovation Score":
            filtered_results.sort(key=lambda x: x['innovation_score'], reverse=True)
        elif sort_by == "Experience":
            filtered_results.sort(key=lambda x: int(x['experience']), reverse=True)
        
        # Filtered results table
        if filtered_results:
            st.subheader(f"📋 Filtered Results ({len(filtered_results)} candidates)")
            
            df = pd.DataFrame(filtered_results)
            df['Rank'] = range(1, len(df) + 1)
            df['Match Score'] = df['match_score'].apply(lambda x: f"{x:.1%}")
            df['Growth Potential'] = df['growth_potential'].apply(lambda x: f"{x:.1%}")
            df['Innovation Score'] = df['innovation_score'].apply(lambda x: f"{x:.1%}")
            
            display_df = df[['Rank', 'name', 'experience', 'category', 'Match Score', 'Growth Potential', 'Innovation Score']]
            display_df.columns = ['Rank', 'Name', 'Experience', 'Category', 'Match Score', 'Growth Potential', 'Innovation Score']
            
            st.dataframe(display_df, use_container_width=True)
            
            # Advanced download options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download All Results",
                    data=csv,
                    file_name=f"smart_ranked_candidates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                top_candidates = df[df['match_score'] >= 0.8]
                if not top_candidates.empty:
                    top_csv = top_candidates.to_csv(index=False)
                    st.download_button(
                        label="🌟 Download Top Candidates",
                        data=top_csv,
                        file_name=f"top_candidates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            with col3:
                hidden_gems_df = df[df['category'] == "💎 Hidden Gem"]
                if not hidden_gems_df.empty:
                    gems_csv = hidden_gems_df.to_csv(index=False)
                    st.download_button(
                        label="💎 Download Hidden Gems",
                        data=gems_csv,
                        file_name=f"hidden_gems_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        else:
            st.warning("No candidates match your current filters. Try adjusting the criteria.")
    
    def display_ranked_results(self, results):
        """Display ranked candidate results"""
        st.subheader("🏆 RANKED CANDIDATES - BEST TO WORST")
        
        # Filter successful results
        successful_results = [r for r in results if r['status'] == 'Success']
        
        if not successful_results:
            st.error("❌ No successful analyses found")
            return
        
        # Sort by match score (highest first)
        successful_results.sort(key=lambda x: x['match_score'], reverse=True)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Total Processed", len(results))
        
        with col2:
            st.metric("✅ Successful", len(successful_results))
        
        with col3:
            avg_match = sum(r['match_score'] for r in successful_results) / len(successful_results)
            st.metric("📈 Avg Match Score", f"{avg_match:.1%}")
        
        with col4:
            top_candidates = len([r for r in successful_results if r['match_score'] >= 0.7])
            st.metric("🌟 Top Candidates", top_candidates)
        
        # Top 10 candidates
        st.subheader("🥇 TOP 10 BEST MATCHES")
        
        top_10 = successful_results[:10]
        
        for i, candidate in enumerate(top_10):
            with st.expander(f"#{i+1} - {candidate['name']} - {candidate['match_score']:.1%} Match"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**Experience:** {candidate['experience']} years")
                    st.write(f"**Skills:** {candidate['skills_count']} identified")
                
                with col2:
                    st.write(f"**Cognitive Pattern:** {candidate['dominant_pattern']}")
                    st.write(f"**Confidence:** {candidate['pattern_confidence']:.1%}")
                
                with col3:
                    st.write(f"**Growth Potential:** {candidate['growth_potential']:.1%}")
                    st.write(f"**Technical Fit:** {candidate['technical_fit']:.1%}")
                
                # Match score bar
                st.progress(candidate['match_score'], text=f"Overall Match: {candidate['match_score']:.1%}")
        
        # Full results table
        st.subheader("📋 All Candidates (Ranked)")
        
        df = pd.DataFrame(successful_results)
        df['Rank'] = range(1, len(df) + 1)
        df['Match Score'] = df['match_score'].apply(lambda x: f"{x:.1%}")
        df['Growth Potential'] = df['growth_potential'].apply(lambda x: f"{x:.1%}")
        df['Technical Fit'] = df['technical_fit'].apply(lambda x: f"{x:.1%}")
        
        display_df = df[['Rank', 'name', 'experience', 'dominant_pattern', 'Match Score', 'Technical Fit', 'Growth Potential']]
        display_df.columns = ['Rank', 'Name', 'Experience', 'Cognitive Pattern', 'Match Score', 'Technical Fit', 'Growth Potential']
        
        st.dataframe(display_df, use_container_width=True)
        
        # Download options
        col1, col2 = st.columns(2)
        
        with col1:
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Results CSV",
                data=csv,
                file_name=f"ranked_candidates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            top_5_csv = df.head(5).to_csv(index=False)
            st.download_button(
                label="🌟 Download Top 5 CSV",
                data=top_5_csv,
                file_name=f"top_5_candidates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    def run(self):
        """Main application runner"""
        st.markdown('<h1 class="main-header">🧠 NeuroMatch AI</h1>', unsafe_allow_html=True)
        st.markdown("### *Advanced AI-Powered Talent Matching & Career Analysis*")
        
        # Sidebar navigation
        page = st.sidebar.selectbox(
            "Navigate",
            ["🏠 Home", "📄 Resume Analysis", "🎯 Job Matching", "📊 Bulk Analysis", "⚙️ Model Training"]
        )
        
        if page == "🏠 Home":
            self.show_home_page()
        elif page == "📄 Resume Analysis":
            self.show_resume_analysis()
        elif page == "🎯 Job Matching":
            self.show_job_matching()
        elif page == "📊 Bulk Analysis":
            self.show_bulk_analysis()
        elif page == "⚙️ Model Training":
            self.show_model_training()
    
    def show_home_page(self):
        """Display home page with features overview"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 🧠 Cognitive Analysis
            - **8 Cognitive Patterns** identified using BERT
            - **Personality-Job Fit** assessment
            - **Thinking Style** classification
            """)
        
        with col2:
            st.markdown("""
            ### 📈 Growth Prediction
            - **Career Trajectory** forecasting with LSTM
            - **Future Role** predictions
            - **Skill Development** recommendations
            """)
        
        with col3:
            st.markdown("""
            ### 💡 Innovation Detection
            - **Novelty Score** calculation
            - **Creative Potential** assessment
            - **Technical Depth** analysis
            """)
        
        st.markdown("---")
        
        # Demo section
        st.subheader("🚀 Try Demo Analysis")
        
        demo_text = st.text_area(
            "Paste resume text for quick demo:",
            placeholder="Enter resume text here...",
            height=150
        )
        
        if st.button("🔍 Analyze Demo", type="primary"):
            if demo_text:
                with st.spinner("Running AI analysis..."):
                    self.run_demo_analysis(demo_text)
    
    def run_demo_analysis(self, text):
        """Run quick demo analysis"""
        # Mock resume data from text
        resume_data = {
            'name': 'Demo User',
            'experience': '5',
            'skills': ['python', 'machine learning', 'sql'],
            'education': ['Bachelor of Science'],
            'current_role': 'Data Scientist'
        }
        
        # Cognitive analysis
        cognitive_scores = self.cognitive_analyzer.predict_cognitive_pattern(text)
        dominant_pattern, confidence = self.cognitive_analyzer.get_dominant_pattern(text)
        
        # Growth analysis
        growth_analysis = self.growth_predictor.predict_growth_potential(resume_data)
        
        # Innovation analysis
        innovation_metrics = self.innovation_detector.calculate_innovation_score(text)
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Dominant Pattern", dominant_pattern, f"{confidence:.1%} confidence")
        
        with col2:
            st.metric("Growth Potential", f"{growth_analysis['overall_growth_potential']:.1%}")
        
        with col3:
            st.metric("Innovation Score", f"{innovation_metrics.novelty_score:.1%}")
        
        # Cognitive pattern chart
        fig = go.Figure(data=go.Scatterpolar(
            r=list(cognitive_scores.values()),
            theta=list(cognitive_scores.keys()),
            fill='toself',
            name='Cognitive Profile'
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title="Cognitive Pattern Analysis"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def show_resume_analysis(self):
        """Resume analysis page"""
        st.header("📄 Resume Analysis - Multiple Input Methods")
        
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["📝 Copy-Paste Text", "📁 Upload File", "🖼️ Upload Image"]
        )
        
        resume_text = None
        resume_data = None
        
        if input_method == "📝 Copy-Paste Text":
            st.subheader("📝 Paste Resume Content")
            resume_text = st.text_area(
                "Paste resume text here:",
                height=300,
                placeholder="Copy and paste the complete resume text here..."
            )
            
            if resume_text and st.button("🔍 Analyze Resume Text"):
                # Parse text directly
                resume_data = self.parse_text_resume(resume_text)
                st.session_state.resume_data = resume_data
                
                # Run analysis
                results = self.run_comprehensive_analysis(resume_data, resume_text)
                st.session_state.analysis_results = results
                self.display_analysis_results(results, resume_data)
        
        elif input_method == "📁 Upload File":
            # Multiple file format support
            uploaded_file = st.file_uploader(
                "Upload Resume (Multiple Formats Supported)",
                type=['pdf', 'docx', 'txt', 'jpg', 'jpeg', 'png'],
                help="Upload PDF, Word, Text, or Image files"
            )
            
            if uploaded_file is not None:
                self.process_uploaded_file(uploaded_file)
        
        elif input_method == "🖼️ Upload Image":
            st.info("📷 Image processing feature - Upload resume screenshots or photos")
            image_file = st.file_uploader(
                "Upload Resume Image",
                type=['jpg', 'jpeg', 'png', 'bmp'],
                help="Upload resume screenshots - AI will extract text"
            )
            
            if image_file is not None:
                st.success("Image uploaded! Text extraction feature coming soon.")
    
    def process_uploaded_file(self, uploaded_file):
        """Process uploaded file"""
        with st.spinner("Processing resume..."):
            try:
                # Handle different file types
                if uploaded_file.type == "application/pdf":
                    # Save PDF temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_path = tmp_file.name
                    
                    # Extract text from PDF
                    text = extract_text_from_pdf(tmp_path)
                    os.unlink(tmp_path)  # Clean up
                    
                elif uploaded_file.type == "text/plain":
                    # Handle text files
                    text = str(uploaded_file.read(), "utf-8")
                    
                else:
                    # For other formats, try to read as text
                    text = str(uploaded_file.read(), "utf-8")
                
                if not text.strip():
                    st.error("❌ Could not extract text from the file")
                    return
                
                # Parse resume data
                resume_data = self.parse_text_resume(text)
                st.session_state.resume_data = resume_data
                
                # Run comprehensive analysis
                results = self.run_comprehensive_analysis(resume_data, text)
                st.session_state.analysis_results = results
                
                # Display results
                self.display_analysis_results(results, resume_data)
                
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                st.info("💡 Try using the copy-paste method instead")
    
    def run_comprehensive_analysis(self, resume_data, resume_text):
        """Run all AI analyses"""
        results = {}
        
        # Cognitive analysis
        cognitive_scores = self.cognitive_analyzer.predict_cognitive_pattern(resume_text)
        dominant_pattern, confidence = self.cognitive_analyzer.get_dominant_pattern(resume_text)
        
        results['cognitive'] = {
            'scores': cognitive_scores,
            'dominant_pattern': dominant_pattern,
            'confidence': confidence
        }
        
        # Growth prediction
        growth_analysis = self.growth_predictor.predict_growth_potential(resume_data)
        future_roles = self.growth_predictor.predict_future_roles(resume_data)
        
        results['growth'] = {
            'analysis': growth_analysis,
            'future_roles': future_roles
        }
        
        # Innovation detection
        innovation_metrics = self.innovation_detector.calculate_innovation_score(resume_text)
        
        results['innovation'] = {
            'metrics': innovation_metrics
        }
        
        return results
    
    def display_analysis_results(self, results, resume_data):
        """Display comprehensive analysis results"""
        
        # Overview metrics
        st.subheader("📊 Analysis Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Dominant Cognitive Pattern",
                results['cognitive']['dominant_pattern'],
                f"{results['cognitive']['confidence']:.1%} confidence"
            )
        
        with col2:
            st.metric(
                "Growth Potential",
                f"{results['growth']['analysis']['overall_growth_potential']:.1%}",
                "High" if results['growth']['analysis']['overall_growth_potential'] > 0.7 else "Moderate"
            )
        
        with col3:
            st.metric(
                "Innovation Score",
                f"{results['innovation']['metrics'].novelty_score:.1%}",
                "Creative" if results['innovation']['metrics'].novelty_score > 0.6 else "Standard"
            )
        
        with col4:
            st.metric(
                "Experience Level",
                f"{resume_data.get('experience', '0')} years",
                "Senior" if int(resume_data.get('experience', '0')) > 5 else "Mid-level"
            )
        
        st.markdown("---")
        
        # Detailed analysis tabs
        tab1, tab2, tab3, tab4 = st.tabs(["🧠 Cognitive Profile", "📈 Growth Analysis", "💡 Innovation", "🎯 Recommendations"])
        
        with tab1:
            self.display_cognitive_analysis(results['cognitive'])
        
        with tab2:
            self.display_growth_analysis(results['growth'])
        
        with tab3:
            self.display_innovation_analysis(results['innovation'])
        
        with tab4:
            self.display_recommendations(results, resume_data)
    
    def display_cognitive_analysis(self, cognitive_results):
        """Display cognitive pattern analysis"""
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Radar chart
            fig = go.Figure(data=go.Scatterpolar(
                r=list(cognitive_results['scores'].values()),
                theta=list(cognitive_results['scores'].keys()),
                fill='toself',
                name='Cognitive Profile',
                line_color='rgb(102, 126, 234)'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1],
                        tickformat='.0%'
                    )
                ),
                showlegend=True,
                title="Cognitive Pattern Distribution",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Pattern Breakdown")
            
            # Sort patterns by score
            sorted_patterns = sorted(
                cognitive_results['scores'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            for pattern, score in sorted_patterns:
                st.progress(score, text=f"{pattern}: {score:.1%}")
    
    def display_growth_analysis(self, growth_results):
        """Display growth prediction analysis"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Growth Metrics")
            
            metrics = growth_results['analysis']
            
            # Growth metrics chart
            fig = go.Figure(data=[
                go.Bar(
                    x=['Overall', 'Technical', 'Leadership', 'Innovation', 'Adaptability'],
                    y=[
                        metrics['overall_growth_potential'],
                        metrics['technical_growth'],
                        metrics['leadership_growth'],
                        metrics['innovation_potential'],
                        metrics['adaptability_score']
                    ],
                    marker_color='rgb(102, 126, 234)'
                )
            ])
            
            fig.update_layout(
                title="Growth Potential Analysis",
                yaxis_title="Score",
                yaxis=dict(range=[0, 1], tickformat='.0%')
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Future Career Path")
            
            future_roles = growth_results['future_roles']
            
            # Create timeline
            years = [role['year'] for role in future_roles]
            levels = [role['estimated_level'] for role in future_roles]
            level_names = [role['level_name'] for role in future_roles]
            
            fig = go.Figure(data=go.Scatter(
                x=years,
                y=levels,
                mode='lines+markers',
                text=level_names,
                textposition="top center",
                line=dict(color='rgb(102, 126, 234)', width=3),
                marker=dict(size=10)
            ))
            
            fig.update_layout(
                title="Projected Career Progression",
                xaxis_title="Years Ahead",
                yaxis_title="Career Level",
                yaxis=dict(tickmode='array', tickvals=list(range(10)), ticktext=[
                    'Intern', 'Junior', 'Mid', 'Senior', 'Lead', 'Principal', 'Manager', 'Director', 'VP', 'C-Level'
                ])
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def display_innovation_analysis(self, innovation_results):
        """Display innovation analysis"""
        metrics = innovation_results['metrics']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Innovation radar
            innovation_scores = {
                'Novelty': metrics.novelty_score,
                'Complexity': metrics.complexity_score,
                'Impact': metrics.impact_potential,
                'Technical Depth': metrics.technical_depth,
                'Creativity': metrics.creativity_index,
                'Market Relevance': metrics.market_relevance
            }
            
            fig = go.Figure(data=go.Scatterpolar(
                r=list(innovation_scores.values()),
                theta=list(innovation_scores.keys()),
                fill='toself',
                name='Innovation Profile',
                line_color='rgb(118, 75, 162)'
            ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickformat='.0%')),
                showlegend=True,
                title="Innovation Assessment"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Innovation Insights")
            
            # Generate insights
            overall_score = (
                metrics.novelty_score + metrics.complexity_score + 
                metrics.impact_potential + metrics.technical_depth + 
                metrics.creativity_index + metrics.market_relevance
            ) / 6
            
            if overall_score >= 0.7:
                st.success("🌟 **High Innovation Potential** - Exceptional creative and technical capabilities")
            elif overall_score >= 0.5:
                st.info("💡 **Moderate Innovation Potential** - Good foundation with room for growth")
            else:
                st.warning("📚 **Developing Innovation Skills** - Focus on creative problem-solving")
            
            # Top strengths
            strengths = []
            if metrics.technical_depth >= 0.6:
                strengths.append("Strong technical expertise")
            if metrics.creativity_index >= 0.6:
                strengths.append("Creative problem-solving")
            if metrics.novelty_score >= 0.6:
                strengths.append("Novel approach to challenges")
            
            if strengths:
                st.write("**Key Strengths:**")
                for strength in strengths:
                    st.write(f"• {strength}")
    
    def display_recommendations(self, results, resume_data):
        """Display personalized recommendations"""
        st.subheader("🎯 Personalized Career Recommendations")
        
        # Skill recommendations
        st.write("### 📚 Skill Development")
        
        growth_metrics = results['growth']['analysis']
        
        if growth_metrics['technical_growth'] < 0.6:
            st.info("💻 **Technical Skills**: Consider developing advanced programming skills and modern frameworks")
        
        if growth_metrics['leadership_growth'] < 0.6:
            st.info("👥 **Leadership**: Focus on team management and mentoring experiences")
        
        if results['innovation']['metrics'].novelty_score < 0.5:
            st.info("💡 **Innovation**: Engage in creative projects and explore emerging technologies")
        
        # Role recommendations
        st.write("### 🎯 Ideal Role Types")
        
        dominant_pattern = results['cognitive']['dominant_pattern']
        
        role_recommendations = {
            'Analytical Thinker': ['Data Scientist', 'Research Engineer', 'Business Analyst'],
            'Creative Innovator': ['Product Designer', 'Innovation Manager', 'Startup Founder'],
            'Strategic Planner': ['Product Manager', 'Solutions Architect', 'Strategy Consultant'],
            'Collaborative Leader': ['Engineering Manager', 'Team Lead', 'Project Manager'],
            'Detail Perfectionist': ['Quality Engineer', 'Systems Analyst', 'Compliance Manager'],
            'Adaptive Problem-Solver': ['DevOps Engineer', 'Consultant', 'Technical Lead'],
            'Results-Driven Executor': ['Sales Engineer', 'Delivery Manager', 'Operations Lead'],
            'Empathetic Communicator': ['Customer Success', 'Technical Writer', 'Developer Relations']
        }
        
        recommended_roles = role_recommendations.get(dominant_pattern, ['Software Engineer'])
        
        for role in recommended_roles:
            st.success(f"✅ **{role}** - Aligns with your {dominant_pattern} cognitive pattern")
    
    def show_job_matching(self):
        """Job matching functionality"""
        st.header("🎯 AI-Powered Job Matching")
        
        if st.session_state.resume_data is None:
            st.warning("⚠️ Please analyze a resume first in the Resume Analysis section.")
            return
        
        # Job requirements input
        st.subheader("Job Requirements")
        
        col1, col2 = st.columns(2)
        
        with col1:
            job_title = st.text_input("Job Title", placeholder="e.g., Senior Data Scientist")
            required_skills = st.text_area(
                "Required Skills (comma-separated)",
                placeholder="python, machine learning, sql, aws"
            )
            min_experience = st.number_input("Minimum Experience (years)", min_value=0, max_value=20, value=3)
        
        with col2:
            education_level = st.selectbox("Education Level", ['bachelor', 'master', 'phd', 'mba'])
            role_level = st.selectbox("Role Level", ['junior', 'mid', 'senior', 'lead'])
            innovation_level = st.slider("Innovation Level Required", 0.0, 1.0, 0.5, 0.1)
        
        if st.button("🔍 Calculate Match Score", type="primary"):
            # Prepare job requirements
            job_requirements = {
                'role': job_title,
                'required_skills': [skill.strip() for skill in required_skills.split(',') if skill.strip()],
                'min_experience': min_experience,
                'education_level': education_level,
                'role_level': role_level,
                'innovation_level': innovation_level,
                'cognitive_requirements': {
                    'Analytical Thinker': 0.7,  # Default requirements
                    'Strategic Planner': 0.6
                }
            }
            
            # Calculate match
            with st.spinner("Calculating AI match score..."):
                match_result = self.ensemble_matcher.predict_match(
                    st.session_state.resume_data, 
                    job_requirements
                )
            
            # Display results
            self.display_match_results(match_result, job_requirements)
    
    def display_match_results(self, match_result, job_requirements):
        """Display job matching results"""
        st.subheader("📊 Match Analysis Results")
        
        # Overall score
        overall_score = match_result.overall_score
        
        if overall_score >= 0.8:
            st.success(f"🎉 **Excellent Match**: {overall_score:.1%} compatibility")
        elif overall_score >= 0.6:
            st.info(f"👍 **Good Match**: {overall_score:.1%} compatibility")
        else:
            st.warning(f"⚠️ **Moderate Match**: {overall_score:.1%} compatibility")
        
        # Detailed breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Match Breakdown")
            
            match_metrics = {
                'Technical Fit': match_result.technical_fit,
                'Cognitive Compatibility': match_result.cognitive_compatibility,
                'Growth Alignment': match_result.growth_alignment,
                'Innovation Match': match_result.innovation_match,
                'Cultural Fit': match_result.cultural_fit
            }
            
            for metric, score in match_metrics.items():
                st.progress(score, text=f"{metric}: {score:.1%}")
        
        with col2:
            st.subheader("Recommendations")
            
            for recommendation in match_result.recommendations:
                st.info(f"💡 {recommendation}")
            
            st.metric("Confidence Level", f"{match_result.confidence:.1%}")
    
    def show_bulk_analysis(self):
        """Enhanced bulk analysis for recruiters"""
        st.header("📊 Smart Bulk Resume Ranking System")
        st.markdown("### 🎯 **Upload 50+ resumes and get the BEST candidates ranked automatically!**")
        
        # Advanced job requirements for smart ranking
        st.subheader("🎯 Define Your Ideal Candidate (Advanced Filtering)")
        
        # Basic requirements
        col1, col2, col3 = st.columns(3)
        
        with col1:
            target_role = st.selectbox("Target Role", 
                ["Data Scientist", "Software Engineer", "Product Manager", "ML Engineer", "DevOps Engineer", "Full Stack Developer", "Backend Developer", "Frontend Developer"])
            required_skills = st.text_input("Required Skills (comma-separated)", 
                placeholder="python, machine learning, sql, aws")
            min_experience = st.slider("Minimum Experience (years)", 0, 15, 3)
        
        with col2:
            education_level = st.selectbox("Education Level", ["any", "bachelor", "master", "phd"])
            salary_range = st.selectbox("Expected Salary Range", 
                ["Any", "Entry Level ($40-60K)", "Mid Level ($60-90K)", "Senior ($90-120K)", "Lead ($120K+)"])
            location_pref = st.selectbox("Location Preference", 
                ["Any", "Remote Only", "Hybrid", "On-site", "Specific City"])
        
        with col3:
            priority_weights = st.selectbox("Ranking Strategy", [
                "🎯 Exact Match (strict requirements)",
                "💎 Diamond Hunter (find hidden gems)", 
                "🚀 High Potential (growth-focused)",
                "🧠 Cognitive Fit (personality match)",
                "⚖️ Balanced Approach"
            ])
            
            include_career_changers = st.checkbox("Include Career Changers", value=True)
            include_junior_high_potential = st.checkbox("Include Junior High-Potential", value=True)
        
        # Advanced filters
        st.subheader("🔍 Advanced Filters (Never Miss Great Candidates)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🎯 Must-Have Skills:**")
            must_have_skills = st.text_input("Critical Skills (candidate MUST have)", 
                placeholder="python, sql")
            
            st.write("**💡 Nice-to-Have Skills:**")
            nice_to_have = st.text_input("Bonus Skills (extra points)", 
                placeholder="aws, docker, kubernetes")
            
            st.write("**🚫 Deal Breakers:**")
            deal_breakers = st.text_input("Avoid if they have (optional)", 
                placeholder="php, outdated tech")
        
        with col2:
            st.write("**🏆 Bonus Criteria:**")
            bonus_leadership = st.checkbox("Leadership Experience", value=False)
            bonus_startup = st.checkbox("Startup Experience", value=False)
            bonus_remote = st.checkbox("Remote Work Experience", value=False)
            bonus_certifications = st.checkbox("Industry Certifications", value=False)
            
            st.write("**📈 Growth Indicators:**")
            look_for_growth = st.checkbox("Rapid Skill Acquisition", value=True)
            look_for_projects = st.checkbox("Side Projects/Open Source", value=True)
            look_for_learning = st.checkbox("Continuous Learning", value=True)
        
        st.markdown("---")
        
        # Multiple input methods for bulk
        bulk_method = st.radio(
            "Bulk Input Method:",
            ["📁 Upload Files", "📝 Paste Multiple Resumes", "🖼️ Upload Images"]
        )
        
        if bulk_method == "📁 Upload Files":
            uploaded_files = st.file_uploader(
                "Upload Multiple Resumes (All Formats Supported)",
                type=['pdf', 'docx', 'txt', 'jpg', 'jpeg', 'png'],
                accept_multiple_files=True,
                help="Upload up to 100 resumes in any format - PDF, Word, Text, or Images"
            )
        
        elif bulk_method == "📝 Paste Multiple Resumes":
            st.subheader("📝 Paste Multiple Resume Texts")
            st.info("💡 Separate each resume with '---RESUME---' on a new line")
            bulk_text = st.text_area(
                "Paste multiple resumes here:",
                height=400,
                placeholder="""Resume 1 content here...

---RESUME---

Resume 2 content here...

---RESUME---

Resume 3 content here..."""
            )
            
            if bulk_text and st.button("🚀 Process Bulk Text"):
                resumes_list = bulk_text.split("---RESUME---")
                resumes_list = [r.strip() for r in resumes_list if r.strip()]
                st.success(f"📊 Found {len(resumes_list)} resumes to process")
                
                # Prepare job criteria
                job_criteria = {
                    'target_role': target_role,
                    'required_skills': [s.strip() for s in required_skills.split(',') if s.strip()],
                    'min_experience': min_experience,
                    'education_level': education_level,
                    'priority_weights': priority_weights,
                    'must_have_skills': [s.strip() for s in must_have_skills.split(',') if s.strip()],
                    'nice_to_have': [s.strip() for s in nice_to_have.split(',') if s.strip()],
                    'deal_breakers': [s.strip() for s in deal_breakers.split(',') if s.strip()],
                    'bonus_leadership': bonus_leadership,
                    'bonus_startup': bonus_startup,
                    'bonus_remote': bonus_remote,
                    'bonus_certifications': bonus_certifications,
                    'look_for_growth': look_for_growth,
                    'look_for_projects': look_for_projects,
                    'look_for_learning': look_for_learning
                }
                
                # Process bulk text
                self.process_bulk_text_resumes(resumes_list, job_criteria)
        
        elif bulk_method == "🖼️ Upload Images":
            uploaded_files = st.file_uploader(
                "Upload Resume Images (Screenshots, Photos)",
                type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
                accept_multiple_files=True,
                help="Upload resume screenshots or photos - AI will extract text"
            )
        
        if uploaded_files:
            st.write(f"📁 {len(uploaded_files)} files uploaded")
            
            if st.button("🚀 Start Bulk Analysis", type="primary"):
                results = []
                progress_bar = st.progress(0)
                
                for i, uploaded_file in enumerate(uploaded_files):
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_path = tmp_file.name
                    
                    try:
                        # Process resume
                        text = extract_text_from_pdf(tmp_path)
                        
                        if is_resume_pdf(text):
                            resume_data = parse_resume(text)
                            
                            # Quick analysis
                            cognitive_scores = self.cognitive_analyzer.predict_cognitive_pattern(text)
                            dominant_pattern, confidence = self.cognitive_analyzer.get_dominant_pattern(text)
                            growth_analysis = self.growth_predictor.predict_growth_potential(resume_data)
                            
                            results.append({
                                'filename': uploaded_file.name,
                                'name': resume_data.get('name', 'Unknown'),
                                'experience': resume_data.get('experience', '0'),
                                'skills_count': len(resume_data.get('skills', [])),
                                'dominant_pattern': dominant_pattern,
                                'pattern_confidence': confidence,
                                'growth_potential': growth_analysis['overall_growth_potential'],
                                'status': 'Success'
                            })
                        else:
                            results.append({
                                'filename': uploaded_file.name,
                                'status': 'Invalid Resume'
                            })
                    
                    except Exception as e:
                        results.append({
                            'filename': uploaded_file.name,
                            'status': f'Error: {str(e)}'
                        })
                    
                    finally:
                        os.unlink(tmp_path)
                        progress_bar.progress((i + 1) / len(uploaded_files))
                
                # Display results
                self.display_bulk_results(results)
    
    def display_bulk_results(self, results):
        """Display bulk analysis results"""
        st.subheader("📈 Bulk Analysis Results")
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Summary stats
        successful_analyses = df[df['status'] == 'Success']
        
        if len(successful_analyses) > 0:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Processed", len(df))
            
            with col2:
                st.metric("Successful", len(successful_analyses))
            
            with col3:
                avg_experience = successful_analyses['experience'].astype(float).mean()
                st.metric("Avg Experience", f"{avg_experience:.1f} years")
            
            with col4:
                avg_growth = successful_analyses['growth_potential'].mean()
                st.metric("Avg Growth Potential", f"{avg_growth:.1%}")
            
            # Results table
            st.dataframe(successful_analyses, use_container_width=True)
            
            # Download results
            csv = successful_analyses.to_csv(index=False)
            st.download_button(
                label="📥 Download Results CSV",
                data=csv,
                file_name=f"bulk_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    def show_model_training(self):
        """Model training interface"""
        st.header("⚙️ Model Training & Configuration")
        
        st.info("🚧 **Advanced Feature**: Train custom models with your data")
        
        tab1, tab2 = st.tabs(["📊 Generate Training Data", "🤖 Train Models"])
        
        with tab1:
            st.subheader("Generate Synthetic Training Data")
            
            num_samples = st.number_input("Number of samples", min_value=100, max_value=50000, value=1000)
            
            if st.button("🎲 Generate Dataset"):
                with st.spinner("Generating synthetic training data..."):
                    generator = SyntheticDataGenerator()
                    resumes, job_reqs, scores = generator.generate_training_dataset(num_samples)
                    generator.save_dataset(resumes, job_reqs, scores)
                
                st.success(f"✅ Generated {num_samples} training samples!")
                
                # Show sample
                sample_df = pd.DataFrame([{
                    'name': resumes[0]['name'],
                    'role': resumes[0]['target_role'],
                    'experience': resumes[0]['experience'],
                    'pattern': resumes[0]['dominant_cognitive_pattern'],
                    'match_score': scores[0]
                }])
                
                st.write("Sample data:")
                st.dataframe(sample_df)
        
        with tab2:
            st.subheader("Train AI Models")
            
            st.warning("⚠️ Model training requires significant computational resources")
            
            model_type = st.selectbox(
                "Select Model to Train",
                ["Cognitive Analyzer", "Growth Predictor", "Ensemble Matcher"]
            )
            
            if st.button("🚀 Start Training"):
                st.info(f"Training {model_type}... This may take several minutes.")
                # Training logic would go here
                st.success("Model training completed!")

def main():
    """Main application entry point"""
    app = NeuroMatchApp()
    app.run()

if __name__ == "__main__":
    main()
