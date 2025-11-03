"""
Indian Salary Database - Real-world accurate salary ranges
Based on 2024-2025 market data for India
"""

# Salary data in INR (Indian Rupees) per annum
# Data sources: Glassdoor, Naukri, AmbitionBox, PayScale (India)

INDIAN_SALARY_DATA = {
    # Software Engineering & Development
    'software_engineer': {
        'fresher': {'min': 300000, 'max': 600000, 'avg': 450000},
        'junior': {'min': 600000, 'max': 1200000, 'avg': 900000},
        'mid': {'min': 1200000, 'max': 2500000, 'avg': 1800000},
        'senior': {'min': 2500000, 'max': 5000000, 'avg': 3500000},
        'lead': {'min': 4000000, 'max': 8000000, 'avg': 6000000}
    },
    'full_stack_developer': {
        'fresher': {'min': 350000, 'max': 700000, 'avg': 500000},
        'junior': {'min': 700000, 'max': 1400000, 'avg': 1000000},
        'mid': {'min': 1400000, 'max': 2800000, 'avg': 2000000},
        'senior': {'min': 2800000, 'max': 5500000, 'avg': 4000000},
        'lead': {'min': 4500000, 'max': 9000000, 'avg': 6500000}
    },
    'backend_developer': {
        'fresher': {'min': 350000, 'max': 650000, 'avg': 480000},
        'junior': {'min': 650000, 'max': 1300000, 'avg': 950000},
        'mid': {'min': 1300000, 'max': 2600000, 'avg': 1900000},
        'senior': {'min': 2600000, 'max': 5200000, 'avg': 3800000},
        'lead': {'min': 4200000, 'max': 8500000, 'avg': 6200000}
    },
    'frontend_developer': {
        'fresher': {'min': 300000, 'max': 600000, 'avg': 420000},
        'junior': {'min': 600000, 'max': 1200000, 'avg': 850000},
        'mid': {'min': 1200000, 'max': 2400000, 'avg': 1700000},
        'senior': {'min': 2400000, 'max': 4800000, 'avg': 3400000},
        'lead': {'min': 3800000, 'max': 7500000, 'avg': 5500000}
    },
    
    # Data Science & Analytics
    'data_scientist': {
        'fresher': {'min': 400000, 'max': 800000, 'avg': 600000},
        'junior': {'min': 800000, 'max': 1600000, 'avg': 1200000},
        'mid': {'min': 1600000, 'max': 3200000, 'avg': 2400000},
        'senior': {'min': 3200000, 'max': 6500000, 'avg': 4800000},
        'lead': {'min': 5500000, 'max': 12000000, 'avg': 8500000}
    },
    'data_analyst': {
        'fresher': {'min': 300000, 'max': 600000, 'avg': 450000},
        'junior': {'min': 600000, 'max': 1200000, 'avg': 900000},
        'mid': {'min': 1200000, 'max': 2400000, 'avg': 1800000},
        'senior': {'min': 2400000, 'max': 4800000, 'avg': 3600000},
        'lead': {'min': 4000000, 'max': 8000000, 'avg': 6000000}
    },
    'machine_learning_engineer': {
        'fresher': {'min': 500000, 'max': 1000000, 'avg': 750000},
        'junior': {'min': 1000000, 'max': 2000000, 'avg': 1500000},
        'mid': {'min': 2000000, 'max': 4000000, 'avg': 3000000},
        'senior': {'min': 4000000, 'max': 8000000, 'avg': 6000000},
        'lead': {'min': 7000000, 'max': 15000000, 'avg': 10000000}
    },
    'ai_engineer': {
        'fresher': {'min': 500000, 'max': 1000000, 'avg': 750000},
        'junior': {'min': 1000000, 'max': 2000000, 'avg': 1500000},
        'mid': {'min': 2000000, 'max': 4000000, 'avg': 3000000},
        'senior': {'min': 4000000, 'max': 8000000, 'avg': 6000000},
        'lead': {'min': 7000000, 'max': 15000000, 'avg': 10000000}
    },
    
    # DevOps & Cloud
    'devops_engineer': {
        'fresher': {'min': 400000, 'max': 800000, 'avg': 600000},
        'junior': {'min': 800000, 'max': 1600000, 'avg': 1200000},
        'mid': {'min': 1600000, 'max': 3200000, 'avg': 2400000},
        'senior': {'min': 3200000, 'max': 6500000, 'avg': 4800000},
        'lead': {'min': 5500000, 'max': 11000000, 'avg': 8000000}
    },
    'cloud_engineer': {
        'fresher': {'min': 450000, 'max': 900000, 'avg': 650000},
        'junior': {'min': 900000, 'max': 1800000, 'avg': 1300000},
        'mid': {'min': 1800000, 'max': 3600000, 'avg': 2700000},
        'senior': {'min': 3600000, 'max': 7200000, 'avg': 5400000},
        'lead': {'min': 6000000, 'max': 12000000, 'avg': 9000000}
    },
    
    # Product & Design
    'product_manager': {
        'fresher': {'min': 600000, 'max': 1200000, 'avg': 900000},
        'junior': {'min': 1200000, 'max': 2400000, 'avg': 1800000},
        'mid': {'min': 2400000, 'max': 4800000, 'avg': 3600000},
        'senior': {'min': 4800000, 'max': 9600000, 'avg': 7200000},
        'lead': {'min': 8000000, 'max': 16000000, 'avg': 12000000}
    },
    'ui_ux_designer': {
        'fresher': {'min': 300000, 'max': 600000, 'avg': 450000},
        'junior': {'min': 600000, 'max': 1200000, 'avg': 900000},
        'mid': {'min': 1200000, 'max': 2400000, 'avg': 1800000},
        'senior': {'min': 2400000, 'max': 4800000, 'avg': 3600000},
        'lead': {'min': 4000000, 'max': 8000000, 'avg': 6000000}
    },
    
    # QA & Testing
    'qa_engineer': {
        'fresher': {'min': 250000, 'max': 500000, 'avg': 375000},
        'junior': {'min': 500000, 'max': 1000000, 'avg': 750000},
        'mid': {'min': 1000000, 'max': 2000000, 'avg': 1500000},
        'senior': {'min': 2000000, 'max': 4000000, 'avg': 3000000},
        'lead': {'min': 3500000, 'max': 7000000, 'avg': 5000000}
    },
    'automation_engineer': {
        'fresher': {'min': 300000, 'max': 600000, 'avg': 450000},
        'junior': {'min': 600000, 'max': 1200000, 'avg': 900000},
        'mid': {'min': 1200000, 'max': 2400000, 'avg': 1800000},
        'senior': {'min': 2400000, 'max': 4800000, 'avg': 3600000},
        'lead': {'min': 4000000, 'max': 8000000, 'avg': 6000000}
    },
    
    # Business & Analytics
    'business_analyst': {
        'fresher': {'min': 350000, 'max': 700000, 'avg': 525000},
        'junior': {'min': 700000, 'max': 1400000, 'avg': 1050000},
        'mid': {'min': 1400000, 'max': 2800000, 'avg': 2100000},
        'senior': {'min': 2800000, 'max': 5600000, 'avg': 4200000},
        'lead': {'min': 4500000, 'max': 9000000, 'avg': 6750000}
    },
    
    # Mobile Development
    'android_developer': {
        'fresher': {'min': 300000, 'max': 600000, 'avg': 450000},
        'junior': {'min': 600000, 'max': 1200000, 'avg': 900000},
        'mid': {'min': 1200000, 'max': 2400000, 'avg': 1800000},
        'senior': {'min': 2400000, 'max': 4800000, 'avg': 3600000},
        'lead': {'min': 4000000, 'max': 8000000, 'avg': 6000000}
    },
    'ios_developer': {
        'fresher': {'min': 350000, 'max': 700000, 'avg': 525000},
        'junior': {'min': 700000, 'max': 1400000, 'avg': 1050000},
        'mid': {'min': 1400000, 'max': 2800000, 'avg': 2100000},
        'senior': {'min': 2800000, 'max': 5600000, 'avg': 4200000},
        'lead': {'min': 4500000, 'max': 9000000, 'avg': 6750000}
    },
    
    # Cybersecurity
    'security_engineer': {
        'fresher': {'min': 400000, 'max': 800000, 'avg': 600000},
        'junior': {'min': 800000, 'max': 1600000, 'avg': 1200000},
        'mid': {'min': 1600000, 'max': 3200000, 'avg': 2400000},
        'senior': {'min': 3200000, 'max': 6500000, 'avg': 4800000},
        'lead': {'min': 5500000, 'max': 11000000, 'avg': 8000000}
    },
    
    # Default fallback
    'default': {
        'fresher': {'min': 300000, 'max': 600000, 'avg': 450000},
        'junior': {'min': 600000, 'max': 1200000, 'avg': 900000},
        'mid': {'min': 1200000, 'max': 2400000, 'avg': 1800000},
        'senior': {'min': 2400000, 'max': 4800000, 'avg': 3600000},
        'lead': {'min': 4000000, 'max': 8000000, 'avg': 6000000}
    }
}

# City-wise multipliers (cost of living adjustment)
CITY_MULTIPLIERS = {
    'bangalore': 1.15,
    'mumbai': 1.20,
    'delhi': 1.10,
    'ncr': 1.10,
    'gurgaon': 1.12,
    'noida': 1.08,
    'hyderabad': 1.05,
    'pune': 1.08,
    'chennai': 1.00,
    'kolkata': 0.95,
    'ahmedabad': 0.95,
    'other': 0.90
}

# Company size multipliers
COMPANY_SIZE_MULTIPLIERS = {
    'startup': 0.85,
    'small': 0.90,
    'medium': 1.00,
    'large': 1.15,
    'mnc': 1.25,
    'faang': 1.50  # FAANG/Top-tier companies
}


def get_experience_level(years):
    """Determine experience level from years"""
    if years == 0:
        return 'fresher'
    elif years <= 2:
        return 'junior'
    elif years <= 5:
        return 'mid'
    elif years <= 10:
        return 'senior'
    else:
        return 'lead'


def normalize_job_title(title):
    """Normalize job title to match database keys"""
    title_lower = title.lower().strip()
    
    # Mapping variations to standard titles
    title_mappings = {
        'sde': 'software_engineer',
        'software development engineer': 'software_engineer',
        'developer': 'software_engineer',
        'programmer': 'software_engineer',
        'full stack': 'full_stack_developer',
        'fullstack': 'full_stack_developer',
        'backend': 'backend_developer',
        'back end': 'backend_developer',
        'frontend': 'frontend_developer',
        'front end': 'frontend_developer',
        'data science': 'data_scientist',
        'ds': 'data_scientist',
        'ml engineer': 'machine_learning_engineer',
        'ml': 'machine_learning_engineer',
        'ai': 'ai_engineer',
        'devops': 'devops_engineer',
        'cloud': 'cloud_engineer',
        'product': 'product_manager',
        'pm': 'product_manager',
        'designer': 'ui_ux_designer',
        'ux': 'ui_ux_designer',
        'ui': 'ui_ux_designer',
        'qa': 'qa_engineer',
        'tester': 'qa_engineer',
        'test': 'qa_engineer',
        'automation': 'automation_engineer',
        'business analyst': 'business_analyst',
        'ba': 'business_analyst',
        'android': 'android_developer',
        'ios': 'ios_developer',
        'security': 'security_engineer',
        'cyber': 'security_engineer'
    }
    
    # Check for matches
    for key, value in title_mappings.items():
        if key in title_lower:
            return value
    
    return 'default'


def calculate_indian_salary(job_title, experience_years, city='other', company_size='medium', skills_count=0):
    """
    Calculate realistic Indian salary ranges
    
    Args:
        job_title: Job title/role
        experience_years: Years of experience
        city: City location (for cost of living adjustment)
        company_size: Company size (startup/small/medium/large/mnc/faang)
        skills_count: Number of skills (bonus factor)
    
    Returns:
        Dictionary with salary ranges in INR
    """
    # Normalize inputs
    normalized_title = normalize_job_title(job_title)
    experience_level = get_experience_level(experience_years)
    city_lower = city.lower().strip()
    company_size_lower = company_size.lower().strip()
    
    # Get base salary data
    if normalized_title in INDIAN_SALARY_DATA:
        salary_data = INDIAN_SALARY_DATA[normalized_title][experience_level]
    else:
        salary_data = INDIAN_SALARY_DATA['default'][experience_level]
    
    # Get multipliers
    city_multiplier = CITY_MULTIPLIERS.get(city_lower, CITY_MULTIPLIERS['other'])
    company_multiplier = COMPANY_SIZE_MULTIPLIERS.get(company_size_lower, 1.0)
    
    # Skills bonus (1-2% per skill, max 20%)
    skills_bonus = min(1 + (skills_count * 0.02), 1.20)
    
    # Calculate final salary
    total_multiplier = city_multiplier * company_multiplier * skills_bonus
    
    min_salary = int(salary_data['min'] * total_multiplier)
    max_salary = int(salary_data['max'] * total_multiplier)
    avg_salary = int(salary_data['avg'] * total_multiplier)
    
    # Recommended offer (slightly above average)
    recommended_offer = int(avg_salary * 1.05)
    
    return {
        'min_salary': min_salary,
        'max_salary': max_salary,
        'avg_salary': avg_salary,
        'recommended_offer': recommended_offer,
        'experience_level': experience_level,
        'city_multiplier': city_multiplier,
        'company_multiplier': company_multiplier,
        'skills_bonus': skills_bonus,
        'currency': 'INR',
        'per': 'annum'
    }


def format_indian_salary(amount):
    """Format salary in Indian numbering system (Lakhs/Crores)"""
    if amount >= 10000000:  # 1 Crore+
        return f"₹{amount/10000000:.2f} Cr"
    elif amount >= 100000:  # 1 Lakh+
        return f"₹{amount/100000:.2f} LPA"
    else:
        return f"₹{amount:,}"


def format_salary_dict(salary_dict):
    """Format entire salary dictionary for display"""
    return {
        'range': f"{format_indian_salary(salary_dict['min_salary'])} - {format_indian_salary(salary_dict['max_salary'])}",
        'average': format_indian_salary(salary_dict['avg_salary']),
        'recommended': format_indian_salary(salary_dict['recommended_offer']),
        'monthly_avg': format_indian_salary(salary_dict['avg_salary'] // 12),
        'monthly_recommended': format_indian_salary(salary_dict['recommended_offer'] // 12)
    }


def get_salary_breakdown(salary_dict):
    """Get formatted salary breakdown with components"""
    return {
        'base': format_indian_salary(salary_dict['avg_salary']),
        'city_adjustment': f"+{int((salary_dict['city_multiplier'] - 1) * 100)}%",
        'company_multiplier': f"{salary_dict['company_multiplier']}x",
        'skills_bonus': f"+{int((salary_dict['skills_bonus'] - 1) * 100)}%"
    }
