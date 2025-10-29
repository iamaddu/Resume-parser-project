"""
NeuroMatch AI - Setup Configuration
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="neuromatch-ai",
    version="1.0.0",
    author="NeuroMatch AI Team",
    author_email="team@neuromatch-ai.com",
    description="Advanced AI-Powered Talent Matching & Career Analysis System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/neuromatch-ai/neuromatch-ai",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-cov>=4.1.0",
            "black>=23.11.0",
            "flake8>=6.1.0",
        ],
        "gpu": [
            "torch>=2.1.0+cu118",
        ],
    },
    entry_points={
        "console_scripts": [
            "neuromatch-app=web_app.neuromatch_app:main",
            "neuromatch-train=data.synthetic_training:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    zip_safe=False,
)
