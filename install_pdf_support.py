"""
Install PDF support for NeuroMatch AI
"""

import subprocess
import sys

def install_pdf_support():
    """Install required packages for PDF support"""
    packages = [
        'PyPDF2',
        'pypdf',
        'pdfplumber'
    ]
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")
            try:
                # Try alternative package
                if package == 'PyPDF2':
                    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pypdf'])
                    print(f"✅ pypdf installed as alternative")
            except:
                print(f"❌ Alternative installation also failed")

if __name__ == "__main__":
    install_pdf_support()
    print("\n🎉 PDF support installation completed!")
    print("Now you can run: streamlit run simple_app.py")
