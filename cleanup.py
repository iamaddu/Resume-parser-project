"""Clean up duplicate code from simple_app.py"""

# Read the file
with open(r'c:\Users\harsh\OneDrive\Desktop\aimlcie3\simple_app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the first occurrence of "if __name__ == "__main__":"
main_line = None
for i, line in enumerate(lines):
    if 'if __name__ == "__main__":' in line:
        main_line = i
        break

if main_line:
    # Keep everything up to and including the main() call (2 lines after if __name__)
    clean_lines = lines[:main_line + 2]
    
    # Write back
    with open(r'c:\Users\harsh\OneDrive\Desktop\aimlcie3\simple_app.py', 'w', encoding='utf-8') as f:
        f.writelines(clean_lines)
    
    print(f"SUCCESS: File cleaned! Kept {len(clean_lines)} lines, removed {len(lines) - len(clean_lines)} duplicate lines")
else:
    print("ERROR: Could not find main block")
