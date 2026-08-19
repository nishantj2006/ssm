import os

def build_dataset(source_folder, output_file):
    print(f"Scanning '{source_folder}' for Python files...")
    total_size = 0
    file_count = 0
    
    # Open the master text file
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # Walk through every folder and subfolder
        for root, dirs, files in os.walk(source_folder):
            # Skip hidden folders like .git or __pycache__
            if '.git' in root or '__pycache__' in root:
                continue
                
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as infile:
                            content = infile.read()
                            
                            # Add a comment header so the AI learns where files begin and end!
                            outfile.write(f"\n\n# {'='*40}\n")
                            outfile.write(f"# File: {file}\n")
                            outfile.write(f"# {'='*40}\n\n")
                            
                            outfile.write(content)
                            total_size += len(content)
                            file_count += 1
                    except Exception as e:
                        print(f"Skipping {file} due to read error: {e}")
                        
    size_mb = total_size / (1024 * 1024)
    print(f"\nDone! Stitched {file_count} Python files together.")
    print(f"Created {output_file} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    # Point this at the folder you just downloaded
    SOURCE_DIR = "./Python" 
    OUTPUT_NAME = "python_dataset.txt"
    
    if os.path.exists(SOURCE_DIR):
        build_dataset(SOURCE_DIR, OUTPUT_NAME)
    else:
        print(f"Error: Could not find the folder '{SOURCE_DIR}'. Did you clone the repository?")