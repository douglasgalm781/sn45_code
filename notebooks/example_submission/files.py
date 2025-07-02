import os
from typing import List

def load_directory(directory: str) -> List[str]:
    # Create repo_files dict from task.repo.path
    repo_files = {}

    # Walk through all files in repo path
    for root, dirs, files in os.walk(directory):
        # Skip __pycache__ directories
        if '__pycache__' in dirs:
            dirs.remove('__pycache__')
            
        # Get relative path from repo root
        rel_path = os.path.relpath(root, directory)
        
        # Process all files
        for filename in files:
            # Skip __pycache__ files
            if '__pycache__' in filename:
                continue
                
            file_path = os.path.join(root, filename)
            
            # Get the relative path for the repo_files dict key
            if rel_path == '.':
                repo_key = filename
            else:
                repo_key = os.path.join(rel_path, filename)
                
            # Read file contents
            with open(file_path, 'r', encoding='latin-1') as f:
                repo_files[repo_key] = f.read()
    return repo_files