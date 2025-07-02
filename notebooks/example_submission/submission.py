from fix import fix
from search import search
from diff import create_patch
from files import load_directory
from swebase import SWEBase, Patch


class SWE(SWEBase):
    def __call__(self, repo_location: str, issue_description: str) -> Patch:
        print(f"Searching for relevant files for issue: {issue_description}")
        file_names = search(repo_location, issue_description, self.llm)
        print(f"Found relevant files: {file_names}")
        
        print(f"Loading files from directory: {repo_location}")
        files = load_directory(repo_location)
        print(f"Loaded {len(files)} files")
        
        print("Fixing files...")
        fixed_files = fix(files, file_names, issue_description, self.llm)
        print(f"Fixed {len(fixed_files)} files")
        
        print("Creating patch...")
        patch = create_patch(files, fixed_files)
        print("Patch created")
        return patch
