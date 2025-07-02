from difflib import unified_diff
from typing import Dict
from swebase import Patch, Edit

def create_patch(original_files: Dict[str, str], edited_files: Dict[str, str]) -> Patch:
    """
    Create a Patch by comparing original and edited file contents line by line.
    For lines that differ or are added, an Edit is created.
    """
    edits = []
    for filename in edited_files:
        if filename not in original_files:
            continue
            
        old_lines = original_files[filename].splitlines()
        new_lines = edited_files[filename].splitlines()
        
        # Use the maximum length in case lines were added or removed
        max_lines = max(len(old_lines), len(new_lines))
        for i in range(max_lines):
            old_line = old_lines[i] if i < len(old_lines) else ""
            new_line = new_lines[i] if i < len(new_lines) else ""
            if old_line != new_line:
                edits.append(Edit(
                    file_name=filename,
                    line_number=i,
                    line_content=old_line,
                    new_line_content=new_line
                ))
    return Patch(edits=edits)