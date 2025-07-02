from typing import List, Dict
FIX_PROMPT = """
Given the following file and the issue, rewrite the file to fix the issue. If no issue is found, respond with nothing.

File: {file}

Issue: {issue}
"""


def fix(files: Dict[str, str], file_names: List[str], issue: str, llm) -> Dict[str, str]:
    fixed_files = {}
    for file_name in file_names:
        prompt = FIX_PROMPT.format(file=files[file_name], issue=issue)
        response, _ = llm(prompt, "gpt-4o")
        
        # Extract code block if present
        if "```python" in response:
            start = response.find("```python") + len("```python")
            end = response.find("```", start)
            response = response[start:end]
        elif "```" in response:
            start = response.find("```") + len("```")
            end = response.find("```", start)
            response = response[start:end]
            
        if response:
            fixed_files[file_name] = response.strip()
            
    return fixed_files