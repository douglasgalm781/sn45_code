import ast
from typing import List

SEARCH_PROMPT = """
Given the following file names, find the file that contains the code that is relevant to the issue.

{file_names}

Issue: {issue}

Your response should be a python list of file names.
"""

def search(file_names: List[str], issue: str, llm) -> str:
    prompt = SEARCH_PROMPT.format(file_names=file_names, issue=issue)
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
        
    # Clean and parse the response
    response = response.strip()
    try:
        # Safely evaluate the string as a Python literal
        import ast
        files = ast.literal_eval(response)
        if not isinstance(files, list):
            files = [files]
    except:
        # Fallback to basic string parsing if eval fails
        files = response.replace("[", "").replace("]", "").replace("'", "").replace("\"", "").split(",")
        files = [f.strip() for f in files if f.strip()]
        
    return files