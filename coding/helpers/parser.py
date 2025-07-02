import re


def extract_python_code(markdown_string):
    """
    Extracts Python code blocks from a Markdown string.

    Parameters:
        markdown_string (str): The Markdown string to extract Python code from.

    Returns:
        list of str: A list of extracted Python code blocks.
    """
    # Regular expression to match Python code blocks
    python_code_pattern = re.compile(r"```python\n(.*?)\n```", re.DOTALL)

    # Find all Python code blocks
    python_code_blocks = python_code_pattern.findall(markdown_string)

    return python_code_blocks
