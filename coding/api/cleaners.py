from detect_secrets.core import scan
from detect_secrets.settings import default_settings

FIM_PREFIXES = ["<fim_prefix>", "[PREFIX]", "<PRE>", "<|fim_begin|>"]
FIM_ENDS = ["<fim_middle>", "[SUFFIX]", "<SUF>", "<|fim_end|>"]
FIM_HOLES = ["<fim_suffix>"]


def clean_fixes(text):
    for prefix in FIM_PREFIXES:
        text = text.replace(prefix, "")
    for end in FIM_ENDS:
        text = text.replace(end, "")
    for hole in FIM_HOLES:
        text = text.replace(hole, "<|fim_hole|>")
    return text


def remove_secret_lines(multiline_string):
    # Split the input string into individual lines
    lines = multiline_string.split("\n")

    # Initialize a list to hold lines without secrets
    clean_lines = []

    # Scan each line for secrets
    with default_settings() as settings:
        settings.disable_plugins("Base64HighEntropyString", "HexHighEntropyString")
        for line in lines:
            is_secret = False
            for secret in scan.scan_line(line):
                is_secret = True
                break  # Exit the inner loop if a secret is found

            # If no secret is found, add the line to clean_lines
            if not is_secret:
                clean_lines.append(line)

    # Join the clean lines back into a single string
    return "\n".join(clean_lines)


def remove_generate_prompt(string):
    """
    Cleaner to remove the blocks that are used by continue.dev when running `Generate Code`
    """
    blocks = [
        "<|im_start|>user\n",
        "<|im_end|>\n",
        "<|im_start|>assistant\n",
        "Sure! Here's the entire rewritten code block:\n```python\n",
    ]
    for block in blocks:
        string = string.replace(block, "")

    return string
