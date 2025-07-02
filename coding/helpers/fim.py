import random
from typing import Tuple


def insert_fim_hole(code: str) -> Tuple[str, str]:
    lines = code.splitlines()
    if len(lines) < 2:
        return code, []

    # Determine the maximum possible size of the hole (between 1 and 15 lines)
    max_hole_size = min(15, len(lines))

    # Randomly select the start index and the size of the hole
    start_index = random.randint(0, len(lines) - 1)
    hole_size = random.randint(1, max_hole_size)

    # Ensure the hole does not exceed the bounds of the code
    end_index = min(start_index + hole_size - 1, len(lines) - 1)

    # Extract the selected lines
    replaced_lines = lines[start_index : end_index + 1]

    # Replace the selected lines with "<|fim_hole|>"
    lines[start_index : end_index + 1] = ["<|fim_hole|>"]

    # Reconstruct the code
    new_code = "\n".join(lines)

    return new_code, "\n".join(replaced_lines)
