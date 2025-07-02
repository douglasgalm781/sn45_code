import random

REWRITE_REASONS = [
    "more concise",
    "more verbose",
    "more pythonic",
    "more efficient",
    "more readable",
    "more correct",
    "more efficient",
    "a little different",
    "super concise",
    "super verbose",
    "super pythonic",
    "super efficient",
    "super readable",
    "super correct",
]


def rewrite_code(code: str, model: str) -> str:
    res = model.invoke(
        f"Rewrite the following code to be {random.choice(REWRITE_REASONS)}, make sure it does the same thing though: {code}"
    ).content

    if "```" in res:
        start = res.find("```") + 3  # Skip the backticks and newline
        start = res.find("\n", start) + 1

        end = res.rfind("```")
        res = res[start:end].strip()
    return res
