import ast
from typing import List, Dict
from coding.constants import DISALLOWED_IMPORTS

def check_nonvalid_imports(
    code: str, unallowed_imports: List[str] = DISALLOWED_IMPORTS
) -> tuple[bool, str]:
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in unallowed_imports:
                        return False, f"Disallowed import: {alias.name}"
    except Exception as e:
        return False, f"Error during parsing: {e}"
    return True, "Code is safe"

def verify_code_usage(
    code: str, allowed_modules: List[str], allowed_imports: Dict[str, List[str]]
) -> tuple[bool, str]:
    try:
        tree = ast.parse(code)
        imported_modules = set()
        imported_names = {}  # Track what names were imported from each module
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    # Only block import if module is in allowed_imports but used without restrictions
                    if (
                        alias.name in allowed_imports
                        and not allowed_imports[alias.name]
                    ):
                        return (
                            False,
                            f"Disallowed unrestricted use of module: {alias.name}",
                        )
                    if (
                        alias.name not in allowed_modules
                        and alias.name not in allowed_imports
                    ):
                        return False, f"Disallowed module: {alias.name}"
                    imported_modules.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if (
                    node.module not in allowed_modules
                    and node.module not in allowed_imports
                ):
                    return False, f"Disallowed module: {node.module}"
                # Track imported names from restricted modules
                if node.module in allowed_imports:
                    imported_names[node.module] = set()
                    for alias in node.names:
                        if alias.name not in allowed_imports[node.module]:
                            return (
                                False,
                                f"Disallowed import {alias.name} from module {node.module}",
                            )
                        imported_names[node.module].add(alias.name)
                imported_modules.add(node.module)
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    # Check if attribute access like os.getenv is allowed
                    if isinstance(node.func.value, ast.Name):
                        module_name = node.func.value.id
                        # Only check restricted functions if module was imported and has restrictions
                        if (
                            module_name in imported_modules
                            and module_name in allowed_imports
                        ):
                            if node.func.attr not in allowed_imports[module_name]:
                                return (
                                    False,
                                    f"Disallowed function {module_name}.{node.func.attr}",
                                )
                elif isinstance(node.func, ast.Name):
                    if node.func.id == "eval" or node.func.id == "exec":
                        return (
                            False,
                            f"Dangerous built-in function call: {node.func.id}",
                        )
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                if isinstance(node.value.func, ast.Name):
                    if node.value.func.id in ["eval", "exec"]:
                        return (
                            False,
                            f"Dangerous built-in function call: {node.value.func.id}",
                        )
        return True, "Code is safe"
    except Exception as e:
        return False, f"Error during parsing: {e}"


def check_large_literals(script, max_items=1000, max_length=10000):
    """
    Checks the provided script for large literal definitions.

    Parameters:
        script (str): A string containing a Python script.
        max_items (int): Maximum allowed number of items in a list, tuple, set, or dict.
        max_length (int): Maximum allowed length of a string literal.


    Returns:
        True if the script passes all checks. Else returns a tuple with False and the error message.
    """
    try:
        tree = ast.parse(script)
    except SyntaxError as e:
        # raise ValueError(f"Invalid Python script: {e}")
        return False, f"Invalid Python script: {e}"

    for node in ast.walk(tree):
        # Check dictionary literals
        if isinstance(node, ast.Dict):
            num_items = len(node.keys)
            if num_items > max_items:
                # raise ValueError(f"Dictionary literal with {num_items} items exceeds the limit of {max_items}.")
                return (
                    False,
                    f"Dictionary literal with {num_items} items exceeds the limit of {max_items}.",
                )

        # Check list, tuple, and set literals
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            num_items = len(node.elts)
            if num_items > max_items:
                literal_type = type(node).__name__
                # raise ValueError(f"{literal_type} literal with {num_items} items exceeds the limit of {max_items}.")
                return (
                    False,
                    f"{literal_type} literal with {num_items} items exceeds the limit of {max_items}.",
                )

        # For Python 3.8+, ast.Constant is used for literals like strings.
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                if len(node.value) > max_length:
                    # raise ValueError(f"String literal of length {len(node.value)} exceeds the limit of {max_length}.")
                    return (
                        False,
                        f"String literal of length {len(node.value)} exceeds the limit of {max_length}.",
                    )

        # For earlier versions of Python, string literals might be represented by ast.Str.
        elif isinstance(node, ast.Str):
            if len(node.s) > max_length:
                # raise ValueError(f"String literal of length {len(node.s)} exceeds the limit of {max_length}.")
                return (
                    False,
                    f"String literal of length {len(node.s)} exceeds the limit of {max_length}.",
                )
    return True, "Code is valid"
