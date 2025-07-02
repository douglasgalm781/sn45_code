import os
from coding.protocol import LogicSynapse


def miner_process(self, synapse: LogicSynapse) -> LogicSynapse:
    """
    The miner process function is called every time the miner receives a request. This function should contain the main logic of the miner.
    """
    logic = {}
    test_submission_dir = ""

    # Read all files in test-submission directory
    for root, dirs, files in os.walk(test_submission_dir):
        # Skip __pycache__ directories
        if "__pycache__" in dirs:
            dirs.remove("__pycache__")

        # Get relative path from test_submission_dir
        rel_path = os.path.relpath(root, test_submission_dir)

        # Process all files in current directory
        for filename in files:
            # Skip __pycache__ files
            if "__pycache__" in filename:
                continue

            file_path = os.path.join(root, filename)
            # Get the relative path for the logic dict key
            if rel_path == ".":
                logic_key = filename
            else:
                logic_key = os.path.join(rel_path, filename)

            with open(file_path, "r", encoding="latin-1") as f:
                logic[logic_key] = f.read()
    synapse.logic = logic
    return synapse
