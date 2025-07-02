import os
import shutil
import tempfile
import weakref
from git import Repo


class GitRepo:
    def __init__(self, repo_name: str, commit_hash: str):
        """
        Initialize a Git repository object that manages cloning and cleanup.

        Args:
            repo_name (str): Name/URL of the repository to clone
            commit_hash (str): Specific commit hash to checkout

        Raises:
            git.exc.GitCommandError: If repository does not exist or other git error occurs
        """
        self.repo_name = repo_name
        self.commit_hash = commit_hash
        self.temp_dir = tempfile.mkdtemp(prefix="git-repo-")
        self.repo = None
        self._initialize_repo()

    def _initialize_repo(self):
        """Initialize/reinitialize the git repository"""
        if (
            self.temp_dir
            and os.path.exists(self.temp_dir)
            and os.listdir(self.temp_dir)
        ):
            # self._finalizer = weakref.finalize(self, self._cleanup)
            return
        # Ensure repo name includes full GitHub URL if not already
        if not self.repo_name.startswith(("http://", "https://", "git://")):
            self.repo_name = f"https://github.com/{self.repo_name}"
        try:
            # Clone repo with minimal history and specific commit
            self.repo = Repo.clone_from(
                self.repo_name,
                self.temp_dir,
                depth=1,  # Only get most recent commit
                no_single_branch=True,  # Allow fetching specific commit
                no_tags=True,  # Don't fetch any tags
            )
        except:  # if clone fails try again
            self.repo = Repo.clone_from(
                self.repo_name,
                self.temp_dir,
                depth=1,  # Only get most recent commit
                no_single_branch=True,  # Allow fetching specific commit
                no_tags=True,  # Don't fetch any tags
            )
        # Fetch only the specific commit
        self.repo.git.fetch("origin", self.commit_hash, depth=1)
        self.repo.git.checkout(self.commit_hash)
        # Register cleanup to be called when object is deleted
        # self._finalizer = weakref.finalize(self, self._cleanup)

    def __getstate__(self):
        """Called when pickling - return state without repo objects"""
        state = self.__dict__.copy()
        # Remove unpicklable objects
        state["repo"] = None
        state["_finalizer"] = None
        return state

    def __setstate__(self, state):
        """Called when unpickling - restore state and reinitialize repo"""
        self.__dict__.update(state)
        if self.temp_dir == None:
            self.temp_dir = tempfile.mkdtemp(prefix="git-repo-")
        self._initialize_repo()

    def _cleanup(self):
        """
        Clean up the temporary directory containing the cloned repository.
        """
        try:
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")

    @property
    def path(self) -> str:
        """
        Get the path to the cloned repository.

        Returns:
            str: Path to the repository directory
        """
        return self.temp_dir

    @property
    def files(self) -> dict[str, str]:
        logic = {}
        # Read all files in test-submission directory
        for root, dirs, files in os.walk(self.path):
            # Skip __pycache__ directories
            if "__pycache__" in dirs:
                dirs.remove("__pycache__")

            # Get relative path from test_submission_dir
            rel_path = os.path.relpath(root, self.path)

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
        return logic

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cleanup()
