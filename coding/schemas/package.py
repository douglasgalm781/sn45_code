import re
import random
import string
from typing import List, Dict
from pydantic import BaseModel

from .file import File


class Package(BaseModel):
    files: List[File]

    def update_file(self, new_file: File):
        for i, file in enumerate(self.files):
            if file.path == new_file.path:
                self.files[i] = new_file
                return
        raise ValueError(f"File with path {new_file.path} not found in package.")


class ObscurePackage(Package):
    mapping: Dict = {}

    def obscure_package(self):
        mapping = {}

        # Obscure file paths
        for file in self.files:
            new_path = self._generate_random_string(len(file.path)) + ".py"
            mapping[file.path] = new_path
            file.path = new_path

        # Obscure classes and contents
        for file in self.files:
            file.content, class_mapping = self._obscure_classes(file.content)
            mapping.update(class_mapping)
            file.content = self._obscure_contents(file.content, mapping)

        self.mapping = mapping

    def undo_obscure_package(self):
        if hasattr(self, "mapping"):
            # Undo obscuring file paths
            reverse_mapping = {v: k for k, v in self.mapping.items()}
            for file in self.files:
                if file.path in reverse_mapping:
                    file.path = reverse_mapping[file.path]

            # Undo obscuring classes and contents
            for file in self.files:
                file.content = self._undo_obscure_contents(
                    file.content, reverse_mapping
                )
                file.content = self._undo_obscure_classes(file.content, reverse_mapping)

            del self.mapping

    def obscure_string(self, script: str):
        if not hasattr(self, "mapping"):
            raise ValueError(
                "Package must be obscured before obscuring a script string."
            )

        script, class_mapping = self._obscure_classes(script)
        script = self._obscure_contents(script, self.mapping)
        script = self._obscure_contents(script, class_mapping)
        return script

    def undo_obscure_string(self, script: str):
        if not hasattr(self, "mapping"):
            raise ValueError(
                "Package must be obscured before undoing obscuring a script string."
            )

        reverse_mapping = {v: k for k, v in self.mapping.items()}
        script = self._undo_obscure_contents(script, reverse_mapping)
        script = self._undo_obscure_classes(script, reverse_mapping)
        return script

    def _generate_random_string(self, length):
        return "".join(random.choices(string.ascii_letters + string.digits, k=length))

    def _generate_random_class_name(self, length):
        return "".join(
            random.choices(string.ascii_uppercase, k=1)
            + random.choices(string.ascii_lowercase, k=length - 1)
        )

    def _obscure_contents(self, contents, mapping):
        for original, obscure in mapping.items():
            contents = re.sub(r"\b" + re.escape(original) + r"\b", obscure, contents)
        return contents

    def _undo_obscure_contents(self, contents, reverse_mapping):
        for obscure, original in reverse_mapping.items():
            contents = re.sub(r"\b" + re.escape(obscure) + r"\b", original, contents)
        return contents

    def _obscure_classes(self, contents):
        class_pattern = r"\bclass\s+(\w+)"
        class_names = re.findall(class_pattern, contents)
        class_mapping = {}
        for class_name in class_names:
            new_class_name = self._generate_random_class_name(len(class_name))
            class_mapping[class_name] = new_class_name
            contents = re.sub(
                r"\b" + re.escape(class_name) + r"\b", new_class_name, contents
            )
        return contents, class_mapping

    def _undo_obscure_classes(self, contents, reverse_mapping):
        class_names = list(reverse_mapping.keys())
        for obscure_name in class_names:
            original_name = reverse_mapping[obscure_name]
            contents = re.sub(
                r"\b" + re.escape(obscure_name) + r"\b", original_name, contents
            )
        return contents
