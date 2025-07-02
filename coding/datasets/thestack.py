# The MIT License (MIT)
# Copyright © 2024 Yuma Rao
# Copyright © 2023 Opentensor Foundation
# Copyright © 2024 Macrocosmos

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
import re
import boto3
import random
import itertools
import numpy as np
from smart_open import open
from datasets import load_dataset, Dataset, interleave_datasets

from .base import Dataset
from coding.schemas import Context
from coding.helpers.selector import Selector

LANGUAGES = {
    "C++": {
        "keywords": [
            "auto",
            "break",
            "case",
            "char",
            "const",
            "continue",
            "default",
            "do",
            "double",
            "else",
            "enum",
            "extern",
            "float",
            "for",
            "goto",
            "if",
            "int",
            "long",
            "register",
            "return",
            "short",
            "signed",
            "sizeof",
            "static",
            "struct",
            "switch",
            "typedef",
            "union",
            "unsigned",
            "void",
            "volatile",
            "while",
        ],
        "libraries": [
            "iostream",
            "fstream",
            "string",
            "vector",
            "map",
            "set",
            "algorithm",
            "cmath",
            "cstdio",
            "cstdlib",
            "ctime",
            "cstring",
            "cassert",
            "cctype",
            "cerrno",
            "cfloat",
            "ciso646",
            "climits",
            "clocale",
            "cmath",
            "csetjmp",
            "csignal",
            "cstdarg",
            "cstddef",
            "cstdio",
            "cstdlib",
            "cstring",
            "ctime",
            "cwchar",
            "cwctype",
            "complex",
            "deque",
            "exception",
            "fstream",
            "functional",
            "iomanip",
            "ios",
            "iosfwd",
            "iostream",
            "istream",
            "iterator",
            "limits",
            "list",
            "locale",
            "map",
            "memory",
            "new",
            "numeric",
            "ostream",
            "queue",
            "set",
            "sstream",
            "stack",
            "stdexcept",
            "streambuf",
            "string",
            "typerow",
            "utility",
            "valarray",
            "vector",
        ],
        "comments": ["//", "/*", "*/"],
        "multiline_comments": [("/*", "*/")],
    },
    "Dockerfile": {
        "keywords": [
            "from",
            "maintainer",
            "run",
            "cmd",
            "expose",
            "env",
            "add",
            "copy",
            "entrypoint",
            "volume",
            "user",
            "workdir",
            "onbuild",
        ],
        "libraries": [],
        "comments": ["#"],
        "multiline_comments": [],
    },
    "HTML": {
        "keywords": [
            "div",
            "span",
            "input",
            "ul",
            "body",
            "tag",
            "html",
            "head",
            "title",
            "meta",
            "link",
            "script",
            "style",
            "a",
            "img",
            "table",
            "label",
        ],
        "libraries": [],
        "comments": ["<!--", "-->"],
        "multiline_comments": [("<!--", "-->")],
    },
    "Java": {
        "keywords": [
            "abstract",
            "assert",
            "boolean",
            "break",
            "byte",
            "case",
            "catch",
            "char",
            "class",
            "continue",
            "default",
            "do",
            "double",
            "else",
            "enum",
            "extends",
            "final",
            "finally",
            "float",
            "for",
            "if",
            "implements",
            "import",
            "instanceof",
            "int",
            "interface",
            "long",
            "native",
            "new",
            "package",
            "private",
            "protected",
            "public",
            "return",
            "short",
            "static",
            "strictfp",
            "super",
            "switch",
            "synchronized",
            "this",
            "throw",
            "throws",
            "transient",
            "try",
            "void",
            "volatile",
            "while",
        ],
        "libraries": [
            "java.awt",
            "java.awt.event",
            "java.io",
            "java.lang",
            "java.math",
            "java.net",
            "java.text",
            "java.util",
            "javax.swing",
        ],
        "comments": ["//", "/*", "*/", "*"],
        "multiline_comments": [("/*", "*/")],
    },
    "JavaScript": {
        "keywords": [
            "abstract",
            "arguments",
            "boolean",
            "break",
            "byte",
            "case",
            "catch",
            "char",
            "class",
            "const",
            "continue",
            "debugger",
            "default",
            "delete",
            "do",
            "double",
            "else",
            "enum",
            "eval",
            "export",
            "extends",
            "false",
            "final",
            "finally",
            "float",
            "for",
            "function",
            "goto",
            "if",
            "implements",
            "import",
            "in",
            "instanceof",
            "int",
            "interface",
            "let",
            "long",
            "native",
            "module.exports" "new",
            "null",
            "package",
            "private",
            "protected",
            "public",
            "return",
            "short",
            "static",
            "super",
            "switch",
            "synchronized",
            "this",
            "throw",
            "throws",
            "transient",
            "true",
            "try",
            "typeof",
            "var",
            "void",
            "volatile",
            "while",
            "with",
            "yield",
        ],
        "libraries": [
            "react",
            "express",
            "mongoose",
            "axios",
            "redux",
            "react-redux",
            "react-router-dom",
            "react-dom",
            "react-scripts",
            "material-ui",
        ],
        "comments": ["//", "/*", "*/"],
        "multiline_comments": [("/*", "*/")],
    },
    "Python": {
        "keywords": [
            "False",
            "None",
            "True",
            "and",
            "as",
            "assert",
            "break",
            "class",
            "continue",
            "def",
            "del",
            "elif",
            "else",
            "except",
            "finally",
            "for",
            "from",
            "global",
            "if",
            "import",
            "in",
            "is",
            "lambda",
            "nonlocal",
            "not",
            "or",
            "pass",
            "raise",
            "return",
            "try",
            "while",
            "with",
            "yield",
        ],
        "libraries": [
            "numpy",
            "pandas",
            "matplotlib",
            "seaborn",
            "scipy",
            "sklearn",
            "tensorflow",
            "keras",
            "pytorch",
            "django",
            "flask",
            "requests",
            "bs4",
            "selenium",
            "pyautogui",
            "pyperclip",
            "pyinputplus",
            "pillow",
        ],
        "comments": ["#"],
        "multiline_comments": [('"""', '"""'), ("'''", "'''")],
    },
    "SQL": {
        "keywords": [
            "add",
            "all",
            "alter",
            "and",
            "any",
            "as",
            "asc",
            "backup",
            "between",
            "case",
            "check",
            "column",
            "constraint",
            "create",
            "database",
            "default",
            "delete",
            "desc",
            "distinct",
            "drop",
            "exec",
            "exists",
            "foreign",
            "from",
            "full",
            "group",
            "having",
            "in",
            "index",
            "inner",
            "insert",
            "into",
            "is",
            "join",
            "key",
            "left",
            "like",
            "limit",
            "not",
            "null",
            "on",
            "or",
            "order",
            "outer",
            "primary",
            "procedure",
            "right",
            "rownum",
            "select",
            "set",
            "table",
            "top",
            "truncate",
            "union",
            "unique",
            "update",
            "values",
            "view",
            "where",
        ],
        "comments": ["--", "/*", "*/"],
    },
    "Shell": {
        "keywords": [
            "alias",
            "bg",
            "bind",
            "break",
            "builtin",
            "caller",
            "cd",
            "command",
            "compgen",
            "complete",
            "continue",
            "declare",
            "dirs",
            "disown",
            "echo",
            "enable",
            "eval",
            "exec",
            "exit",
            "export",
            "false",
            "fc",
            "fg",
            "getopts",
            "hash",
            "help",
            "history",
            "jobs",
            "kill",
            "let",
            "local",
            "logout",
            "popd",
            "printf",
            "pushd",
            "pwd",
            "read",
            "readonly",
            "return",
            "set",
            "shift",
            "shopt",
            "source",
            "suspend",
            "test",
            "times",
            "trap",
            "true",
            "type",
            "typeset",
            "ulimit",
            "umask",
            "unalias",
            "unset",
            "wait",
        ],
        "comments": ["#"],
        "multiline_comments": [(":'", "'")],
    },
}


def convert_to_python3(code: str) -> str:
    """
    Convert Python 2/3 code to Python 3 code.

    Args:
    - code (str): A string containing Python 2/3 code.

    Returns:
    - str: A string containing Python 3 code.
    """

    def replace_print_statement(match):
        return f"print({match.group(1)})"

    code = re.sub(r"print (.*)", replace_print_statement, code)

    # Replace xrange with range
    code = code.replace("xrange", "range")

    return code


def process_repo_row(row):
    for file in row["files"]:
        blob_id = file["blob_id"]
        src_encoding = file["src_encoding"]
        session = boto3.Session(
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        )
        s3 = session.client("s3")
        s3_url = f"s3://softwareheritage/content/{blob_id}"

        with open(
            s3_url, "rb", compression=".gz", transport_params={"client": s3}
        ) as fin:
            file["content"] = fin.read().decode(src_encoding)

    return row


def process_row(row):
    blob_id = row["blob_id"]
    src_encoding = row["src_encoding"]
    session = boto3.Session(
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )
    s3 = session.client("s3")
    s3_url = f"s3://softwareheritage/content/{blob_id}"

    with open(s3_url, "rb", compression=".gz", transport_params={"client": s3}) as fin:
        content = fin.read().decode(src_encoding)

    row["code"] = content
    return row


def filter_comments(code, language):
    if language not in LANGUAGES:
        return code
    # Filter out multiline comments
    if "multiline_comments" in LANGUAGES[language]:
        for start_tag, end_tag in LANGUAGES[language]["multiline_comments"]:
            code = re.sub(
                rf"{re.escape(start_tag)}.*?{re.escape(end_tag)}",
                "",
                code,
                flags=re.DOTALL,
            )

    # Filter out single-line comments
    lines = []
    for line in code.splitlines():
        if any(
            line.strip().startswith(symbol)
            for symbol in LANGUAGES[language]["comments"]
        ):
            continue
        lines.append(line.lower())

    return "\n".join(lines)


class TheStackDataset(Dataset):
    name = "thestack"

    def __init__(
        self,
        seed=None,
        languages=None,
    ):

        if seed is None:
            seed = random.randint(0, 1000)
        self.seed = seed

        if languages is None:
            languages = list(LANGUAGES.keys())
        self.languages = languages

        # self.dataset = cache_dataset(dataset_id=dataset_id, seed=seed)
        datasets = []
        for language in [
            "Python",
            # "JavaScript",
            # "TypeScript",
            # "Go",
            # "Java",
            # "C++",
            # "C",
            # "SQL",
            # "Shell",
        ]:
            datasets.append(
                load_dataset(
                    "bigcode/the-stack-v2",
                    language,
                    split="train",
                    streaming=True,
                )
            )
        # shuffle the datasets
        for dataset in datasets:
            dataset = dataset.shuffle()
        self.stack_dataset = interleave_datasets(datasets)
        self.stack_dataset = self.stack_dataset.shuffle()
        self.stack_dataset = self.stack_dataset.map(lambda row: process_row(row))
        self.stack_iterset = iter(self.stack_dataset)

        self.stack_repo_dataset = load_dataset(
            "bigcode/the-stack-v2-train-smol-ids", split="train", streaming=True
        )
        self.stack_repo_dataset = self.stack_repo_dataset.shuffle()
        self.stack_repo_iterset = iter(self.stack_repo_dataset)

    def random(
        self,
        min_lines=10,
        max_lines=3000,
        selector: Selector = None,
        include_sibling_docs=False,
        min_sibling_docs=1,
        **kwargs,
    ):
        return self.get(
            min_lines,
            max_lines,
            selector,
            include_sibling_docs,
            min_sibling_docs,
            **kwargs,
        )

    def get(
        self,
        min_lines=25,
        max_lines=3000,
        selector: Selector = None,
        include_sibling_docs=False,
        min_sibling_docs=1,
        **kwargs,
    ):
        content = None
        if include_sibling_docs:
            row = next(self.stack_repo_iterset)
            if not row["gha_language"]:
                row["gha_language"] = ""
        else:
            row = next(self.stack_iterset)
            if not (min_lines <= len(row["code"].splitlines()) <= max_lines):
                return None
            content = row["code"]

        sibling_docs = []
        if include_sibling_docs:
            if (
                row["num_files"] < min_sibling_docs
                or row["num_files"] > 15  # TODO modify this eventually to be different
                or len(row["files"]) < 2
            ):
                return None
            row = process_repo_row(row)
            randindex = random.randint(1, len(row["files"]) - 1)
            # choose all but the random index
            for file in row["files"][:randindex] + row["files"][randindex + 1 :]:
                sibling_docs.append(
                    Context(
                        title=file["path"],
                        content=file["content"],
                        topic=row["gha_language"],
                    )
                )
            content = row["files"][randindex]["content"]

        if ("language" in row and row["language"] == "Python") or (
            "gha_language" in row and row["gha_language"] == "Python"
        ):
            content = convert_to_python3(content)

        if (
            len(content.splitlines()) < min_lines
            or len(content.splitlines()) > max_lines
        ):
            return None

        for sibling_doc in sibling_docs:
            if (
                len(sibling_doc.content.splitlines()) < min_lines
                or len(sibling_doc.content.splitlines()) > max_lines
            ):
                return None

        return {
            "title": row["repo_name"],  # name of the repo
            "topic": (
                row["language"] if "language" in row else row["gha_language"]
            ),  # language of the code
            "subtopic": "",
            "content": filter_comments(
                content, row["language"] if "language" in row else row["gha_language"]
            ),
            "internal_links": [row["repo_name"]],
            "external_links": [],
            "source": "GitHub",
            "tags": [
                row["language"] if "language" in row else row["gha_language"],
                row["repo_name"],
                "",
            ],
            "extras": {
                "sibling_docs": sibling_docs,
            },
        }

    def search(
        self,
        query,
        column="path",
        min_lines=5,
        max_lines=100,
        selector: Selector = None,
        **kwargs,
    ):
        mask = np.array(self.dataset[column]) == query
        filtered_dataset = iter(self.dataset.select(np.where(mask)[0]))

        return [
            {
                "title": row["repo_name"],  # name of the repo
                "topic": row["language"],  # language of the code
                "subtopic": row["path"],
                "content": (
                    convert_to_python3(filter_comments(row["code"], row["language"]))
                    if row["language"] == "Python"
                    else filter_comments(row["code"], row["language"])
                ),
                "internal_links": [row["repo_name"], row["path"], row["language"]],
                "external_links": [],  # TODO complete
                "source": "GitHub",
                "tags": [row["language"], row["repo_name"], row["path"]],
                "extras": {"size": row["size"], "license": row["license"]},
            }
            for row in filtered_dataset
        ]

    def extract_keywords(self, code, language, field):
        matches = set()

        # check which keywords and libraries are present in the code
        for keyword in LANGUAGES[language].get(field, []):
            if re.search(r"\b" + keyword + r"\b", code):
                matches.add(keyword)

        return matches

    def get_special_contents(self, code, language, remove_comments=True):
        if remove_comments:
            code = filter_comments(code, language)

        present_libraries = self.extract_keywords(code, language, "libraries")
        present_keywords = self.extract_keywords(code, language, "keywords")

        return present_keywords, present_libraries
