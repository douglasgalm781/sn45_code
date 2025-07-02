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

import io
import os
import math
import random
import tarfile
import requests

from typing import List
from pydantic import BaseModel

from .base import Dataset
from coding.schemas import Context
from coding.schemas import File
from coding.helpers.selector import Selector


def fetch_pip_repo_contents(
    repo_name: str, size_limit: int = 10 * 1024 * 1024
) -> List[File]:
    """
    Fetch the contents of a pip repository as a list of file objects.

    Parameters:
    - repo_name: The name of the pip repository.
    - size_limit: The maximum allowable size of the tarball in bytes.

    Returns:
    - A list of FileObject instances representing the files in the repository.

    Raises:
    - ValueError if the tarball size exceeds the specified limit or if there are issues fetching data.
    """
    # Fetch the latest release metadata from PyPI
    pypi_url = f"https://pypi.org/pypi/{repo_name}/json"
    response = requests.get(pypi_url)
    if response.status_code != 200:
        raise ValueError(f"Could not fetch repository data for {repo_name}")

    data = response.json()
    latest_version = data["info"]["version"]
    tarball_url = data["releases"][latest_version][-1]["url"]

    # Get the size of the tarball without downloading it
    head_response = requests.head(tarball_url)
    if head_response.status_code != 200:
        raise ValueError(f"Could not fetch tarball metadata for {repo_name}")

    content_length = int(head_response.headers.get("Content-Length", 0))
    if content_length > size_limit:
        raise ValueError(
            f"Tarball size ({content_length} bytes) exceeds the limit of {size_limit} bytes"
        )

    # Download the tarball of the latest release
    tarball_response = requests.get(tarball_url)
    if tarball_response.status_code != 200:
        raise ValueError(f"Could not fetch tarball for {repo_name}")

    # Read the tarball contents
    tarball_file = io.BytesIO(tarball_response.content)
    tar = tarfile.open(fileobj=tarball_file)

    file_objects = []
    for member in tar.getmembers():
        if member.isfile():
            f = tar.extractfile(member)
            if f is not None:
                contents = f.read().decode("utf-8")
                # split the name to remove the package name
                file_objects.append(
                    File(path="/".join(member.name.split("/")[1:]), contents=contents)
                )

    return file_objects


def get_pip_repo_size(repo_name: str) -> int:
    """
    Get the size of the latest tarball for a given pip repository.

    Parameters:
    - repo_name: The name of the pip repository.

    Returns:
    - The size of the latest tarball in bytes.

    Raises:
    - ValueError if the repository data or tarball metadata cannot be fetched.
    """

    # Fetch the latest release metadata from PyPI
    pypi_url = f"https://pypi.org/pypi/{repo_name}/json"
    response = requests.get(pypi_url)
    if response.status_code != 200:
        raise ValueError(f"Could not fetch repository data for {repo_name}")

    data = response.json()
    latest_version = data["info"]["version"]
    tarball_url = data["releases"][latest_version][-1]["url"]

    # Get the size of the tarball without downloading it
    head_response = requests.head(tarball_url)
    if head_response.status_code != 200:
        raise ValueError(f"Could not fetch tarball metadata for {repo_name}")

    content_length = int(head_response.headers.get("Content-Length", 0))

    return content_length


def get_total_pip_packages():
    url = "https://libraries.io/api/search"
    params = {
        "platforms": "pypi",
        "sort": "dependents_count",
        "per_page": 1,  # Get only one result to find out the total count
        "api_key": os.getenv(
            "LIBRARIES_API_KEY", "45cc24a495c25a68a052e3f99af9a05a"
        ),  # TODO remove the api key
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    total_packages = int(response.headers.get("total", 0))
    return total_packages


def get_random_packages(n=100):
    url = "https://libraries.io/api/search"
    total_packages = get_total_pip_packages()
    total_pages = math.ceil(total_packages / n)
    random_offset = random.randint(0, total_pages - n)

    params = {
        "platforms": "pypi",
        "sort": "dependents_count",
        "per_page": n,
        "offset": random_offset,
        "api_key": os.getenv(
            "LIBRARIES_API_KEY", "45cc24a495c25a68a052e3f99af9a05a"
        ),  # TODO remove the api key
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    return [package["name"] for package in data]


class PipDataset(Dataset):
    name = "pip"

    def __init__(
        self,
        seed=None,
    ):
        if seed is None:
            seed = random.randint(0, 1000)
        self.seed = seed

    def get(self, n=100, selector: Selector = None):
        for _ in range(300):

            packages = get_random_packages(n=n)
            package_name = selector(packages)
            if not get_pip_repo_size(package_name) < 10 * 1024 * 1024:  # 10MB
                continue
            return dict(
                title=package_name,
                source="pip",
                # files= fetch_pip_repo_contents(package_name)
            )
        raise Exception("Failed to find a valid pip package")

    def search(self, query, selector: Selector = None, **kwargs):
        pass

    def random(self, n=100, selector: Selector = None, **kwargs):
        return self.get(n=100, selector=selector)
