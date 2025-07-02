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
import random
import requests

from fastcore.xtras import obj2dict
from swebench.collect.build_dataset import create_instance

from .base import Dataset
from coding.helpers.selector import Selector
from coding.helpers.swebench import Repo as SWERepo


def get_package_stats(package_name: str):
    package_url = f"https://pypi.org/pypi/{package_name}/json"
    package_github = None
    response = requests.get(package_url)
    if response.status_code != 200:
        raise Exception(f"Failed to get package data from URL: {package_url}")
    response = response.json()
    if "info" in response:
        if (
            "Source" in response["info"]["project_urls"]
            and "github" in response["info"]["project_urls"]["Source"]
        ):
            package_github = response["info"]["project_urls"]["Source"]
        elif (
            "Homepage" in response["info"]["project_urls"]
            and "github" in response["info"]["project_urls"]["Homepage"]
        ):
            package_github = response["info"]["project_urls"]["Homepage"]
    if not package_github:
        raise Exception(f"No github link found for package: {package_name}")

    return {
        "name": package_name,
        "url": package_url,
        "github": package_github,
    }


def get_top_pip_packages():
    response = requests.get(
        "https://hugovk.github.io/top-pypi-packages/top-pypi-packages-30-days.min.json"
    )
    packages = [row["project"] for row in response.json()["rows"]]
    return packages


class SWEBenchDataset(Dataset):
    name = "swebench"

    def __init__(
        self,
    ):
        pass

    def get(self, n=100, selector: Selector = Selector()) -> dict:
        package_name = selector(get_top_pip_packages())
        package_info = get_package_stats(package_name)
        token = os.environ.get("GITHUB_TOKEN", None)
        if not token:
            raise Exception("GITHUB_TOKEN not set")
        repo = SWERepo(
            package_info["github"].split("/")[-2],
            package_info["github"].split("/")[-1],
            token,
        )

        # Check repo size before proceeding
        if repo.size > 1024 * 1024 * 1024:  # 1GB in bytes
            raise Exception(f"Repository {package_info['github']} is too large (>1GB)")

        valid_pull = None
        err_count = 0
        pulls = [pull for pull in repo.get_all_pulls(state="closed")]
        random.shuffle(pulls)
        for pull in pulls:
            try:
                if valid_pull or err_count > 5:
                    break
                resolved_issues = repo.extract_resolved_issues(pull)
                setattr(pull, "resolved_issues", resolved_issues)
                if len(resolved_issues) > 0:
                    valid_pull = obj2dict(pull)
            except:
                err_count += 1

        if not valid_pull:
            raise Exception(
                f"Could not get a valid SWE pull for {package_info['github']}"
            )
        pull_data = create_instance(repo, valid_pull)
        diff_text = pull_data["patch"]
        return {
            "topic": pull_data["problem_statement"],
            "title": f'{package_info["github"].split("/")[-2]}/{package_info["github"].split("/")[-1]}',
            "content": diff_text,
            "extras": dict(
                pull_number=pull_data["pull_number"],
                base_commit=pull_data["base_commit"],
            ),
        }

    def search(self, query, selector: Selector = None, **kwargs):
        pass

    def random(self, n=100, selector: Selector = None, **kwargs):
        return self.get(n=100, selector=selector)
