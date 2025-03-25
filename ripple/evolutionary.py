import json
import re
import statistics
import subprocess
from pathlib import Path

from tqdm import tqdm

import pandas as pd

from ripple.parse_utils import GitHubRepository
from ripple.utils import (
    map_methods_ids_and_paths,
    remove_prefix_in_path,
    compute_metrics,
)


PATH_TO_ASSETS = "assets"


def get_patch_from_commit_hash(path_to_repo, commit):
    diff_args = ["git", "diff", f"{commit}^", commit, '-U0']
    # Get git diff with additional context lines
    patch = subprocess.run(
            diff_args,
            cwd=str(path_to_repo),
            check=True,
            capture_output=True,
    ).stdout.decode('utf-8', errors='replace')

    return patch


def extract_change_and_test_patches(patch):
    """Get change and test patches from a diff patch.

    Source:
    https://github.com/princeton-nlp/SWE-bench/blob/main/swebench/collect/utils.py

    Args:
        patch: Diff patch.
    
    Returns:
        change_text, test_text:  Change and test patch strings.
    """
    if patch.endswith("\n"):
        patch = patch[:-1]

    # Create change patch and test patch
    patch_change, patch_test = [], []

    # Flag to determine if current diff block is a test or general change
    # Values: 'test', 'diff', None
    flag = None

    for line in patch.split("\n"):
        # In swe-bench, commit specific metadata is omitted. We retain it.
        if line.startswith("index "):
            patch_change.append(line)

        # Determine if current diff block is a test or general change
        if line.startswith("diff --git a/"):
            words = set(re.split(r" |_|\/|\.", line.lower()))
            flag = (
                "test"
                if ("test" in words or "tests" in words or "testing" in words)
                else "diff"
            )
        
        # Append line to separate patch depending on flag status
        if flag == "test":
            patch_test.append(line)
        elif flag == "diff":
            patch_change.append(line)

    change_text = ""
    if patch_change:
        change_text = "\n".join(patch_change) + "\n"

    test_text = ""
    if patch_test:
        test_text = "\n".join(patch_test) + "\n"

    return change_text, test_text


def parse_repository_snapshot(repo_path, commit):
    repo_cg = GitHubRepository(repo_path, commit)
    all_methods = {
        idx: {
            'name': row['name'],
            'path': row['path'],
            'method_start': row['start_line'],
            'method_end': row['end_line'],
        } for idx, row in repo_cg.method_df.iterrows()
    }
    return all_methods


def parse_git_diff_enhanced(change_text):
    """Utility to parse git diff and extract all components.

    Args:
        change_text (str): Change patch string in diff patch.

    Returns:
        parsed_diff: A dictionary of all items in the diff. It has the following
            keys 'before_index' 'after_index', 'mode', 'filename', and 'changes'. 
    
        Here, 'changes' is a list of dictionaries representing each change
        in a file, with the following keys: 'text' (hunk segment with context),
        'before_start_line' (line in source file before revision), 'before_count'
        (number of lines in hunk before revision), 'after_start_line' (line in
        source file after revision), 'after_count' (number of lines in hunk
        after revision), and 'diff_groups' (a list of fine-grained changes in
        the hunk). Each fine-grained change is a dictionary with the following
    """
    # Regex patterns to capture different components of the diff
    file_pattern = r"diff --git a/(.+) b/\1"
    index_pattern = r"index (\w+)\.\.(\w+) (\d+)"
    header_pattern = r"@@ -(\d+),(\d+) \+(\d+),(\d+) @@\s+(.*?)(?=(^@@ -)|\Z)"

    # Split the diff text into file sections
    file_sections = re.split(file_pattern, change_text)[1:]

    parsed_diff = {}
    for i in range(0, len(file_sections), 2):
        file_name = file_sections[i].strip()
        if not file_name.strip().endswith('.java'):
            continue

        file_content = file_sections[i+1]
        # Extracting changes
        changes = re.findall(header_pattern, file_content, re.DOTALL | re.MULTILINE)
        changed_line_numbers = [int(change[0]) for change in changes]
        parsed_diff[file_name] = changed_line_numbers

    return parsed_diff


def get_containing_methods(
    line_number,
    file,
    repo,
    path_to_assets,
    all_methods,
    all_methods_current,
):
    """
    """
    methods_to_ids, _, paths_to_ids = map_methods_ids_and_paths(
        repo, all_methods, path_to_assets
    )
    ids_to_methods = {v: k for k, v in methods_to_ids.items()}

    methods_to_ids_current, _, paths_to_ids_current = map_methods_ids_and_paths(
        repo, all_methods_current, path_to_assets
    )

    new_methods_to_ids_current = {}
    for k, v in methods_to_ids_current.items():
        _k = k.split('<sep>')[0] + '<sep>' + k.split('<sep>')[1]
        if _k in new_methods_to_ids_current:
            new_methods_to_ids_current[_k].append(v)
        else:
            new_methods_to_ids_current[_k] = [v]

    if file not in paths_to_ids or \
       file not in paths_to_ids_current:
        return []

    containing_methods = []
    for method_idx in paths_to_ids[file]:
        path, name, start, end = ids_to_methods[method_idx].split('<sep>')
        if int(start) <= line_number <= int(end):
            containing_methods.append(f"{path}<sep>{name}")

    containing_methods_current = [
        new_methods_to_ids_current[item]
        for item in containing_methods
        if item in new_methods_to_ids_current
    ]
    containing_methods_current = [
        item
        for sublist in containing_methods_current
        for item in sublist
    ]

    return containing_methods_current


def get_impacted_files_from_history(seed_file, history, top_N=None):
    """
    """
    relevant_history = [
        commit_files for commit_files in history.values()
        if seed_file in commit_files
    ]
    if top_N:
        relevant_history = relevant_history[:top_N]

    impacted_files = set([
        item
        for sublist in relevant_history
        for item in sublist
    ])
    return impacted_files


def get_impacted_methods_from_history(
    seed_file,
    seed_method,
    repo,
    path_to_assets,
    all_methods_current,
    history,
    top_N=None,
):
    """
    """
    relevant_history = [
        commit_hash
        for commit_hash, files in history.items()
        if seed_file in files
    ]

    if top_N:
        relevant_history = relevant_history[:top_N]

    path_to_repo = path_to_assets / f'projects/{repo}'

    impacted_methods = []
    for commit_idx in range(len(relevant_history) - 1):
        commit = relevant_history[commit_idx]
        parent_commit = relevant_history[commit_idx+1]
        commit_id = f"{commit}<sep>{parent_commit}"
        diff = get_patch_from_commit_hash(
            path_to_repo,
            commit,
        )
        change_text, _ = extract_change_and_test_patches(diff)

        all_methods = parse_repository_snapshot(
            Path(path_to_assets) / 'projects' / repo,
            parent_commit,
        )

        parsed_diff = parse_git_diff_enhanced(change_text)
        containing_methods_in_files = [
            get_containing_methods(
                line_number,
                file,
                repo,
                path_to_assets,
                all_methods,
                all_methods_current,
            )
            for file, line_numbers in parsed_diff.items()
            for line_number in line_numbers
            # if file in paths_to_ids and file.endswith('java')
        ]

        impacted_methods_in_commit = list(set([
            item
            for sublist in containing_methods_in_files
            for item in sublist
        ]))

        if seed_method in impacted_methods_in_commit:
            impacted_methods += impacted_methods_in_commit
    
    return set(impacted_methods)
