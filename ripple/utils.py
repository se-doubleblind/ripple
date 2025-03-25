import os
import random
import re
import statistics
from pathlib import Path

import networkx as nx


def compute_metrics_for_pair(pred, gold):
    """
    Compute precision, recall, and F1-score based on set intersection.
    
    Args:
        pred (list of str): Predicted values.
        gold (list of str): Ground-truth values.
        
    Returns:
        dict: Precision, Recall, and F1-score.
    """
    pred_set, gold_set = set(pred), set(gold)
    intersection = pred_set & gold_set  # Common elements
    precision = len(intersection) / len(pred_set) if pred_set else 0.0
    recall = len(intersection) / len(gold_set) if gold_set else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    return {"precision": precision, "recall": recall, "f1-score": f1}


def compute_hit_k_over_n_runs(predicted, true, k, n, seed):
    random.seed(seed)
    hit_k_over_n_runs = []
    for _ in range(n):
        hit_k = 0
        for instance_predicted, instance_true in zip(predicted, true):
            instance_predicted = sorted(list(instance_predicted))
            instance_true = sorted(list(instance_true))
            if len(instance_predicted) > k:
                instance_sample = random.sample(instance_predicted, k)
            else:
                instance_sample = instance_predicted

            if len(set(instance_sample).intersection(set(instance_true))) > 0:
                hit_k += 1
        hit_k_over_n_runs.append(hit_k / len(predicted))

    return statistics.mean(hit_k_over_n_runs)


def compute_hit_custom_over_n_runs(predicted, true, n, seed):
    random.seed(seed)
    hit_k_over_n_runs = []
    for _ in range(n):
        hit_k = 0
        for instance_predicted, instance_true in zip(predicted, true):
            k = len(instance_predicted)
            instance_predicted = sorted(list(instance_predicted))
            instance_true = sorted(list(instance_true))
            if len(instance_predicted) > k:
                instance_sample = random.sample(instance_predicted, k)
            else:
                instance_sample = instance_predicted

            if len(set(instance_sample).intersection(set(instance_true))) > 0:
                hit_k += 1
        hit_k_over_n_runs.append(hit_k / len(predicted))

    return statistics.mean(hit_k_over_n_runs)


def compute_micro_metrics(predicted, true):
    precision, recall = [], []
    for pred, gold in zip(predicted, true):
        pred_set, gold_set = set(pred), set(gold)
        intersection = pred_set & gold_set  # Common elements
        precision_tuple = (len(intersection), len(pred_set)) if pred_set else (0, 1)
        precision.append(precision_tuple)

        recall_tuple = (len(intersection), len(gold_set)) if gold_set else (0, 1)
        recall.append(recall_tuple)

    return {
        'precision': sum([x[0] for x in precision]) / sum([x[1] for x in precision]),
        'recall': sum([x[0] for x in recall]) / sum([x[1] for x in recall])
    }

def compute_metrics(predicted, true, n=100, seed=42):
    precision, recall, f1 = [], [], []
    for instance_predicted, instance_true in zip(predicted, true):
        instance_metrics = compute_metrics_for_pair(
            instance_predicted,
            instance_true,
        )
        precision.append(instance_metrics['precision'])
        recall.append(instance_metrics['recall'])
        f1.append(instance_metrics['f1-score'])
    

    hit_5 = compute_hit_k_over_n_runs(predicted, true, 5, n, seed)
    hit_10 = compute_hit_k_over_n_runs(predicted, true, 10, n, seed)
    hit_custom = compute_hit_custom_over_n_runs(predicted, true, n, seed)

    micro_metrics = compute_micro_metrics(predicted, true)

    return {
        'precision': statistics.mean(precision),
        'recall': statistics.mean(recall),
        'micro-precision': micro_metrics['precision'],
        'micro-recall': micro_metrics['recall'],
        'f1-score': statistics.mean(f1),
        'hit@5': hit_5,
        'hit@10': hit_10,
        'hit@custom': hit_custom,
    }


def remove_prefix_in_path(path, repo, path_to_assets):
    prefix = f"{path_to_assets}/projects/{repo}/"
    index = path.find(prefix)
    trimmed_path = path[index + len(prefix):]
    return trimmed_path


def map_methods_ids_and_paths(repo, methods, path_to_assets):
    methods_to_ids, ids_to_paths, paths_to_ids = {}, {}, {}
    for idx, method in methods.items():
        corrected_path = str(remove_prefix_in_path(
            method['path'],
            repo,
            path_to_assets,
        ))
        start, end = method['method_start'], method['method_end']
        methods_to_ids[f"{corrected_path}<sep>{method['name']}<sep>{start}<sep>{end}"] = idx
        ids_to_paths[idx] = corrected_path
        if corrected_path not in paths_to_ids:
            paths_to_ids[corrected_path] = [idx]
        else:
            paths_to_ids[corrected_path].append(idx)
    return methods_to_ids, ids_to_paths, paths_to_ids


def get_connected_components(edges):
    G = nx.Graph()
    G.add_edges_from(edges)
    connected_components = list(nx.connected_components(G))
    return connected_components


def generate_repo_data(component_methods, summaries, repo, path_to_assets):
    all_paths = {}
    for mid, item in component_methods.items():
        path = remove_prefix_in_path(item['path'], repo, path_to_assets)
        if path in all_paths: all_paths[path].append((mid, item))
        else: all_paths[path] = [(mid, item)]

    all_paths = dict(sorted(all_paths.items()))

    repo_data = [
        {
            'path': path,
            'methods': [
                {'name': item['name'], 'summary': summaries[mid]}
                for mid, item in method_items
            ]
        }
        for path, method_items in all_paths.items()
    ]

    return repo_data


def generate_complete_repo_structure_in_xml(repo_data, include_summary=False):
    repo_structure = '<repository>\n'

    for package_data in repo_data:
        path = package_data['path']
        methods = package_data['methods']

        package_name = '.'.join(path.split('/')[:-1])

        repo_structure += f'  <package_name="{package_name}">\n'

        class_name = Path(path).stem
        method_names = ', '.join([m['name'] for m in methods])
        repo_structure += f'    <class_name="{class_name}" methods="{method_names}" />\n'
        repo_structure += '  </package>\n'

    repo_structure += '</repository>'
    return repo_structure


def generate_repo_structure_in_xml(repo_data, include_summary=False):
    repo_structure = '<repository>\n'

    for package_data in repo_data:
        path = package_data['path']
        methods = package_data['methods']

        package_name = '.'.join(path.split('/')[:-1])

        repo_structure += f'  <package_name="{package_name}">\n'

        class_name = Path(path).stem
        repo_structure += f'    <class_name="{class_name}">\n'

        for method in methods:
            method_name = method['name']
            method_summary = method['summary']
            repo_structure += f'      <method_name="{method_name}">\n'
            if include_summary:
                repo_structure += f'        {method_summary}\n'
            repo_structure += f'      </method>\n'

        repo_structure += '    </class>\n'
        repo_structure += '  </package>\n'

    repo_structure += '</repository>'
    return repo_structure


def generate_repo_structure_lightweight(repo_data, include_summary=False):
    repo_structure = ""
    for package_data in repo_data:
        path = package_data['path']
        methods = package_data['methods']

        package_name = '.'.join(path.split('/')[:-1])

        repo_structure += f'  package_name="{package_name}"\n'

        class_name = Path(path).stem
        repo_structure += f'    class_name="{class_name}"\n'

        for method in methods:
            method_name = method['name']
            method_summary = method['summary']
            if include_summary:
                repo_structure += f'      method="{method_name}"\t{method_summary}\n'
            else:
                repo_structure += f'      method="{method_name}"\n'


    return repo_structure


def extract_summary_from_docstring(docstring):
   # Remove comment markers /** and */
    docstring = re.sub(r"/\*\*|\*/", "", docstring).strip()
    
    # Remove leading '*' from each line
    docstring = re.sub(r"^\s*\*", "", docstring, flags=re.MULTILINE).strip()

    # Extract first paragraph or sentence before annotations (@param, @return, etc.)
    summary_match = re.search(r"(.+?)(?=\n\s*@|\Z)", docstring, re.DOTALL)
    
    if summary_match: summary = summary_match.group(1).strip()
    else: summary = ""

    # Remove inline HTML tags like <p>, <code>, <b>, etc.
    summary = re.sub(r"<[^>]+>", "", summary).strip()

    summary = ' '.join([line.strip() for line in summary.split('\n')])

    return summary


def get_class_name(method, repo, path_to_assets):
    corrected_path = str(Path(remove_prefix_in_path(
        method['path'],
        repo,
        path_to_assets)).with_suffix('')
    )
    class_name = '.'.join(corrected_path.split('/'))
    return class_name


def extract_impact_set(candidate_impact_set, methods):
    """
    """
    method_name_to_ids = {}
    for idx, method in methods.items():
        key = f"{Path(method['path']).stem}.{method['name']}"
        if key in method_name_to_ids:
            method_name_to_ids[key].append(str(idx))
        else:
            method_name_to_ids[key] = [str(idx)]

    impact_set = [
        method_name_to_ids[item]
        for item in candidate_impact_set
        if item in method_name_to_ids
    ]
    return [item for sublist in impact_set for item in sublist]
