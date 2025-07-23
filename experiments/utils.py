import random
import re
import statistics
from copy import deepcopy


def extract_tag_list(tag, text, remove_leading_newline=False):
    """Extract a list of tags from a given XML string.

    Args:
        tag: The XML tag to extract.
        text: The XML string.
        remove_leading_newline: Whether to remove the leading newline from the
            extracted values.

    Returns:
        A list of values extracted from the provided tag.
    """
    # Define a regular expression pattern to match the tag
    pattern = rf"<{tag}(?:\s+[^>]*)?>(.*?)</{tag}>"

    # Use re.findall to extract all occurrences of the tag
    values = re.findall(pattern, text, re.DOTALL)

    if len(values) == 0:
        pattern = rf"<{tag}(?:\s+[^>]*)?>(.*)"
        values = re.findall(pattern, text, re.DOTALL)

    if remove_leading_newline:
        values = [v[1:] if v[0] == "\n" else v for v in values]
    return values


def extract_tag(tag, text, remove_leading_newline=False):
    """Extract a tag from a given XML string.

    Args:
        tag: The XML tag to extract.
        text: The XML string.
        remove_leading_newline: Whether to remove the leading newline from the
            extracted values.

    Returns:
        An extracted string.
    """
    values = extract_tag_list(tag, text, remove_leading_newline)
    if len(values) > 0:
        return values[0]
    else:
        return ""


def process_response_with_impact_set(candidate_response):
    """
    """
    response = candidate_response.strip()
    impacted_methods = [
        '.'.join(line.split(','))
        for line in extract_tag('impacted_methods', response).strip().split('\n')
    ]
    return impacted_methods


def process_response_with_impact_set_gemini(candidate_response):
    """
    """
    response = candidate_response.strip()
    impacted_methods = [
        '.'.join(line.split('.'))
        for line in extract_tag('impacted_methods', response).strip().split('\n')
    ]
    return impacted_methods


def get_call_dependencies_from_method_id(seed_mid, methods, max_depth):
    """
    """
    _dependencies = {
        0: set([
            (seed_mid, str(item))
            for item in methods[seed_mid]['call_edges']
        ])
    }
    for depth in range(1, max_depth+1):
        prev_methods = _dependencies[depth-1]
        curr_edges = set()
        for _, to_mid in prev_methods:
            curr_edges = curr_edges.union(
                set([
                    (to_mid, str(item))
                    for item in methods[to_mid]['call_edges']
                ])
            )
        _dependencies[depth] = deepcopy(curr_edges)

    dependencies = set.union(*[item for item in _dependencies.values()])
    return dependencies


def get_in_class_dependencies(methods, paths_to_ids, ids_to_paths):
    dependencies = set.union(*[
        set([(mid, to_mid) for to_mid in paths_to_ids[ids_to_paths[mid]]])
        for mid in methods
    ])
    return dependencies
        

def get_impacted_methods_from_dependencies(dependence_edges):
    """
    """
    callers = set([item[0] for item in dependence_edges])
    callees = set([item[1] for item in dependence_edges])
    impacted_methods = callers.union(callees)
    return impacted_methods


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

    hit_custom = compute_hit_custom_over_n_runs(predicted, true, n, seed)

    micro_metrics = compute_micro_metrics(predicted, true)

    return {
        'precision': statistics.mean(precision),
        'recall': statistics.mean(recall),
        'micro-precision': micro_metrics['precision'],
        'micro-recall': micro_metrics['recall'],
        'f1-score': statistics.mean(f1),
        'hit@custom': hit_custom,
    }


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


def remove_prefix_in_path(path, repo, path_to_assets):
    prefix = f"{path_to_assets}/projects/{repo}/"
    index = path.find(prefix)
    trimmed_path = path[index + len(prefix):]
    return trimmed_path
