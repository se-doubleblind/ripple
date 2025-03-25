import json
import statistics
from copy import deepcopy
from itertools import combinations
from pathlib import Path

import networkx as nx

from ripple.utils import compute_metrics, remove_prefix_in_path


PATH_TO_ASSETS = "assets"

MAX_DEPTH = 1


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
