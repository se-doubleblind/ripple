import argparse
import ast
import json
import logging
import statistics
from copy import deepcopy
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from utils import (
    get_call_dependencies_from_method_id,
    get_in_class_dependencies,
    get_impacted_methods_from_dependencies,
    compute_metrics,
    map_methods_ids_and_paths
)


PATH_TO_ASSETS = "../assets"


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_instance_metadata(commit_id, repo, all_instances):
    for instance in all_instances[repo]:
        if f"{instance['commit']}<sep>{instance['parent-commit']}" == commit_id:
            break
    return instance


def compute_top_k_from_ranks(ranks, k):
    return sum([1 for item in ranks if item <= k])


def compute_hit_k(ranks, k):
    if min(ranks) <= k:
        return 1
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run IntelliSMT with OpenAI GPT-X")

    ## Pipeline arguments
    parser.add_argument("--path_to_assets", type=str, default=PATH_TO_ASSETS,
                        help="Path to all data assets.")
    args = parser.parse_args()

    with open(str(Path(args.path_to_assets) / 'all-methods.json'), 'r') as f:
        all_methods = json.load(f)

    # LLM response sampling strategy
    sampling_fn = set.intersection

    with open(str(Path(PATH_TO_ASSETS) / 'evolutionary.100.json'), 'r') as f:
        evolutionary_outputs = json.load(f)

    df = pd.read_csv(str(Path(PATH_TO_ASSETS) / 'athena_whole.csv'))

    for outputs_cache, entities_cache in zip(
        ['ranking-based', 'llm-based', 'entity-matching'],
        ['entities-cosine.json', 'entities-llm.json', 'entities-matched.json']
    ):
        stage = ' '.join(outputs_cache.split('-')).upper()
        with open(str(Path(args.path_to_assets) / 'cia-dataset.json'), 'r') as f:
            all_instances = json.load(f)

        matched_entities = None
        if entities_cache:
            with open(str(Path(args.path_to_assets) / entities_cache), 'r') as f:
                matched_entities = json.load(f)

        outputs_path = Path(args.path_to_assets) / f'all-outputs/sensitivity/{outputs_cache}/gpt-4o'
        llm_output_paths = sorted(list(Path(outputs_path).iterdir()))

        predicted, true = [], []
        athena_precision, athena_recall, athena_f1, athena_hitk = [], [], [], []
        for path in tqdm(llm_output_paths):
            with open(path, 'r') as f:
                instance_outputs = json.load(f)

            repo = instance_outputs['repo']
            commit_id = instance_outputs['commit-id']
            commit, parent_commit = commit_id.split('<sep>')
            instance_metadata = get_instance_metadata(commit_id, repo, all_instances)
            method_path = f"{instance_metadata['path']}/{instance_metadata['name']}"
            iid = instance_metadata['id']

            instance_true = set(instance_metadata['impact-set-methods'])
            if matched_entities:
                instance_true.add(instance_metadata['focal-method-id'])

                if 'cosine' in entities_cache:
                    new_seed = matched_entities[iid]
                elif 'matched' in entities_cache:
                    new_seed = matched_entities[iid][0]
                else:
                    new_seed = matched_entities[iid][1]

                instance_metadata['focal-method-id'] = new_seed
                instance_true.discard(new_seed)

            # Baseline 1: Evolutionary coupling
            impacted_methods_from_evolutionary = evolutionary_outputs[iid]['method']

            # Baseline 2: Call + Class-Member dependencies
            methods = all_methods[repo][commit_id]
            methods_to_ids, ids_to_paths, paths_to_ids = map_methods_ids_and_paths(
                repo, methods, args.path_to_assets
            )

            call_dependencies = get_call_dependencies_from_method_id(
                instance_metadata['focal-method-id'], methods, 1
            )
            if call_dependencies:
                impacted_methods_from_call_dependencies = \
                    get_impacted_methods_from_dependencies(call_dependencies)
            else:
                impacted_methods_from_call_dependencies = {instance_metadata['focal-method-id']}

            in_class_dependencies = get_in_class_dependencies(
                impacted_methods_from_call_dependencies,
                paths_to_ids,
                ids_to_paths
            )
            dependencies = call_dependencies.union(in_class_dependencies)
            impacted_methods_from_dependencies = get_impacted_methods_from_dependencies(
                dependencies
            )

            # RIPPLE
            impacted_methods_from_reasoning = set()
            for component_outputs in instance_outputs['impact-analysis']:
                candidate_outputs = [
                    set(candidate['impact-set-predicted'])
                    for candidate in component_outputs['components'].values()
                    if candidate['impact-set-predicted']
                ]
                if candidate_outputs:
                    candidate_outputs = sampling_fn(*candidate_outputs)
                else:
                    candidate_outputs = set()

                candidate_outputs.discard(instance_metadata['focal-method-id'])
                impacted_methods_from_reasoning = impacted_methods_from_reasoning.union(candidate_outputs)

            predicted.append(deepcopy(impacted_methods_from_reasoning))
            true.append(deepcopy(instance_true))

            # Athena baseline
            row = df[
                (df['repo'] == repo) & \
                (df['parent commit'] == parent_commit) & \
                (df['method path'] == method_path)
            ]
            binstance_gold_size = row['ground truth size'].item()
            binstance_ranks = ast.literal_eval(row['sort_ranks'].item())
            _k = len(impacted_methods_from_reasoning)
            binstance_tp = compute_top_k_from_ranks(binstance_ranks, _k)
            binstance_precision = binstance_tp / _k if _k else 0.0
            binstance_hit_k = compute_hit_k(binstance_ranks, _k)

            binstance_recall = binstance_tp / binstance_gold_size
            binstance_f1 = 2 * binstance_precision * binstance_recall / (binstance_precision + binstance_recall) \
                           if (binstance_precision + binstance_recall != 0) \
                           else 0
            
            athena_precision.append(binstance_precision)
            athena_recall.append(binstance_recall)
            athena_f1.append(binstance_f1)
            athena_hitk.append(binstance_hit_k)

        print(f'Results for method-level impact analysis, ATHENA ({stage.upper()}):')
        print(f'  Hit @ custom: {sum(athena_hitk) / len(athena_hitk)}')
        print(f'  Precision @ custom: {statistics.mean(athena_precision)}')
        print(f'  Recall @ custom: {statistics.mean(athena_recall)}')
        print(f'  F1-Score @ custom: {statistics.mean(athena_f1)}')
        print()

        metrics = compute_metrics(predicted, true)
        print(f'Results for method-level impact analysis, ({stage.upper()}):')
        print(f"  Hit @ k over 100 runs: {metrics['hit@custom']}")
        print(f"  Mean Precision: {metrics['precision']}")
        print(f"  Mean Recall: {metrics['recall']}")
        print(f"  Mean F1-Score: {metrics['f1-score']}")
        print()
        print('-'*20)

