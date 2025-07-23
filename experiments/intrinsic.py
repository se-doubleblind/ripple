import argparse
import json
import logging
from copy import deepcopy
from pathlib import Path

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run IntelliSMT with OpenAI GPT-X")

    ## Pipeline arguments
    parser.add_argument("--path_to_assets", type=str, default=PATH_TO_ASSETS,
                        help="Path to all data assets.")
    parser.add_argument("--llm", type=str, default='gpt', choices=['gpt', 'claude', 'gemini'],
                        help=("LLM name."))
    parser.add_argument('--max_depth', type=int, default=1,
                        help=('Max number of dependence hops'))
    parser.add_argument('--num_commits', type=int, default=100,
                        help=('Number of commits in evolutionary coupling'))

    args = parser.parse_args()

    with open(str(Path(args.path_to_assets) / 'cia-dataset.json'), 'r') as f:
        all_instances = json.load(f)

    with open(str(Path(args.path_to_assets) / 'all-methods.json'), 'r') as f:
        all_methods = json.load(f)

    # LLM response sampling strategy
    sampling_fn = set.intersection

    evolutionary_filename = f'evolutionary.{args.num_commits}.json'
    with open(str(Path(PATH_TO_ASSETS) / evolutionary_filename), 'r') as f:
        evolutionary_outputs = json.load(f)

    all_predicted = {
        key: [] for key in [
            'evolutionary',
            'dependence-based',
            f'ripple-{args.llm}',
        ]
    }
    true = []

    outputs_path = Path(args.path_to_assets) / f'all-outputs/{args.llm}'
    llm_output_paths = sorted(list(Path(outputs_path).iterdir()))

    no_predictions = 0
    for path in tqdm(llm_output_paths):
        with open(path, 'r') as f:
            instance_outputs = json.load(f)

        repo = instance_outputs['repo']
        commit_id = instance_outputs['commit-id']
        commit, parent_commit = commit_id.split('<sep>')
        instance_metadata = get_instance_metadata(commit_id, repo, all_instances)
        iid = instance_metadata['id']

        # Baseline 1: Evolutionary coupling
        impacted_methods_from_evolutionary = evolutionary_outputs[iid]['method']
        all_predicted['evolutionary'].append(impacted_methods_from_evolutionary)

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
        all_predicted['dependence-based'].append(impacted_methods_from_dependencies)


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

        if not impacted_methods_from_reasoning: no_predictions += 1

        all_predicted[f'ripple-{args.llm}'].append(deepcopy(impacted_methods_from_reasoning))

        true.append(set(instance_metadata['impact-set-methods']))

    for stage, predicted in all_predicted.items():
        metrics = compute_metrics(predicted, true)
        print(f'Results for method-level impact analysis, ({stage.upper()}):')
        print(f"  Hit @ k over 100 runs: {metrics['hit@custom']}")
        print(f"  Mean Precision: {metrics['precision']}")
        print(f"  Mean Recall: {metrics['recall']}")
        print(f"  Mean F1-Score: {metrics['f1-score']}")
        print()
