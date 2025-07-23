import argparse
import json
import logging
from pathlib import Path

from tqdm import tqdm

from utils import compute_metrics


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
    parser.add_argument('--max_depth', type=int, default=1,
                        help=('Max number of dependence hops'))
    parser.add_argument("--llm", type=str, default='gpt', choices=['gpt', 'claude', 'gemini'],
                        help=("LLM name."))
    parser.add_argument('--num_commits', type=int, default=100,
                        help=('Number of commits in evolutionary coupling'))
    parser.add_argument("--aggregate", action='store_true',
                        help=("If True, sample and aggregate. Default is sample and marginalize."))

    args = parser.parse_args()

    with open(str(Path(args.path_to_assets) / 'cia-dataset.json'), 'r') as f:
        all_instances = json.load(f)

    with open(str(Path(args.path_to_assets) / 'all-methods.json'), 'r') as f:
        all_methods = json.load(f)

    # LLM response sampling strategy
    sampling_fn = set.union if args.aggregate else set.intersection

    instances_with_predictions = {}
    outputs_path = Path(args.path_to_assets) / f'all-outputs/{args.llm}'
    llm_output_paths = sorted(list(Path(outputs_path).iterdir()))

    for path in tqdm(llm_output_paths):
        with open(path, 'r') as f:
            instance_outputs = json.load(f)

        repo = instance_outputs['repo']
        commit_id = instance_outputs['commit-id']
        instance_metadata = get_instance_metadata(commit_id, repo, all_instances)
        iid = instance_metadata['id']
        if iid not in instances_with_predictions:
            instances_with_predictions[iid] = instance_metadata

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

        instances_with_predictions[iid][f'impact-set-predicted-{args.llm}'] = impacted_methods_from_reasoning


    leq_5_split, greater_5_split = [], []
    for iid, instance in instances_with_predictions.items():
        if len(instance['impact-set-files']) <= 5: leq_5_split.append(instance)
        else: greater_5_split.append(instance)

    for key, _instances in zip(['<=5', '>5', 'full'], [leq_5_split, greater_5_split, leq_5_split+greater_5_split]):
        predicted = [item[f'impact-set-predicted-{args.llm}'] for item in _instances]
        true = [item['impact-set-methods'] for item in _instances]

        metrics = compute_metrics(predicted, true)
        print(f'Results for method-level impact analysis ({args.llm.upper()}, {key}):')
        print(f"  Micro Precision: {metrics['micro-precision']}")
        print(f"  Micro Recall: {metrics['micro-recall']}")
        print(f"  Micro F1: {2*metrics['micro-precision']*metrics['micro-recall']/(metrics['micro-precision']+metrics['micro-recall'])}")
        print(f"  Macro Precision: {metrics['precision']}")
        print(f"  Macro Recall: {metrics['recall']}")
        print(f"  Macro F1: {metrics['f1-score']}")
        print()
