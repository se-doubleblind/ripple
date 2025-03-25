import argparse
import json
import logging
import random
import statistics
import sys
from copy import deepcopy
from pathlib import Path

from tqdm import tqdm

from ripple.evolutionary import (
    get_impacted_files_from_history,
    get_impacted_methods_from_history
)
from ripple.llm import (
    AnthropicModel,
    GeminiModel,
    GPTModel,
    process_response_with_change_plan,
    process_response_with_impact_set,
    process_response_with_impact_set_gemini,
)
from ripple.prompt_templates import (
    SYSTEM_PROMPT, PLAN_PROMPT, IA_PROMPT
)
from ripple.structural import (
    get_call_dependencies_from_method_id,
    get_in_class_dependencies,
    get_impacted_methods_from_dependencies
)
from ripple.utils import (
    compute_metrics,
    extract_impact_set,
    extract_summary_from_docstring,
    get_class_name,
    get_connected_components,
    generate_repo_data,
    generate_repo_structure_in_xml,
    generate_complete_repo_structure_in_xml,
    map_methods_ids_and_paths,
    remove_prefix_in_path
)

sys.path.append(Path.cwd().absolute() / 'ripple')

PATH_TO_ASSETS = "/data/axy190020/research/cia/assets-2"


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def select_dataset_by_modified_files(path_to_assets):
    with open(str(Path(path_to_assets) / 'cia-dataset.json'), 'r') as f:
        all_instances = json.load(f)

    # Stratify dataset based on number of modified files
    less_than_5, greater_than_5 = [], []
    for repo, commits in all_instances.items():
        for instance in commits:
            instance['repo'] = repo
            if len(instance['impact-set-files']) <= 5:
                less_than_5.append(deepcopy(instance))
            else:
                greater_than_5.append(deepcopy(instance))
    if len(greater_than_5) > 50: num_greater = 50
    else: num_greater = len(greater_than_5)

    num_less = 100 - num_greater
    less_than_5 = random.sample(less_than_5, num_less)
    greater_than_5 = random.sample(greater_than_5, num_greater)
    selected = less_than_5 + greater_than_5
    return selected, less_than_5, greater_than_5


if __name__ == '__main__':
    # load_dotenv()

    parser = argparse.ArgumentParser(description="Run IntelliSMT with OpenAI GPT-X")

    ## Pipeline arguments
    parser.add_argument("--path_to_assets", type=str, default=PATH_TO_ASSETS,
                        help="Path to all data assets.")
    parser.add_argument("--llm", type=str, default='gpt-4o-old', choices=['gpt-4o-old', 'claude', 'gemini'],
    # parser.add_argument("--llm", type=str, default='gpt', choices=['gpt', 'claude', 'gemini'],
                        help=("LLM name."))
    parser.add_argument("--exrange", type=int, nargs=2, default=[0, 5],
                        help="Range of examples to process: upper-limit not considered.")
    parser.add_argument('--max_depth', type=int, default=1,
                        help=('Max number of dependence hops'))
    parser.add_argument('--num_commits', type=int, default=100,
                        help=('Number of commits in evolutionary coupling'))
    parser.add_argument("--seed_only", action='store_true',
                        help=("If True, consider only seed method in Stage 1 of pipeline."))
    parser.add_argument("--aggregate", action='store_true',
                        help=("If True, sample and aggregate. Default is sample and marginalize."))

    # LLM arguments
    parser.add_argument('--num_responses', type=int, default=5,
                        help=('Number of responses to generate with Reasoner LLM. Useful for '
                              'experimenting with self-consistency'))
    parser.add_argument('--seed', type=int, default=42, help="Set common system-level random seed.")
    parser.add_argument("--top_p", default=0.7, type=float,
                        help=("Only the most probable tokens with probabilities that add "
                              "up to top_p or higher are considered during decoding. "
                              "The valid range is 0.0 to 1.0. 1.0 is equivalent to disabled "
                              "and is the default. Only applies to sampling mode."))
    parser.add_argument("--temperature", default=0.6, type=float,
                        help=("A value used to warp next-token probabilities in sampling mode. "
                              "Values less than 1.0 sharpen the probability distribution, "
                              "resulting in a less random output. Values greater than 1.0 "
                              "flatten the probability distribution, resulting in a more "
                              "random output. A value of 1.0 has no effect and is the default."
                              "The allowed range is 0.0 to 2.0."))
    parser.add_argument('--max_tokens_to_sample', type=int, default=4096,
                        help='Max tokens to sample in Claude LLM.')

    args = parser.parse_args()

    # Set random seed.
    random.seed(args.seed)

    # Print arguments
    logger.info(f'Run arguments are: {args}')

    with open(str(Path(PATH_TO_ASSETS) / 'all-methods.json'), 'r') as f:
        all_methods = json.load(f)

    with open(str(Path(PATH_TO_ASSETS) / 'commit-history.json'), 'r') as f:
        all_commits = json.load(f)

    with open(str(Path(PATH_TO_ASSETS) / 'generated-summaries.json'), 'r') as f:
        all_summaries = json.load(f)

    # Initialize LLM
    if args.llm.startswith('gpt'):
        planner_llm = GPTModel(
            temperature=args.temperature,
            top_p=args.top_p,
            num_responses=1,
            seed=args.seed,
        )
        reasoner_llm = GPTModel(
            temperature=args.temperature,
            top_p=args.top_p,
            num_responses=args.num_responses,
            seed=args.seed,
        )

    elif args.llm == 'claude':
        planner_llm = AnthropicModel(
            max_tokens_to_sample=args.max_tokens_to_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            num_responses=1,
        )
        reasoner_llm = AnthropicModel(
            max_tokens_to_sample=args.max_tokens_to_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            num_responses=args.num_responses,
        )
    elif args.llm == 'gemini':
        planner_llm = GeminiModel(
            temperature=args.temperature,
            top_p=args.top_p,
            num_responses=1,
            seed=args.seed,
        )
        reasoner_llm = GeminiModel(
            temperature=args.temperature,
            top_p=args.top_p,
            num_responses=args.num_responses,
            seed=args.seed,
        )

    # LLM response sampling strategy
    sampling_fn = set.union if args.aggregate else set.intersection

    try:
        if args.seed_only:
            evolutionary_filename = f'evolutionary.{args.num_commits}.seed.json'
        else:
            evolutionary_filename = f'evolutionary.{args.num_commits}.json'

        with open(str(Path(args.path_to_assets) / evolutionary_filename), 'r') as f:
            evolutionary_outputs = json.load(f)
        evolutionary_cached = True

    except FileNotFoundError:
        evolutionary_cached, evolutionary_outputs = False, {}

    full, lite, complex = select_dataset_by_modified_files(args.path_to_assets)

    outputs_path = Path(args.path_to_assets) / f'outputs2/{args.llm}'
    outputs_path.mkdir(exist_ok=True, parents=True)

    for key, _instances in zip(
        # ['full', 'lite', 'complex'], [full, lite, complex]
        ['full'], [full]
    ):
        instances = _instances

        all_predicted = {
            key: {'file': [], 'method': []}
            for key in [
                'commit_history',
                'commit_history_and_call_dependencies',
                'commit_history_and_dependencies',
                'ripple'
            ]
        }
        true = {'file': [], 'method': []}

        ratios = {}
        for cia_instance in tqdm(instances):
            iid = cia_instance['id']
            repo = cia_instance['repo']
            commit_id = f"{cia_instance['commit']}<sep>{cia_instance['parent-commit']}"
            methods_to_ids, ids_to_paths, paths_to_ids = map_methods_ids_and_paths(
                repo, all_methods[repo][commit_id], args.path_to_assets
            )

            # Stage 1: Get file and method impact sets from evolutionary coupling
            if evolutionary_cached:
                files_from_evolutionary = evolutionary_outputs[iid]['file']
                methods_from_evolutionary = evolutionary_outputs[iid]['method']
            else:
                history = all_commits[commit_id]
                files_from_evolutionary = set(get_impacted_files_from_history(
                    cia_instance['path'],
                    history,
                    args.num_commits,
                ))

                # Remove files in history which are not present in the current repo snapshot.
                files_from_evolutionary = set([
                    item
                    for item in files_from_evolutionary
                    if item in paths_to_ids
                ])

                methods_from_evolutionary = set(get_impacted_methods_from_history(
                    cia_instance['path'],
                    str(cia_instance['focal-method-id']),
                    repo,
                    Path(args.path_to_assets),
                    all_methods[repo][commit_id],
                    history,
                    args.num_commits,
                ))

                evolutionary_outputs[iid] = {
                    'file': deepcopy(list(files_from_evolutionary)),
                    'method': deepcopy(list(methods_from_evolutionary)),
                }

            all_predicted['commit_history']['file'].append(
                deepcopy(files_from_evolutionary)
            )
            all_predicted['commit_history']['method'].append(
                deepcopy(methods_from_evolutionary)
            )

            # Stage 2:  Get list of methods and corresponding files from call dependencies
            # Get dependent methods from seed methods from evolutionary coupling.
            if not methods_from_evolutionary:
                methods_from_evolutionary = {str(cia_instance['focal-method-id'])}

            if args.seed_only:
                call_dependencies = get_call_dependencies_from_method_id(
                    str(cia_instance['focal-method-id']),
                    all_methods[repo][commit_id],
                    max_depth=args.max_depth,
                )
            else:
                call_dependencies = set.union(*[
                    get_call_dependencies_from_method_id(
                        mid,
                        all_methods[repo][commit_id],
                        max_depth=args.max_depth,
                    )
                    for mid in methods_from_evolutionary
                ])

            methods_from_call_dependencies = get_impacted_methods_from_dependencies(
                call_dependencies
            )
            # Merge methods from call dependencies and evolutionary coupling
            methods_from_call_dependencies = methods_from_call_dependencies.union(
                methods_from_evolutionary
            )

            # Get impacted files based on dependencies of seed file.
            files_from_call_dependencies = set([
                str(remove_prefix_in_path(
                    all_methods[repo][commit_id][item]['path'],
                    repo,
                    args.path_to_assets,
                ))
                for item in methods_from_call_dependencies
            ])

            all_predicted['commit_history_and_call_dependencies']['method'].append(
                deepcopy(methods_from_call_dependencies)
            )
            all_predicted['commit_history_and_call_dependencies']['file'].append(
                deepcopy(files_from_call_dependencies)
            )

            # Stage 3: Expand methods with in-class dependencies
            in_class_dependencies = get_in_class_dependencies(
                methods_from_call_dependencies, paths_to_ids, ids_to_paths
            )

            methods_from_in_class_dependencies = get_impacted_methods_from_dependencies(
                in_class_dependencies
            )
            methods_from_dependencies = methods_from_call_dependencies.union(
                methods_from_in_class_dependencies
            )

            all_predicted['commit_history_and_dependencies']['method'].append(
                deepcopy(methods_from_dependencies)
            )

            # Stage 4: Build graph to get connected components
            all_dependencies = call_dependencies.union(in_class_dependencies)
            ratio = (len(all_methods[repo][commit_id]) - len(methods_from_dependencies)) / len(all_methods[repo][commit_id]) 
            if repo not in ratios:
                ratios[repo] = [ratio]
            else:
                ratios[repo].append(ratio)

            connected_components = get_connected_components(all_dependencies)

            if (outputs_path / f'{iid}.json').is_file():
                with open(str(outputs_path / f'{iid}.json'), 'r') as f:
                    instance_outputs = json.load(f)

                methods_from_reasoning = set()
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

                    candidate_outputs.discard(cia_instance['focal-method-id'])
                    methods_from_reasoning = methods_from_reasoning.union(candidate_outputs)

                all_predicted['ripple']['method'].append(deepcopy(methods_from_reasoning))

                true['file'].append(set(cia_instance['impact-set-files']))
                true['method'].append(set(cia_instance['impact-set-methods']))

                continue

            # Stage 5: Reason about change impact on the connected components independently.
            methods_in_instance = all_methods[repo][commit_id]
            instance_summary_keys = [
                f"{Path(m['path']).stem}<sep>{m['source']}"
                for m in methods_in_instance.values()
            ]
            method_summaries = [
                extract_summary_from_docstring(all_summaries[repo][key]['docstring'])
                if all_summaries[repo][key]['docstring-useful'] == "False"
                else all_summaries[repo][key]['generated-summary']
                for key in instance_summary_keys
            ]
            instance_summaries = dict(zip(list(methods_in_instance.keys()), method_summaries))

            # complete_repo_data = generate_repo_data(
            #     methods_in_instance,
            #     instance_summaries,
            #     repo,
            #     args.path_to_assets,
            # )

            plan_prompt = PLAN_PROMPT.format(**{
                'issue_summary': cia_instance['issue-summary'],
                'issue_description': cia_instance['issue-description'],
                'file_name': cia_instance['path'],
                'seed_code': all_methods[repo][commit_id][cia_instance['focal-method-id']]['source'],
                # 'repo_structure': generate_complete_repo_structure_in_xml(complete_repo_data, False)
            })
            plan_response, change_plan, plan_usage = planner_llm.invoke(
                SYSTEM_PROMPT,
                plan_prompt,
                process_response_with_change_plan
            )[0]

            instance_outputs = {
                'repo': repo,
                'commit-id': commit_id,
                'change-plan': {
                    'prompt': plan_prompt, 'response': change_plan, 'usage': plan_usage,
                },
                'impact-analysis': []
            }

            methods_from_reasoning = []
            for method_ids in connected_components:
                connected_methods = [
                    all_methods[repo][commit_id][mid]
                    for mid in method_ids
                ]
                connected_methods_and_ids = dict(zip(method_ids, connected_methods))
                summary_keys = [
                    f"{Path(m['path']).stem}<sep>{m['source']}"
                    for m in connected_methods
                ]
                component_summaries = [
                    extract_summary_from_docstring(all_summaries[repo][key]['docstring'])
                    if all_summaries[repo][key]['docstring-useful'] == "False"
                    else all_summaries[repo][key]['generated-summary']
                    for key in summary_keys
                ]
                component_summaries = dict(zip(method_ids, component_summaries))

                repo_data = generate_repo_data(
                    connected_methods_and_ids,
                    component_summaries,
                    repo,
                    args.path_to_assets,
                )
                # repo_data = generate_repo_data(
                #     method_ids,
                #     all_methods[repo][commit_id],
                #     paths_to_ids,
                #     component_summaries,
                #     repo,
                #     args.path_to_assets,
                # )

                # component_call_dependencies = [
                #     f"{all_methods[repo][commit_id][edge[0]]['name']} calls "
                #     f"{get_class_name(all_methods[repo][commit_id][edge[1]], repo, args.path_to_assets)}"
                #     f".{all_methods[repo][commit_id][edge[1]]}"
                #     for edge in call_dependencies
                #     if edge[0] in method_ids and edge[1] in method_ids
                # ]

                ia_prompt = IA_PROMPT.format(**{
                    'issue_summary': cia_instance['issue-summary'],
                    'change_plan': change_plan,
                    'repo_structure': generate_repo_structure_in_xml(repo_data, True),
                })

                reasoner_responses = reasoner_llm.invoke(
                    SYSTEM_PROMPT, ia_prompt, process_response_with_impact_set
                )

                component_outputs = {
                    f'candidate-{idx}': {
                        'response': reasoner_responses[idx-1][0],
                        'impact-set-predicted': extract_impact_set(
                            reasoner_responses[idx-1][1], connected_methods_and_ids,
                        ),
                        'usage': reasoner_responses[idx-1][2],
                    }
                    for idx in range(1, args.num_responses+1)
                }

                instance_outputs['impact-analysis'].append({
                    'prompt': ia_prompt, 'components': component_outputs
                })

                candidate_outputs = [
                    set(candidate['impact-set-predicted'])
                    for candidate in component_outputs.values()
                    if candidate['impact-set-predicted']
                ]
                if candidate_outputs:
                    candidate_outputs = sampling_fn(*candidate_outputs)
                else:
                    candidate_outputs = set([])

                candidate_outputs.discard(cia_instance['focal-method-id'])
                methods_from_reasoning += list(candidate_outputs)

            all_predicted['ripple']['method'].append(deepcopy(methods_from_reasoning))
            true['file'].append(set(cia_instance['impact-set-files']))
            true['method'].append(set(cia_instance['impact-set-methods']))

            with open(str(outputs_path / f'{iid}.json'), 'w') as f:
                json.dump(instance_outputs, f, indent=2)

        if not evolutionary_cached:
            with open(str(Path(args.path_to_assets) / evolutionary_filename), 'w') as f:
                json.dump(evolutionary_outputs, f, indent=2)

        for stage, predicted in all_predicted.items():
            if stage in [
                'commit_history',
                'commit_history_and_call_dependencies',
                'commit_history_and_dependencies',
                'ripple'
            ]:
                for level in ['file', 'method']:
                    if not predicted[level]: continue
                    metrics = compute_metrics(
                        predicted[level],
                        true[level]
                    )

                    print(f'{key}: Results for {level}-level impact analysis, ({stage.upper()}):')
                    print(f"  Hit @ 5 over 100 runs: {metrics['hit@5']}")
                    print(f"  Hit @ 10 over 100 runs: {metrics['hit@10']}")
                    print(f"  Mean Precision: {metrics['precision']}")
                    print(f"  Mean Recall: {metrics['recall']}")
                    print(f"  Mean F1-Score: {metrics['f1-score']}")
                    print(f"  Micro Precision: {metrics['micro-precision']}")
                    print(f"  Micro Recall: {metrics['micro-recall']}")
                    print()
