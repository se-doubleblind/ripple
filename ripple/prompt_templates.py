from string import Template


SYSTEM_PROMPT = """
You will be acting as an AI assistant for coding with strong software engineering experience.

You will be given one or more tasks. When performing the task, be concise and also be very careful about the format.
"""

PLAN_PROMPT = """Given an issue description and the initial location that needs to be modified, create a technical summary of the changes needed to fix the issue. Focus on:
- What modifications need to be made
- Where similar changes might be needed (e.g., other parts of the same module, related functions, or code paths affected by the issue)
- Any dependent updates potentially required
- Do not bother about any updates to tests

## Input Format
- Here is a summary of the issue/ticket:
{issue_summary}

- Detailed description of the issue/ticket:
{issue_description}

- Source code at the initial or seed editing location in {file_name}
{seed_code}

## Output Format:
<change_plan>
[Provide a high-level summary of what changes may need to potentially be made to fix the issue, discussing key impact points and potential ripple effects.]
</change_plan>

If there's insufficient information to make a definitive determination, make any assumptions as required while clearly stating such assumptions. Respond only with the change_plan tags as described above. Do not include any other text in your response.
"""


IA_PROMPT = """For the given issue/ticket, based on the following change plan (a first draft), determine which of the methods provided in the repository structure will be impacted, i.e., will need changes to be made in.

## Input Format
- Here is a summary of the issue/ticket:
{issue_summary}

- A first draft for the change plan:
{change_plan}

- Repository structure with method summaries:
{repo_structure}

**Task:**
- Identify impacted methods from the repository structure based on their role and dependencies.
- Consider how class structure affects impact propogation.
- Justify the impact decision for each method. If impact is unclear, state assumptions.

## Output Format
<impacted_methods>
class_name1,method_name1
class_name1,method_name2
class_name2,method_name3
...
</impacted_methods>

Your assessment should be based solely on the provided Java repository structure. Note that you may need to make educated guesses about method interactions, and the focus is on precise identification.
"""


ABLATION_PROMPT = """For the given issue/ticket, determine which of the methods provided in the repository structure will be impacted, i.e., will need changes to be made in.

## Input Format
- Here is a summary of the issue/ticket:
{issue_summary}

- Repository structure with method summaries:
{repo_structure}

**Task:**
- Identify impacted methods from the repository structure based on their role and dependencies.
- Consider how class structure affects impact propogation.
- Justify the impact decision for each method. If impact is unclear, state assumptions.

## Output Format
<impacted_methods>
class_name1,method_name1
class_name1,method_name2
class_name2,method_name3
...
</impacted_methods>

Your assessment should be based solely on the provided Java repository structure. Note that you may need to make educated guesses about method interactions, and the focus is on precise identification.
"""
