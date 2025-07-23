# Artifact for "From Seed to Scope: Reasoning to Identify Change Impact Sets"

`ripple` is a reasoning-based, intent-aware change impact analysis tool that allows users to prioritize higher precision or recall as per preferences.

## Getting Started
This section describes the preqrequisites, and contains instructions, to get the project up and running.

### Setup
#### Hardware Requirements
`ripple` requires access to OpenAI/Google/Anthropic API credentials to use GPT-4o/Gemini-Flash-2.0/Claude-3.5-Sonnet LLMs, respectively. However, we also provide all LLM responses for the dataset (see `outputs.zip`), and this can be skipped for experiments' replication.

#### Project Environment
Currently, `ripple` has been tested and works well on Ububtu OS, and can be set up easily with all the prerequisite packages by following these instructions (if `conda` is already installed, update to the latest version with conda update conda, and skip steps 1 - 3): 
  1. Download the latest, appropriate version of [conda](https://repo.anaconda.com/miniconda/) for your machine (tested with ``conda 23.11.0``).
  2. Install  it by running the `conda_install.sh` file, with the command:
     ```bash
     $ bash conda_install.sh
     ```
  3. Add `conda` to bash profile:
     ```bash
     $ source ~/.bashrc
     ```
  4. Navigate to ``ripple`` (top-level directory) and create a conda virtual environment with the included `environment.yml` file using the following command:     
     ```bash
     $ conda env create -f environment.yml
     ```

     To test successful installation, make sure ``ripple`` appears in the list of conda environments returned with ``conda env list``.
  5. Activate the virtual environment with the following command:     
     ```bash
     $ conda activate ripple
     ```

### Directory Structure

#### 1. Data Artifacts


#### 2. Code
* ``ripple``: package code used in ``pipeline.py``
* ``experiments``: Code for all experiments (RQ1-RQ3)

### Usage Guide
1. Navigate to ``experiments/`` to find the source code for replicating the experiments in RQ1--RQ3 in the paper. This assumes the LLM outputs (e.g., ``gpt4o``, as stored in ``outputs``) are being used.

  * **Option 1.** Run experiments independently:
  
    - Intrinsic evaluation (RQ1, Table 1)
      ```bash
      python intrinsic.py --llm {gpt|claude|gemini}
      ```

    - Stratified evaluation (RQ1, Table 2)
      ```bash
      python intrinsic_stratified.py
      ```

    - Sensitivity to seed edit localization (RQ1, Table 3)
      ```bash
      python intrinsic_sensitivity.py
      ```

    - Qualitative evaluation
      * *Sample-and-marginalize v/s Sample-and-aggregate* (RQ2.2)
      ```bash
      python intrinsic_stratified.py --llm gpt {--aggregate}
      ```  

      * *Granularity* (RQ2.3)
      ```bash
      python granularity.py
      ```

    - Ablation Study evaluation (RQ3)
    ```bash
    python ablation.py
    ```

2. Navigate to top-level directory to build x-LLM model outputs from scratch.

    **Option 2.** Run ``pipeline.py``, which is the main entry-point for the package. It has the following arguments:

    | Argument                | Default                 | Description |
    | :---------------------: | :---------------------: | :---- |
    | ``--path_to_data``      | ``../dataset``          | Path to processed string constraints dataset file  |
    | ``--path_to_outputs``   | ``../all-outputs/gpt``     | Path to cache GPT-x responses |

    ***Note:*** We use the ``seed`` parameter when using the OpenAI chat completion client to ensure reproducible outputs. However, as the OpenAI team notes, sometimes, determinism may be impacted due to necessary changes to model configurations ([[link]](https://platform.openai.com/docs/guides/text-generation/reproducible-outputs)).

    ***Note:*** A prerequisite to running ``pipeline.py`` is that it expects an ``.env`` file at the top-level directory with the following key-value pairs:
    * AZURE_OPENAI_ENDPOINT=<name-of-endpoint>
    * OPENAI_API_KEY=<openai-api-key>
    * OPENAI_API_VERSION=<api-version>
    * AZURE_OPENAI_DEPLOYMENT_NAME=<gpt-model-deployment-name>
    * ANTHROPIC_API_KEY=<anthropic-api-key>
    * GEMINI_API_KEY=<gemini-api-key>

## Contributing Guidelines
There are no specific guidelines for contributing, apart from a few general guidelines we tried to follow, such as:
* Code should follow PEP8 standards as closely as possible
* Code should carry appropriate comments, wherever necessary, and follow the docstring convention in the repository.

If you see something that could be improved, send a pull request! 
We are always happy to look at improvements, to ensure that `ripple`, as a project, is the best version of itself. 

If you think something should be done differently (or is just-plain-broken), please create an issue.

## License
See the [LICENSE](https://github.com/se-doubleblind/ripple/tree/main/LICENSE) file for more details.
