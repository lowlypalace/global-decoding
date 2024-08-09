# Global vs Local Decoding for Large Language Models

This repository provides a Python script for generating text sequences using various large language models (LLMs), conducting Markov Chain Monte Carlo (MCMC) analysis, and evaluating the generated sequences. The script explores different decoding strategies, focusing on local versus global normalization techniques to understand their effects on the quality and diversity of the generated text.

## Requirements

- Python 3.8.6 or higher

## Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/lowlypalace/global-decoding.git
cd global-decoding
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Usage

The script can be run from the command line with various arguments to customize the text generation, MCMC analysis, and evaluation process.

### Basic Command
```bash
python main.py \
  --top_k 100 \
  --sequence_count 1000 \
  --batch_size_seq 32 \
  --batch_size_prob 16 \
  --model_name gpt2-medium \
  --mcmc_num_samples 100 \
  --eval_num_sequences 100 \
  --seed 0
```

By default, all of the actions are performed in the following order: `generate_seqs`, `compute_probs`, `run_mcmc`, `run_eval_mauve`,`run_eval_bleu`. The actions to run can be specified using the `--actions` argument. The actions depend on each other, so the order of the actions should be preserved. Specifically, `run_eval_bleu` does not depend on  `run_eval_mauve`, but both require `run_mcmc` to have been completed first.

 The available actions can be seen by running `python main.py --help`.

### Resume Computation
If the sequences were generated, but some of the subsequent steps (e.g. BLEU evaluation) failed or timed out, the task could be resumed as follows:
```bash
python main.py \
--preload_dir 562fb1 \
--model_name pythia-1.4b \
--actions run_eval_bleu
```
The other arguments will be fetched from the `metadata.json` file in the `--preload_dir` directory.

### Arguments
Run `python main.py --help` to see all available arguments and their descriptions.

## Testing
To run the tests, use the following command:

```bash
  python -m unittest
```
## Output

Outputs are saved in the specified `--output_dir` directory, including generated sequences, logs, and evaluation results. Example of the output directory structure:
```
pythia-1.4b
└── 0bb090
│   └── eval
│   │   ├── mauve_results_global.json
│   │   ├── mauve_results_local.json
│   │   ├── self_bleu_results.json
│   ├──  mcmc
│   │   └── plots
│  │   ├── sampled_sequences_decoded.json
│   │   ├── sampled_sequences_ids.json
│   │   ├── sampled_target_logprobs.json
│   ├── sequences
│   │   └── plots
│   │   ├── logprobs_proposal_tokens.json
│   │   ├── logprobs_proposal.json
│   │   ├── logprobs_target_tokens.json
│   │   ├── logprobs_target.json
│   │   ├── sequences_decoded.json
│   │   ├── sequences_ids.json
│  ├── log.txt
│   ├── metadata.json
├── 1c0331
└── 1dbe15
```

## Linting

To lint the code, use the following command:

```bash
black .
```

## Contributing

Contributions to this project are welcome! Please fork the repository, make your changes, and submit a pull request.
