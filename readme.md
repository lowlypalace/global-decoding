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
  --model_name pythia-1.4b \
  --mcmc_num_samples 100 \
  --eval_num_sequences 100 \
  --seed 0
```

### Resume Computation
If the sequences were generated, but some of the subsequent steps (e.g. BLEU evaluation) failed or timed out, the task could be resumed as follows:
```bash
python main.py \
--preload_dir 562fb1 \
--model_name pythia-1.4b \
```
The other arguments will be fetched from the `metadata.json` file.

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
│   │   ├── sampled_sequences_decoded.json
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
│   ├── log.txt
│   ├── metadata.json
├── 1c0331
└── 1dbe15
```

## Visualizing and Exporting Results

The `res.py` script is responsible for generating results for the experiments. It processes output from different models and saves them as DataFrames. The results are then saved as CSV files.

### Functions

- `save_results(top_k_df, top_p_df, model_name, results_dir)`: Saves the DataFrames to CSV files.
- `parse_args()`: Parses command-line arguments for model names, seed, and results directory.
- `main()`: Main function that orchestrates the result generation, saving, and plotting.

### DataFrame Columns

The results are stored in two DataFrames: `top_k_df` and `top_p_df`. Below are the columns and their descriptions:

- `sub_dir`: Subdirectory name where the results are stored.
- `top_k`: The top-k value used in generation.
- `top_p`: The top-p value used in generation.
- `mauve_local`: Local MAUVE evaluation score.
- `mauve_global`: Global MAUVE evaluation score.
- `bleu_local`: Local BLEU evaluation score.
- `bleu_global`: Global BLEU evaluation score.
- `log_likelihood_local`: Local log likelihood of sequences.
- `log_likelihood_global`: Global log likelihood of sequences.
- `avg_length_local`: Average length of sequences without padding.
- `avg_length_global`: Average length of MCMC sampled sequences without padding.
- `sequence_local`: Example of a locally decoded sequence.
- `sequence_global`: Example of a globally decoded sequence.
- `constants_products`: Decoding constants used in normalization.

The output CSV files are saved in the specified `--results_dir` directory. The directory contains:

- CSV Files: Raw data with MAUVE, BLEU, log likelihoods, and average sequence lengths for various models and decoding strategies.

- HTML and PDF Files: Visualizations of evaluation metrics:
  - `average_lengths`: Average sequence lengths for different models and strategies.
  - `average_log_likelihood`: Average log likelihoods of generated sequences.
  - `bleu_top_k` and `bleu_top_p`: BLEU scores for top-k and top-p strategies.
  - `mauve_top_k` and `mauve_top_p`: MAUVE scores indicating distribution divergence.

These outputs help compare the performance and characteristics of different decoding strategies.

## Linting

To lint the code, use the following command:

```bash
black .
```

## Contributing

Contributions to this project are welcome! Please fork the repository, make your changes, and submit a pull request.
