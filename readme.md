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
python src/main.py \
  --top_k 100 \
  --sequence_count 1000 \
  --batch_size_seq 32 \
  --batch_size_prob 16 \
  --model_name gpt2-medium \
  --mcmc_rate 10 \
  --mcmc_burnin 0.2 \
  --eval_num_sequences 100 \
  --seed 0
```

### Arguments
Run `python src/main.py --help` to see all available arguments and their descriptions.

## Testing
To run the tests, use the following command:

```bash
  python -m unittest
```
## Output

Outputs are saved in the specified `--output_dir` directory, including generated sequences, logs, and evaluation results.

## Contributing

Contributions to this project are welcome! Please fork the repository, make your changes, and submit a pull request.
