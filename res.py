import os
import random
import logging
import argparse

from src.results.get_results import get_results
from src.results.plots import (
    plot_sequences_lengths,
    plot_average_log_likelihood,
    plot_bleu_evaluation_metrics,
    plot_mauve_evaluation_metrics,
    generate_latex_table,
)


def save_results(top_k_df, top_p_df, model_name, results_dir):
    top_k_df.to_csv(os.path.join(results_dir, f"top_k_{model_name}.csv"), sep="\t")
    top_p_df.to_csv(os.path.join(results_dir, f"top_p_{model_name}.csv"), sep="\t")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate results for the experiments.")

    parser.add_argument(
        "--model_names",
        nargs="+",
        default=[
            "pythia-70m",
            "pythia-410m",
            "pythia-1.4b",
            "pythia-2.8b",
        ],
        choices=[
            "gpt2",
            "gpt2-medium",
            "gpt2-large",
            "gpt2-xl",
            "pythia-70m",
            "pythia-160m",
            "pythia-410m",
            "pythia-1b",
            "pythia-1.4b",
            "pythia-2.8b",
            "pythia-6.9b",
            "pythia-12b",
        ],
        help="Models to use for generating results.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory to save the results files.",
    )

    args = parser.parse_args()

    return args


def main():
    logging.basicConfig(level=logging.INFO)

    args = parse_args()
    model_names = args.model_names
    results_dir = args.results_dir

    # Set seed for random number generators
    random.seed(args.seed)

    # Create a directory to save the output files
    os.makedirs(args.results_dir, exist_ok=True)

    results = {}

    for model_name in args.model_names:
        top_k_df, top_p_df = get_results(model_name)

        save_results(top_k_df, top_p_df, model_name, args.results_dir)

        results[model_name] = {"top_k": top_k_df, "top_p": top_p_df}

    # Plot the results
    logging.info("Plotting results...")
    plot_sequences_lengths(results, results_dir)
    plot_average_log_likelihood(results, results_dir, "global")
    plot_average_log_likelihood(results, results_dir, "local")
    plot_mauve_evaluation_metrics(results, model_names, results_dir)
    plot_bleu_evaluation_metrics(results, model_names, results_dir)
    generate_latex_table(results, model_names, results_dir, table_type="mauve")
    generate_latex_table(results, model_names, results_dir, table_type="bleu")

    # TODO: plot decoding constants
    # TODO: plot the graph mcmc number of samples


if __name__ == "__main__":
    main()
