import numpy as np
import json
import argparse
import logging
import os
import time

from utils import (
    create_filename,
)


from plots import plot_mcmc_distribution, plot_chain
from utils import setup_logging, save_args, get_timestamp, timer


def indicator_top_k(sequence):
    # In our case, we can simply return 1 as we are using top-k sampling
    return 1


def metropolis_hastings(
    sequence_count,
    burnin,
    sequences_ids,
    sequences_decoded,
    target_logprobs,
    proposal_logprobs,
    rate,
    output_dir,
):
    # List to store the generated samples
    sampled_sequences = []
    sampled_decoded_sequences = []
    sampled_target_logprobs = []

    # Calculate the number of burn-in samples
    burnin_index = int(burnin * sequence_count)

    # Get the first sequence and its probabilities
    current_sequence = sequences_ids[0]
    current_decoded_seq = sequences_decoded[0]
    logprob_target_current, logprob_proposal_current = (
        target_logprobs[0],
        proposal_logprobs[0],
    )

    # This is a top-level loop to generate multiple sequences
    for i in range(1, sequence_count):
        # Get the sequence to propose
        proposed_sequence = sequences_ids[i]
        # Get the probabilities for the proposed sequences
        logprob_target_proposed, logprob_proposal_proposed = (
            # target_logprobs[i].item(),
            # proposal_logprobs[i].item(),
            target_logprobs[i],
            proposal_logprobs[i],
        )

        # Calculate the acceptance ratio
        numerator = (
            logprob_target_proposed
            + indicator_top_k(proposed_sequence)
            + logprob_proposal_current
        )
        denominator = (
            logprob_target_current
            + indicator_top_k(current_sequence)
            + logprob_proposal_proposed
        )
        log_acceptance_ratio = numerator - denominator

        # Accept or reject the new sequence based on the acceptance ratio
        if np.log(np.random.uniform(0, 1)) < log_acceptance_ratio:
            current_sequence = proposed_sequence
            logprob_target_current = logprob_target_proposed
            logprob_proposal_current = logprob_proposal_proposed

        # After the burn-in period, add the current state to the list of samples at the specified rate
        if i >= burnin_index and i % rate == 0:
            sampled_sequences.append(current_sequence)
            # Append the decoded sequence and its probabilities to samples
            sampled_decoded_sequences.append(current_decoded_seq)
            sampled_target_logprobs.append(logprob_target_current)

    with open(create_filename("sampled_sequences", "json", output_dir), "w") as f:
        json.dump(sampled_sequences, f)
    with open(
        create_filename("sampled_decoded_sequences", "json", output_dir), "w"
    ) as f:
        json.dump(sampled_decoded_sequences, f)
    with open(create_filename("sampled_target_logprobs", "json", output_dir), "w") as f:
        json.dump(sampled_target_logprobs, f)

    return sampled_sequences, sampled_decoded_sequences, sampled_target_logprobs


# Define the function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Perform MCMC analysis.")

    parser.add_argument(
        "--burnin",
        type=float,
        default=0.2,
        help="Burn-in period as a fraction of the total number of samples.",
    )
    parser.add_argument(
        "--rate",
        type=int,
        default=1,
        help="Rate at which to sample sequences after the burn-in period.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=os.path.join("output", "sequences"),
        help="Directory to load the input files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join("output", "mcmc"),
        help="Directory to save the output files.",
    )
    parser.add_argument(
        "--dirs",
        nargs="+",
        help="Directories inside the input folder to load the sequences from.",
    )

    args = parser.parse_args()
    return args


def load_sequences(input_dir, directories):
    sequences = []
    sequences_decoded = []
    target_logprobs = []
    proposal_logprobs = []

    for directory in directories:
        sequences_filename = os.path.join(input_dir, directory, "sequences_ids.json")
        sequences_decoded_filename = os.path.join(
            input_dir, directory, "sequences_decoded.json"
        )
        target_logprobs_filename = os.path.join(
            input_dir, directory, "logprobs_target.json"
        )
        proposal_logprobs_filename = os.path.join(
            input_dir, directory, "logprobs_proposal.json"
        )

        with open(sequences_filename, "r") as f:
            sequences.extend(json.load(f))
        with open(sequences_decoded_filename, "r") as f:
            sequences_decoded.extend(json.load(f))
        with open(target_logprobs_filename, "r") as f:
            target_logprobs.extend(json.load(f))
        with open(proposal_logprobs_filename, "r") as f:
            proposal_logprobs.extend(json.load(f))

    return sequences, sequences_decoded, target_logprobs, proposal_logprobs


def main():
    # Parse command-line arguments
    args = parse_args()
    burnin = args.burnin
    rate = args.rate
    seed = args.seed
    input_dir = args.input_dir
    dirs = args.dirs
    output_dir = args.output_dir

    # Add a directory with a timestamp to the output directory
    output_dir = os.path.join(output_dir, get_timestamp())
    # Create a directory to save the output files
    os.makedirs(output_dir, exist_ok=True)
    # Save log messages to a file
    setup_logging(log_file=os.path.join(output_dir, "log.txt"))
    # Save command-line arguments to JSON
    save_args(args, output_dir)

    # Set random seed for reproducibility numpy
    np.random.seed(seed)

    # Load the sequences and their probabilities
    (
        sequences_ids,
        sequences_decoded,
        target_logprobs,
        proposal_logprobs,
    ) = load_sequences(input_dir, dirs)

    # Get the number of sequences
    sequence_count = len(sequences_ids)
    logging.info(f"Loaded {sequence_count} sequences.")

    # Run the Independent Metropolis-Hastings algorithm
    with timer("Running MCMC algorithm"):
        (
            sampled_sequences,
            sampled_decoded_sequences,
            sampled_logprobs,
        ) = metropolis_hastings(
            sequence_count=sequence_count,
            burnin=burnin,
            sequences_ids=sequences_ids,
            sequences_decoded=sequences_decoded,
            target_logprobs=target_logprobs,
            proposal_logprobs=proposal_logprobs,
            rate=rate,
            output_dir=output_dir,
        )

    # Plot the distribution of the generated probabilities
    logging.info("Plotting the results...")
    plot_mcmc_distribution(
        sampled_logprobs,
        plot_type="histogram",
        show=False,
        output_dir=os.path.join(output_dir, "plots"),
    )
    plot_mcmc_distribution(
        sampled_logprobs,
        plot_type="kde",
        show=False,
        output_dir=os.path.join(output_dir, "plots"),
    )
    # Plot the chain of generated samples
    plot_chain(
        sampled_logprobs,
        burnin=burnin,
        show=False,
        output_dir=os.path.join(output_dir, "plots"),
    )


if __name__ == "__main__":
    main()
