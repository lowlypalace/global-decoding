import numpy as np
import json
import argparse
import logging
import torch
import os
import time

from utils import (
    create_filename,
)


from metropolis_hastings import metropolis_hastings
from plots import plot_mcmc_distribution, plot_chain
from utils import setup_logging, save_args, get_timestamp


def indicator_top_k(sequence):
    # In our case, we can simply return 1 as we are using top-k sampling
    return 1


def metropolis_hastings(
    # tokenizer,
    sequence_count,
    burnin,
    sequences,
    target_logprobs,
    proposal_logprobs,
    rate,
    save_to_file,
    output_dir,
):
    # List to store the generated samples
    sampled_sequences = []
    sampled_decoded_sequences = []
    sampled_target_logprobs = []
    # proposal_logprobs = []

    # Calculate the number of burn-in samples
    burnin_index = int(burnin * sequence_count)

    # Get the first sequence and its probabilities
    current_sequence = sequences[0]
    logprob_target_current, logprob_proposal_current = (
        target_logprobs[0].item(),
        proposal_logprobs[0].item(),
    )

    # This is a top-level loop to generate multiple sequences
    for i in range(1, sequence_count):
        # Get the sequence to propose
        proposed_sequence = sequences[i]
        # Get the probabilities for the proposed sequences
        logprob_target_proposed, logprob_proposal_proposed = (
            target_logprobs[i].item(),
            proposal_logprobs[i].item(),
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
            # Decode the generated sequence
            # current_decoded_seq = tokenizer.decode(
            #     current_sequence, skip_special_tokens=True
            # )
            # Append the decoded sequence and its probabilities to samples
            # sampled_decoded_sequences.append(current_decoded_seq)
            sampled_target_logprobs.append(logprob_target_current)

    if save_to_file:
        with open(create_filename("sampled_sequences", "pt", output_dir), "wb") as f:
            torch.save(sampled_sequences, f)
        # with open(
        #     create_filename("sampled_decoded_sequences", "json", output_dir), "w"
        # ) as f:
        #     json.dump(sampled_decoded_sequences, f)
        with open(
            create_filename("sampled_target_logprobs", "json", output_dir), "w"
        ) as f:
            json.dump(sampled_target_logprobs, f)

    # return sampled_sequences, sampled_decoded_sequences, sampled_target_logprobs

    return sampled_sequences, sampled_target_logprobs


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
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save the output files.",
    )

    args = parser.parse_args()
    return args


def main():
    # Parse command-line arguments
    args = parse_args()
    sequence_count = args.sequence_count
    burnin = args.burnin
    rate = args.rate
    seed = args.seed
    output_dir = args.output_dir

    # Add a directory with a timestamp to the output directory
    output_dir = os.path.join(output_dir, get_timestamp())
    # Create a directory to save the output files
    os.makedirs(output_dir, exist_ok=True)
    # Save log messages to a file
    setup_logging(log_file=os.path.join(output_dir, "log.txt"))
    # Save command-line arguments to JSON
    save_args(args, output_dir)

    # TODO: Load sequences, target_logprobs, proposal_logprobs

    # Set random seed for reproducibility numpy
    np.random.seed(seed)

    # Run the Independent Metropolis-Hastings algorithm
    start_time = time.time()
    logging.info("Running Independent Metropolis-Hastings algorithm...")
    (
        sampled_sequences,
        sampled_decoded_sequences,
        sampled_logprobs,
    ) = metropolis_hastings(
        # tokenizer=tokenizer,
        sequence_count=sequence_count,
        burnin=burnin,
        sequences=sequences,
        target_logprobs=target_logprobs,
        proposal_logprobs=proposal_logprobs,
        rate=rate,
        save_to_file=True,
        output_dir=os.path.join(output_dir, "mh"),
    )
    end_time = time.time()
    logging.info(
        f"Finished running the algorithm in {end_time - start_time:.2f} seconds."
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
