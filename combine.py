import os
import json
import secrets
from src.mcmc import run_mcmc
from src.eval import evaluate

def find_sequences_and_probs(input_dir, top_k, top_p, model_name):
    sequences_ids = []
    sequences_decoded = []
    logprobs_target = []
    logprobs_proposal = []
    proposal_normalize_constans = []
    target_normalize_constants = []

    input_dir = os.path.join(input_dir, model_name)

    # List all directories in the input directory
    directories = [
        d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))
    ]

    for directory in directories:
        metadata_file = os.path.join(input_dir, directory, "metadata.json")

        # Read metadata.json to check if the directory matches the criteria
        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        if (
            metadata["top_k"] == top_k
            and metadata["top_p"] == top_p
            and metadata["model_name"] == model_name
        ):
            # For debugging
            print(f"Found matching directory: {directory}")

            with open(
                os.path.join(input_dir, directory, "sequences", "sequences_ids.json"),
                "r",
            ) as f:
                sequences_ids.extend(json.load(f))
            with open(
                os.path.join(
                    input_dir, directory, "sequences", "sequences_decoded.json"
                ),
                "r",
            ) as f:
                sequences_decoded.extend(json.load(f))
            with open(
                os.path.join(input_dir, directory, "sequences", "logprobs_target.json"),
                "r",
            ) as f:
                logprobs_target.extend(json.load(f))
            with open(
                os.path.join(
                    input_dir, directory, "sequences", "logprobs_proposal.json"
                ),
                "r",
            ) as f:
                logprobs_proposal.extend(json.load(f))
            with open(
                os.path.join(
                    input_dir,
                    directory,
                    "sequences",
                    "proposal_normalize_constants.json",
                ),
                "r",
            ) as f:
                proposal_normalize_constans.extend(json.load(f))
            with open(
                os.path.join(
                    input_dir, directory, "sequences", "target_normalize_constants.json"
                ),
                "r",
            ) as f:
                target_normalize_constants.extend(json.load(f))

    # For debugging
    print(f"Found {len(sequences_ids)} sequences ids")
    print(f"Found {len(sequences_decoded)} decoded sequences")
    print(f"Found {len(logprobs_target)} target logprobs")
    print(f"Found {len(logprobs_proposal)} proposal logprobs")
    print(f"Found {len(proposal_normalize_constans)} proposal normalize constants")
    print(f"Found {len(target_normalize_constants)} target normalize constants")

    return (
        sequences_ids,
        sequences_decoded,
        logprobs_target,
        logprobs_proposal,
        proposal_normalize_constans,
        target_normalize_constants,
    )

if __name__ == "__main__":

    args = {
        "output_dir": "output",
        "top_k": None,
        "top_p": None,
        "model_name": "pythia-2.8b",
        "actions": ["run_mcmc, run_eval_mauve", "run_eval_bleu"],
        "mcmc_num_samples": 200,
        "seed": 0,
        "eval_dataset_name": "webtext",
        "eval_split": "train",
        "max_length": 512
    }

    (
        sequences_ids,
        sequences_decoded,
        logprobs_target,
        logprobs_proposal,
        proposal_normalize_constans,
        target_normalize_constants,
    ) = find_sequences_and_probs(args["output_dir"], args["top_k"], args["top_p"], args["model_name"])

    output_subdir = os.path.join(args["output_dir"], args["model_name"], "combined", secrets.token_hex(3))

    # Save metadata.json to output_subdir with args
    os.makedirs(output_subdir, exist_ok=True)
    with open(os.path.join(output_subdir, "metadata.json"), "w") as f:
        json.dump(args, f)


    _, sampled_sequences_decoded, _ = run_mcmc(
        args=args,
        output_subdir=os.path.join(output_subdir, "mcmc"),
        sequences_ids=sequences_ids,
        sequences_decoded=sequences_decoded,
        target_logprobs=logprobs_target,  # target_logpropbs are probabilities sampled from the global unnormalized distribution
        proposal_logprobs=logprobs_proposal,  # proposal_logprobs are probabilities sampled from the local normalized distribution
    )

    _, _, _, _ = evaluate(
        args,
        output_subdir=os.path.join(output_subdir, "eval"),
        local_decoding_texts=sequences_decoded,  # sequences_decoded are the sequences sampled from the local normalized distribution
        global_decoding_texts=sampled_sequences_decoded,  # sampled_sequences_decoded are the sequences sampled from the global unnormalized distribution
    )

    # TODO: plots for constants distribution
