import os
import logging
import random
import secrets

from evaluate import load

from src.utils.utils import save_to_json, timer, convert_to_dict, load_data_from_jsonl

from src.eval.download_dataset import download_dataset


def evaluate_mauve(
    args,
    output_subdir,
    eval_local_decoding_texts,
    eval_global_decoding_texts,
    eval_num_sequences,
    seed,
):
    # Parse command-line arguments
    eval_dataset_name = args.eval_dataset_name
    eval_split = args.eval_split
    max_length = args.max_length
    mcmc_num_samples = args.mcmc_num_samples

    # Set the device ID
    device_id = 1 if args.device == "cuda" else 0

    # Set the number of evaluated sequnces to the number of sampled sequences
    if eval_num_sequences is None:
        eval_num_sequences = mcmc_num_samples

    # Set the output subdirectory
    subdir = "data"
    # Download the dataset
    logging.info(f"Downloading the {eval_dataset_name} dataset...")
    download_dataset(subdir=subdir, dataset=eval_dataset_name, splits=[eval_split])
    # Path to the dataset file
    file_path = os.path.join(subdir, f"{eval_dataset_name}.{eval_split}.jsonl")
    data = load_data_from_jsonl(file_path)

    # Load the reference texts
    reference_texts = [item["text"] for item in data]

    # Pick random run_eval_sequences
    reference_texts = random.sample(reference_texts, eval_num_sequences)

    with timer("Evaluating the generated sequences..."):
        # Generate a unique experiment ID
        experiment_id = secrets.token_hex(3)
        # Initialize MAUVE metric
        mauve = load("mauve", experiment_id=experiment_id)
        # Compute MAUVE results for locally decoded strings
        logging.info(
            f"Evaluating {len(eval_local_decoding_texts)} locally decoded strings and {len(reference_texts)} reference strings..."
        )
        mauve_results_local = mauve.compute(
            predictions=eval_local_decoding_texts,
            references=reference_texts,
            device_id=device_id,
            max_text_length=max_length,
            seed=seed,
        )
        logging.info(
            f"Evaluating {len(eval_global_decoding_texts)} globally decoded strings and {len(reference_texts)} reference strings..."
        )
        # Compute MAUVE results for globally decoded strings
        mauve_results_global = mauve.compute(
            predictions=eval_global_decoding_texts,
            references=reference_texts,
            device_id=device_id,
            max_text_length=max_length,
            seed=seed,
        )

    logging.info(f"MAUVE score for locally decoded strings: {mauve_results_local.mauve}")
    logging.info(f"MAUVE score for globally decoded strings: {mauve_results_global.mauve}")

    # Save the MAUVE results to a JSON file
    logging.info("Saving the MAUVE evaluation results...")
    mauve_results_local_dict = convert_to_dict(mauve_results_local)
    mauve_results_global_dict = convert_to_dict(mauve_results_global)
    save_to_json(mauve_results_local_dict, "mauve_results_local", output_subdir)
    save_to_json(mauve_results_global_dict, "mauve_results_global", output_subdir)

    return mauve_results_local, mauve_results_global


def evaluate_bleu(output_subdir, eval_local_decoding_texts, eval_global_decoding_texts):
    def compute_self_bleu(texts):
        # Generate a unique experiment ID
        experiment_id = secrets.token_hex(3)
        # Load the BLEU metric
        bleu = load("bleu", experiment_id=experiment_id)

        self_bleu_scores = []

        for i, prediction in enumerate(texts):
            individual_bleu_scores = []
            for j, reference in enumerate(texts):
                if i != j:
                    # Compute BLEU score for this prediction-reference pair
                    bleu_score = bleu.compute(predictions=[prediction], references=[[reference]])["bleu"]
                    individual_bleu_scores.append(bleu_score)
            # Average BLEU scores for this prediction
            self_bleu_scores.append(sum(individual_bleu_scores) / len(individual_bleu_scores))

        # Return the average Self-BLEU score
        return sum(self_bleu_scores) / len(self_bleu_scores)

    # Compute Self-BLEU for local decoding texts
    logging.info(f"Evaluating Self-BLEU for {len(eval_local_decoding_texts)} locally decoded texts...")
    local_self_bleu = compute_self_bleu(eval_local_decoding_texts)

    # Compute Self-BLEU for global decoding texts
    logging.info(f"Evaluating Self-BLEU for {len(eval_global_decoding_texts)} globally decoded texts...")
    global_self_bleu = compute_self_bleu(eval_global_decoding_texts)

    logging.info(f"Self-BLEU score for locally decoded texts: {local_self_bleu}")
    logging.info(f"Self-BLEU score for globally decoded texts: {global_self_bleu}")

    # Save the Self-BLEU results to a JSON file
    logging.info("Saving the Self-BLEU evaluation results...")
    self_bleu_results = {
        "local_self_bleu": local_self_bleu,
        "global_self_bleu": global_self_bleu,
    }
    save_to_json(self_bleu_results, "self_bleu_results", output_subdir)

    return local_self_bleu, global_self_bleu


def evaluate(
    args,
    output_subdir,
    eval_local_decoding_texts,
    eval_global_decoding_texts,
    eval_num_sequences,
    seed,
):
    # Evaluate the generated sequences using the MAUVE metric
    mauve_results_local, mauve_results_global = evaluate_mauve(
        args,
        output_subdir,
        eval_local_decoding_texts,
        eval_global_decoding_texts,
        eval_num_sequences,
        seed,
    )

    # Evaluate the generated sequences using the BLEU metric
    bleu_results_local, bleu_results_global = evaluate_bleu(
        output_subdir,
        eval_local_decoding_texts,
        eval_global_decoding_texts,
    )

    return (
        mauve_results_local,
        mauve_results_global,
        bleu_results_local,
        bleu_results_global,
    )
