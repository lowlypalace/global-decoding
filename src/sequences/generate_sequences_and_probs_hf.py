import torch
import logging
import logging
import torch


def generate_sequences_and_probs_hf(
    model,
    tokenizer,
    input_ids,
    max_length,
    top_k,
    batch_size,
    sequence_count,
):
    logging.info(
        f"Generating {sequence_count} sequences in batches of size {batch_size} and computing log probabilities..."
    )

    # Container for all generated sequences
    sequences_ids = []
    target_logprobs_sums = []
    proposal_logprobs_sums = []

    with torch.no_grad():
        while len(sequences_ids) < sequence_count:
            # Generate a batch of sequences
            sequences_batch = model.generate(
                input_ids=input_ids,
                max_length=max_length,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                top_k=top_k,
                do_sample=True,
                num_return_sequences=batch_size,
                min_new_tokens=1,  # We don't want to generate empty sequences
                return_dict_in_generate=True,
                output_scores=True,
                output_logits=True,
            )

            # Only use the IDs that were generated, excluding the input IDs
            gen_sequences_ids = sequences_batch.sequences[:, input_ids.shape[-1] :]

            logprobs_target = torch.stack(sequences_batch.logits, dim=1).log_softmax(-1)
            logprobs_proposal = torch.stack(sequences_batch.scores, dim=1).log_softmax(
                -1
            )

            selected_logprobs_proposal = torch.gather(
                logprobs_proposal, 2, gen_sequences_ids[:, :, None]
            ).squeeze(-1)
            selected_logprobs_target = torch.gather(
                logprobs_target, 2, gen_sequences_ids[:, :, None]
            ).squeeze(-1)

            # Set logits of padding tokens to 0 instead of -inf
            selected_logprobs_proposal[selected_logprobs_proposal == float("-inf")] = 0

            proposal_logprob_sum = torch.sum(selected_logprobs_proposal, dim=-1)
            target_logprob_sum = torch.sum(selected_logprobs_target, dim=-1)

            proposal_logprobs_sums.extend(proposal_logprob_sum)
            target_logprobs_sums.extend(target_logprob_sum)

            # Collect the generated sequences
            sequences_ids.extend(gen_sequences_ids)

            # If we've generated enough sequences, stop
            if len(sequences_ids) >= sequence_count:
                break

            # Free memory
            del sequences_batch
            del gen_sequences_ids
            del logprobs_target
            del logprobs_proposal
            del selected_logprobs_proposal
            del selected_logprobs_target
            del proposal_logprob_sum
            del target_logprob_sum

            torch.cuda.empty_cache()

    # If we have more sequences than needed due to the last batch, truncate the list
    sequences_ids = sequences_ids[:sequence_count]
    logging.info(f"Generated {len(sequences_ids)} sequence in total.")

    # Decode sequences to text
    sequences_decoded = tokenizer.batch_decode(sequences_ids, skip_special_tokens=True)

    return (
        torch.stack(sequences_ids),
        sequences_decoded,
        target_logprobs_sums,
        proposal_logprobs_sums,
    )
