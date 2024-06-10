import torch
import logging


def mask_out_pad_token(logprobs, sequences, pad_token_id):
    mask = (
        sequences == pad_token_id
    )  # Create a mask where sequences are equal to pad_token_id.
    cumulative_mask = (
        mask.cumsum(dim=1) > 1
    )  # Create a cumulative sum to identify subsequent pads after the first one.
    logprobs.masked_fill_(
        cumulative_mask, 0
    )  # Set log probabilities to zero where cumulative_mask is True.
    return logprobs


def generate_sequences_and_probs_hf(
    model,
    tokenizer,
    input_ids,
    max_length,
    top_k,
    top_p,
    batch_size,
    sequence_count,
):
    logging.info(
        f"Generating {sequence_count} sequences in batches of size {batch_size} and computing log probabilities..."
    )

    # Container for all generated sequences
    sequences_ids = []

    # Placeholder for the log probability sums
    target_logprob_sums = torch.tensor([], device=input_ids.device)
    proposal_logprob_sums = torch.tensor([], device=input_ids.device)

    # Save the log probabilities for each token in the sequences
    proposal_logprobs_tokens = torch.tensor([], device=input_ids.device)
    target_logprobs_tokens = torch.tensor([], device=input_ids.device)

    with torch.no_grad():
        while len(sequences_ids) < sequence_count:
            # Generate a batch of sequences
            sequences_batch = model.generate(
                input_ids=input_ids,
                max_length=max_length,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                num_return_sequences=batch_size,
                min_new_tokens=1,  # We don't want to generate empty sequences
                return_dict_in_generate=True,
                output_scores=True,
                output_logits=True,
                use_cache=False,
            )

            # Only use the IDs that were generated, excluding the input IDs
            gen_sequences_ids = sequences_batch.sequences[:, input_ids.shape[-1] :]
            index = gen_sequences_ids[:, :, None]

            logprobs_proposal = torch.stack(sequences_batch.scores, dim=1).log_softmax(
                -1
            )
            selected_logprobs_proposal = torch.gather(
                logprobs_proposal, 2, index
            ).squeeze(-1)
            del logprobs_proposal

            logprobs_target = torch.stack(sequences_batch.logits, dim=1).log_softmax(-1)
            selected_logprobs_target = torch.gather(logprobs_target, 2, index).squeeze(
                -1
            )
            del logprobs_target

            selected_logprobs_proposal = mask_out_pad_token(
                selected_logprobs_proposal, gen_sequences_ids, tokenizer.pad_token_id
            )
            selected_logprobs_target = mask_out_pad_token(
                selected_logprobs_target, gen_sequences_ids, tokenizer.pad_token_id
            )

            proposal_logprob_sum = torch.sum(selected_logprobs_proposal, dim=-1)
            target_logprob_sum = torch.sum(selected_logprobs_target, dim=-1)

            proposal_logprob_sums = torch.cat(
                (proposal_logprob_sums, proposal_logprob_sum)
            )
            target_logprob_sums = torch.cat((target_logprob_sums, target_logprob_sum))

            proposal_logprobs_tokens = torch.cat(
                (proposal_logprobs_tokens, selected_logprobs_proposal)
            )
            target_logprobs_tokens = torch.cat(
                (target_logprobs_tokens, selected_logprobs_target)
            )

            # Collect the generated sequences
            sequences_ids.extend(gen_sequences_ids)

            # If we've generated enough sequences, stop
            if len(sequences_ids) >= sequence_count:
                break

            # Free memory
            del sequences_batch
            del gen_sequences_ids
            del index
            del selected_logprobs_proposal
            del selected_logprobs_target
            del proposal_logprob_sum
            del target_logprob_sum
            torch.cuda.empty_cache()

    # If we have more sequences than needed due to the last batch, truncate the list
    sequences_ids = sequences_ids[:sequence_count]
    logging.info(f"Generated {len(sequences_ids)} sequence in total.")

    # If we have more probabilities than needed due to the last batch, truncate the list
    target_logprob_sums = target_logprob_sums[:sequence_count]
    proposal_logprob_sums = proposal_logprob_sums[:sequence_count]

    proposal_logprobs_tokens = proposal_logprobs_tokens[:sequence_count]
    target_logprobs_tokens = target_logprobs_tokens[:sequence_count]

    # Decode sequences to text
    sequences_decoded = tokenizer.batch_decode(sequences_ids, skip_special_tokens=True)

    return (
        torch.stack(sequences_ids),
        sequences_decoded,
        target_logprob_sums,
        proposal_logprob_sums,
        target_logprobs_tokens,
        proposal_logprobs_tokens,
    )
