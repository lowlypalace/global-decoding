import os
import logging
import torch
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    AutoTokenizer,
    GPTNeoXForCausalLM,
)
from src.utils.utils import timer, save_to_json, load_from_json, convert_tensor_to_list
from src.sequences.generate_sequences_util import generate_sequences_util
from src.sequences.sequences_probs import get_sequences_probs

# from src.mcmc.plots import plot_distribution

class ModelHandler:
    def __init__(self, model_name, precision, device):
        self.model_name = model_name
        self.precision = precision
        self.device = device
        self.model = None
        self.tokenizer = None

    def setup_model_and_tokenizer(self):
        if not self.model or not self.tokenizer:
            logging.info("Loading model and tokenizer...")
            if self.model_name.startswith("pythia"):
                self.tokenizer = AutoTokenizer.from_pretrained(f"EleutherAI/{self.model_name}")
                self.model = GPTNeoXForCausalLM.from_pretrained(f"EleutherAI/{self.model_name}")
            else:
                self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
                self.model = GPT2LMHeadModel.from_pretrained(self.model_name)

            # Set precision
            if self.precision == "fp16":
                self.model = self.model.half()
            elif self.precision == "fp64":
                self.model = self.model.double()

            # Move model to device and set to evaluation mode
            self.model.eval()
            self.model.to(self.device)

            # Adjust tokenizer
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"

        return self.model, self.tokenizer

    def get_model_and_tokenizer(self):
        return self.setup_model_and_tokenizer()


def encode_input_text(tokenizer, text, device):
    text = text or tokenizer.eos_token
    input_ids = tokenizer.encode(text, add_special_tokens=True, return_tensors="pt").to(device)
    return input_ids


def save_sequences(output_subdir, sequences_ids, sequences_decoded):
    save_to_json(convert_tensor_to_list(sequences_ids), "sequences_ids", output_subdir)
    save_to_json(sequences_decoded, "sequences_decoded", output_subdir)


def set_max_length(model, max_length):
    max_model_length = model.config.max_position_embeddings
    return min(max_length, max_model_length) if max_length else max_model_length



def save_probs(
    output_subdir,
    target_logprobs,
    proposal_logprobs,
    target_logprobs_tokens,
    proposal_logprobs_tokens,
    target_normalize_constants,
    proposal_normalize_constants,
    target_normalize_constants_products,
    proposal_normalize_constants_products,
):
    save_to_json(target_logprobs, "logprobs_target", output_subdir)
    save_to_json(proposal_logprobs, "logprobs_proposal", output_subdir)
    save_to_json(target_logprobs_tokens, "logprobs_target_tokens", output_subdir)
    save_to_json(proposal_logprobs_tokens, "logprobs_proposal_tokens", output_subdir)
    save_to_json(target_normalize_constants, "target_normalize_constants", output_subdir)
    save_to_json(proposal_normalize_constants, "proposal_normalize_constants", output_subdir)
    save_to_json(target_normalize_constants_products, "target_normalize_constants_products", output_subdir)
    save_to_json(proposal_normalize_constants_products, "proposal_normalize_constants_products", output_subdir)

def generate_sequences(args, output_subdir):
    model_handler = ModelHandler(args.model_name, args.precision, args.device)
    model, tokenizer = model_handler.get_model_and_tokenizer()
    input_ids = encode_input_text(tokenizer, args.text, args.device)
    max_length = set_max_length(model, args.max_length)

    with timer("Generating new sequences"):
        sequences_ids, sequences_decoded = generate_sequences_util(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            max_length=max_length,
            top_k=args.top_k,
            top_p=args.top_p,
            sequence_count=int(args.sequence_count * 1.01),  # Generate 1% more sequences
            batch_size=args.batch_size_seq,
        )

    save_sequences(output_subdir, sequences_ids, sequences_decoded)
    return sequences_ids, sequences_decoded

def compute_probs(args, sequences_ids, output_subdir):
    model_handler = ModelHandler(args.model_name, args.precision, args.device)
    model, tokenizer = model_handler.get_model_and_tokenizer()
    input_ids = encode_input_text(tokenizer, args.text, args.device)

    with timer("Computing probabilities"):
        target_logprobs, proposal_logprobs, _, _, _, _, _, _ = get_sequences_probs(
            model=model,
            sequences_ids=sequences_ids,
            top_k=args.top_k,
            top_p=args.top_p,
            pad_token_id=tokenizer.pad_token_id,
            input_ids=input_ids,
            batch_size=args.batch_size_prob,
        )

    save_probs(output_subdir, target_logprobs, proposal_logprobs)
    return target_logprobs, proposal_logprobs

# def generate_sequences_and_probs(args, output_subdir):
#     device = torch.device(args.device)
#     model_handler = ModelHandler(args.model_name, args.precision, device)

#     if args.preload_dir and os.path.exists(os.path.join(output_subdir, "sequences_ids.json")):
#         logging.info("Loading preloaded sequences...")
#         sequences_ids, sequences_decoded = load_sequences(output_subdir, device)
#     else:
#         with timer("Generating new sequences"):
#             model, tokenizer = model_handler.get_model_and_tokenizer()
#             input_ids = encode_input_text(tokenizer, args.text, device)
#             max_length = set_max_length(model, args.max_length)

#             sequences_ids, sequences_decoded = generate_sequences(
#                 model=model,
#                 tokenizer=tokenizer,
#                 input_ids=input_ids,
#                 max_length=max_length,
#                 top_k=args.top_k,
#                 top_p=args.top_p,
#                 sequence_count=int(args.sequence_count * 1.01),  # Generate 1% more sequences
#                 batch_size=args.batch_size_seq,
#             )

#             logging.info("Saving the generated sequences...")
#             save_sequences(output_subdir, sequences_ids, sequences_decoded)

#     if args.preload_dir and os.path.exists(os.path.join(output_subdir, "logprobs_target.json")):
#         logging.info("Loading precomputed probabilities...")
#         target_logprobs, proposal_logprobs, target_logprobs_tokens, proposal_logprobs_tokens = load_probs(
#             output_subdir, device
#         )
#     else:
#         with timer("Computing probabilities"):
#             model, tokenizer = model_handler.get_model_and_tokenizer()
#             input_ids = encode_input_text(tokenizer, args.text, device)

#             (
#                 target_logprobs,
#                 proposal_logprobs,
#                 target_logprobs_tokens,
#                 proposal_logprobs_tokens,
#                 target_normalize_constants,
#                 proposal_normalize_constants,
#                 target_normalize_constants_products,
#                 proposal_normalize_constants_products,
#             ) = get_sequences_probs(
#                 model=model,
#                 sequences_ids=sequences_ids,
#                 top_k=args.top_k,
#                 top_p=args.top_p,
#                 pad_token_id=tokenizer.pad_token_id,
#                 input_ids=input_ids,
#                 batch_size=args.batch_size_prob,
#             )

#             logging.info("Saving the log probabilities...")
#             save_probs(
#                 output_subdir,
#                 target_logprobs,
#                 proposal_logprobs,
#                 target_logprobs_tokens,
#                 proposal_logprobs_tokens,
#                 target_normalize_constants,
#                 proposal_normalize_constants,
#                 target_normalize_constants_products,
#                 proposal_normalize_constants_products,
#             )

#             # logging.info("Plotting the log probabilities distributions...")
#             # plot_distribution(target_logprobs, plot_type="histogram", prefix="target_logprobs",
#             #                   show=False, output_dir=os.path.join(output_subdir, "plots"))
#             # plot_distribution(proposal_logprobs, plot_type="histogram", prefix="proposal_logprobs",
#             #                   show=False, output_dir=os.path.join(output_subdir, "plots"))

#     sequences_ids = convert_tensor_to_list(sequences_ids)
#     return sequences_ids, sequences_decoded, target_logprobs, proposal_logprobs
