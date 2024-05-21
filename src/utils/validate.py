import sys


def validate_args(args):
    if args.top_k is not None and args.top_k <= 0:
        sys.exit("--top_k must be a positive number.")

    if args.top_p is not None and args.top_p <= 0:
        sys.exit("--top_p must be a positive number.")

    if args.sequence_count <= 0:
        sys.exit("--sequence_count must be a positive number.")

    if args.max_length is not None and args.max_length <= 0:
        sys.exit("--max_length must be a positive number.")

    if args.batch_size_seq <= 0:
        sys.exit("--batch_size_seq must be a positive number")

    if args.batch_size_prob <= 0:
        sys.exit("--batch_size_prob must be a positive number.")

    if args.mcmc_num_subsets <= 0:
        sys.exit("--mcmc_num_subsets must be a positive number.")

    if (
        args.eval_num_sequences is not None
        and args.eval_num_sequences > args.sequence_count
    ):
        sys.exit("--eval_num_sequences must be equal or less than --sequence_count.")
