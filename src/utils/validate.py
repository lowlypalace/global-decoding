import sys


def validate_args(args):
    if args.top_k is not None and args.top_p is not None:
        sys.exit("Either --top_k or --top_p should be provided, not both.")

    if args.top_k is not None and args.top_k <= 0:
        sys.exit(f"--top_k must be an int > 0, but is {args.top_k}")

    if args.top_p is not None and (args.top_p < 0 or args.top_p > 1.0):
        sys.exit(f"`top_p` has to be a float > 0 and < 1, but is {args.top_p}")

    if args.sequence_count <= 0:
        sys.exit(f"--sequence_count must be an int > 0, but is {args.sequence_count}")

    if args.max_length is not None and args.max_length <= 0:
        sys.exit(f"--max_length must be an int > 0, but is {args.max_length}")

    if args.batch_size_seq <= 0:
        sys.exit(f"--batch_size_seq must be an int > 0, but is {args.batch_size_seq}")

    if args.batch_size_prob <= 0:
        sys.exit(f"--batch_size_prob must be an int > 0, but is {args.batch_size_prob}")

    if args.mcmc_num_samples <= 0:
        sys.exit(f"--mcmc_num_samples must be an int > 0, but is {args.mcmc_num_samples}")

    if args.mcmc_num_samples > args.sequence_count:
        sys.exit(f"--mcmc_num_samples must be less than or equal to --sequence_count, but is {args.mcmc_num_samples}")

    if args.eval_num_sequences is not None and args.eval_num_sequences > args.sequence_count:
        sys.exit(
            f"--eval_num_sequences must be less than or equal to --sequence_count, but is {args.eval_num_sequences}"
        )
