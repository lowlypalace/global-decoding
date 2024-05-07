import sys

def validate_args(args):
    if args.top_k is not None and args.top_k <= 0:
        sys.exit("Error: --top_k must be a positive number.")
    if args.top_p is not None and args.top_p <= 0:
        sys.exit("Error: --top_p must be a positive number.")
    if args.sequence_count <= 0:
        sys.exit("Error: --sequence_count must be a positive number.")
    if args.max_length is not None and args.max_length <= 0:
        sys.exit("Error: --max_length must be a positive number.")
    if args.batch_size_seq <= 0 or args.batch_size_seq > args.sequence_count:
        sys.exit("Error: --batch_size_seq must be a positive number")
    if args.batch_size_seq > args.sequence_count:
        sys.exit("Error: --batch_size_prob must be not larger than --sequence_count.")
    if args.batch_size_prob <= 0 or args.batch_size_prob > args.sequence_count:
        sys.exit("Error: --batch_size_prob must be a positive number.")
    if args.batch_size_prob > args.sequence_count:
        sys.exit("Error: --batch_size_prob must be not larger than --sequence_count.")
    if args.burnin <= 0:
        sys.exit("Error: --burnin must be a positive number.")
    if args.sample_rate <= 0:
        sys.exit("Error: --sample_rate must be a positive number.")
    if args.eval_num_sequences > args.sequence_count:
        sys.exit("Error: --eval_num_sequences must be equal or less than --sequence_count.")
