import os
import json
import logging
import logging.handlers
from datetime import datetime


# def setup_logging():
#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s - %(levelname)s - %(message)s",
#     )


def setup_logging(log_file):
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Create console handler and set level to info
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    # Create file handler and set level to info
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)


def save_args(args, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    args_dict = vars(args)  # Convert argparse Namespace to dictionary
    json_path = os.path.join(output_dir, "metadata.json")
    with open(json_path, "w") as f:
        json.dump(args_dict, f, indent=4)
    logging.info(f"Arguments saved to {json_path}")


def create_filename(name, extension, directory):
    # Get the current time
    current_time = datetime.now()
    # Format the time in a user-friendly format
    time_str = current_time.strftime("%d-%m-%Y_%H-%M-%S")
    # Create the filename with the current time
    filename = f"{name}_{time_str}.{extension}"
    # Create the directory if it does not exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    # Define the full path for the file
    full_path = os.path.join(directory, filename)

    return full_path
