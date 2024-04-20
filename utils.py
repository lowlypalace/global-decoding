import os
import logging
import json
from datetime import datetime


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


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
