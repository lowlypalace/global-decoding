import os
import json
import logging
import contextlib
import time
import logging.handlers
from datetime import datetime
import random

import torch
import numpy as np

from types import SimpleNamespace


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextlib.contextmanager
def timer(description: str):
    logging.info(f"{description}...")
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    logging.info(f"{description} completed in {end - start:.2f} seconds.")


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


def get_timestamp():
    # Get the current time
    current_time = datetime.now()
    # Format the time in a user-friendly format
    time_str = current_time.strftime("%d-%m-%Y_%H-%M-%S")

    return time_str


def create_filename(name, extension, directory):
    # Create the filename
    filename = f"{name}.{extension}"
    # Create the directory if it does not exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    # Define the full path for the file
    full_path = os.path.join(directory, filename)

    return full_path


def save_to_json(data, base_name, subdir):
    filename = create_filename(base_name, "json", subdir)
    with open(filename, "w") as f:
        json.dump(data, f)


def load_from_json(file_path):
    with open(f"{file_path}.json", "r") as f:
        data = json.load(f)
    return data


def load_data_from_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        data = [json.loads(line) for line in lines]
    return data


def convert_to_dict(obj):
    if isinstance(obj, SimpleNamespace):
        return {k: convert_to_dict(v) for k, v in vars(obj).items()}
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy arrays to list
    return obj


def convert_tensor_to_list(data):
    if isinstance(data, torch.Tensor):
        if data.ndim == 0:  # It's a scalar tensor
            return data.item()
        else:
            return data.tolist()
    elif isinstance(data, list):
        return [convert_tensor_to_list(item) for item in data]
    else:
        raise TypeError("Input must be a torch.Tensor or a list of torch.Tensors")
