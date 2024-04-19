import os
from datetime import datetime


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


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
