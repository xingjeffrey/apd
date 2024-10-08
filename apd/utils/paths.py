from pathlib2 import Path
import pathlib2
import os
from datetime import datetime
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = Path('/mnt/cube/j8xing/kai_apd/data')
<<<<<<< HEAD
MODEL_DIR = Path('/mnt/cube/j8xing/kai_apd/models')
=======
>>>>>>> 5f0cb13139a31f1c82ea627972eeca53589dee08
ZENODO_DIR = Path('/mnt/cube/Datasets/public_ds_starlings_ts_2019/')
FIGURE_DIR = PROJECT_DIR / "figures"

def ensure_dir(file_path):
    """ create a safely nested folder
    """
    if type(file_path) == str:
        if "." in os.path.basename(os.path.normpath(file_path)):
            directory = os.path.dirname(file_path)
        else:
            directory = os.path.normpath(file_path)
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
            except FileExistsError as e:
                # multiprocessing can cause directory creation problems
                print(e)
    elif type(file_path) == pathlib2.PosixPath:
        # if this is a file
        if len(file_path.suffix) > 0:
            file_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            file_path.mkdir(parents=True, exist_ok=True)