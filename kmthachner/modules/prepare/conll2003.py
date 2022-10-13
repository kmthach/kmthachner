import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import datasets
from constants.dataset_constants import DATASET_NAME


def load_data():
    # Load raw data from datasets library
    raw_datasets = datasets.load_dataset(DATASET_NAME)
    return raw_datasets
