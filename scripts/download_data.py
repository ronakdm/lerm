"""
Download data from UCI repository.
"""

import time
import sys
from zipfile import BadZipFile

sys.path.append(".")
from src.utils.data import load_dataset
from src.utils.training import format_time

tic = time.time()
for dataset in ["yacht", "energy", "concrete"]:
    print(f"Downloading dataset {dataset}...")
    try:
        load_dataset(dataset)
    except BadZipFile:
        pass
toc = time.time()
print("Downloads complete!")
print(f"Time:         {format_time(toc-tic)}.")
