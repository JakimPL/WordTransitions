from collections import defaultdict, Counter
from pathlib import Path
import pickle
import random
from typing import Any, Collection, DefaultDict, Dict, List, Optional, Tuple, Union

from fast_edit_distance import edit_distance
from tqdm.notebook import tqdm

def save_pickle(data: Any, path: Path):
    with open(path, "wb") as file:
        pickle.dump(data, file)

def load_pickle(path: Path) -> Any:
    with open(path, "rb") as file:
        return pickle.load(file)
