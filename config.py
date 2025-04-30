"""Configuration settings for the MiMo-7B tool-calling project."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.absolute()
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"

for dir_path in [DATA_DIR, MODEL_DIR, OUTPUT_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

BASE_MODEL_ID = "XiaomiMiMo/MiMo-7B-RL"
OUTPUT_MODEL_ID = "XiaomiMiMo/MiMo-7B-RL-Tool"

NUM_EXAMPLES = 1000  # Number of examples to generate
