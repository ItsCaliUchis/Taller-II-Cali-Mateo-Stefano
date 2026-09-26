# path to the data folder

from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"

dataset_raw = DATA / "dataset_raw.csv"
dataset_clean = DATA / "dataset_clean.csv"