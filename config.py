from pathlib import Path

# Path to the data folder
DATA = Path(__file__).resolve().parent / "data"
DATA.mkdir(exist_ok=True)

dataset_raw = DATA / "dataset_raw.csv"
dataset_clean = DATA / "dataset_clean.csv"

dataset_raw = DATA / "dataset_raw.csv"
dataset_clean = DATA / "dataset_clean.csv"