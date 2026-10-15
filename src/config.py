from pathlib import Path

# Mantener los datos junto al repositorio, sin depender del directorio
# de trabajo actual. El proyecto se instala en modo editable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data"
DATA.mkdir(exist_ok=True)

dataset_raw = DATA / "dataset_raw.csv"
dataset_clean = DATA / "dataset_clean.csv"
