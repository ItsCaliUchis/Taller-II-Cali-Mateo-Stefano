from pathlib import Path
import json

# Mantener los datos junto al repositorio, sin depender del directorio
# de trabajo actual. El proyecto se instala en modo editable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data"

DATA.mkdir(exist_ok=True)

dataset_raw = DATA / "dataset_raw.csv"
dataset_clean = DATA / "dataset_clean.csv"
bow_vect = DATA / "bow_vect.csv"
negaciones_path = DATA / "negaciones.json"

if Path(negaciones_path).exists():
    with open(negaciones_path, "r", encoding="utf-8") as file:
        negaciones = json.load(file)
else:
    negaciones = []