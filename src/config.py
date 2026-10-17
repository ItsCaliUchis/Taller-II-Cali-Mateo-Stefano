from pathlib import Path
import json

# Mantener los datos junto al repositorio, sin depender del directorio
# de trabajo actual. El proyecto se instala en modo editable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data"

DATA.mkdir(exist_ok=True)

# Rutas utilizadas para la herramienta de registro de comandos.
JSON_COMMANDS = PROJECT_ROOT / "src" / "commands.info.json"
TOML_PATH = PROJECT_ROOT / "pyproject.toml"

# Rutas de los archivos de datos.
dataset_raw = DATA / "dataset_raw.csv"
dataset_clean = DATA / "dataset_clean.csv"
bow_vect = DATA / "bow_vect.csv"
idf_bigram = DATA / "idf_bigram.csv"
negaciones_path = DATA / "negaciones.json"

def cargar_negaciones():
    if Path(negaciones_path).exists():
        with open(negaciones_path, "r", encoding="utf-8") as file:
            contenido = file.read().strip()
            return json.loads(contenido) if contenido else []

    return []