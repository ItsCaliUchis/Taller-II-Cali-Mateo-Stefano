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
idf_monogram = DATA / "idf_monogram.csv"
elmo_vect = DATA / "elmo_vect.npz" # exportación de elmo 
elmo_load = DATA / "model_elmo.zip" # carga de elmo
vocabulario_elmo = DATA / "elmo_vocab.csv"
vocabulario_bert = DATA / "bert_vocab.csv"
bert_vect = DATA / "bert_vect.npz"

# Configuración de FastText
MODELO_FASTTEXT = "fasttext-wiki-news-subwords-300"
fasttext_embeddings = DATA / "fasttext_embeddings.csv"
FASTTEXT_NROWS = None

negaciones_path = DATA / "negaciones.json"

def cargar_negaciones():
    if Path(negaciones_path).exists():
        with open(negaciones_path, "r", encoding="utf-8") as file:
            contenido = file.read().strip()
            return json.loads(contenido) if contenido else []

    return []