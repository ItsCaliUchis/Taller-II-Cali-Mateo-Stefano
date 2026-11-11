from pathlib import Path
import json
import os

# Mantener los datos junto al repositorio, sin depender del directorio
# de trabajo actual. El proyecto se instala en modo editable.
def _find_project_root():
    module_path = Path(__file__).resolve()
    for candidate in module_path.parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src").is_dir():
            return candidate

    configured_root = os.environ.get("TALLER_PROJECT_ROOT")
    candidates = [Path(configured_root)] if configured_root else []
    candidates.append(Path.cwd())
    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate / "pyproject.toml").is_file() and (candidate / "src").is_dir():
            return candidate

    raise RuntimeError(
        "No se pudo localizar la raiz del proyecto. Ejecuta desde el repositorio "
        "o define TALLER_PROJECT_ROOT."
    )


PROJECT_ROOT = _find_project_root()
DATA = PROJECT_ROOT / "data"
CLASSIFICATION_REPORTS = PROJECT_ROOT / "classification_reports"
OPTIMIZATION_RESULTS = PROJECT_ROOT / "optimization_results"

DATA.mkdir(exist_ok=True)
CLASSIFICATION_REPORTS.mkdir(exist_ok=True)
OPTIMIZATION_RESULTS.mkdir(exist_ok=True)

# Rutas utilizadas para la herramienta de registro de comandos.
JSON_COMMANDS = PROJECT_ROOT / "src" / "commands.info.json"
TOML_PATH = PROJECT_ROOT / "pyproject.toml"

# Rutas de los archivos de datos.
dataset_raw = DATA / "dataset_raw.csv"
dataset_clean = DATA / "dataset_clean.csv"
bow_vect = DATA / "bow_vect.npz"
bow_vocab = DATA / "bow_vocab.csv"
idf_bigram = DATA / "idf_bigram.npz"
idf_monogram = DATA / "idf_monogram.npz"
elmo_vect = DATA / "elmo_vect.npz" # exportación de elmo 
elmo_load = DATA / "model_elmo.zip" # carga de elmo
vocabulario_elmo = DATA / "elmo_vocab.csv"
vocabulario_bert = DATA / "bert_vocab.csv"
bert_vect = DATA / "bert_vect.npz"


# Configuración de FastText
MODELO_FASTTEXT = "fasttext-wiki-news-subwords-300"
fasttext_embeddings = DATA / "fasttext_embeddings.npz"
fasttext_vocab = DATA / "ft_vocab.csv"
FASTTEXT_NROWS = None

negaciones_path = DATA / "negaciones.json"

# Rutas de informes de clasificacion
df_sb_sgd = CLASSIFICATION_REPORTS / "classification_report_sb_sgd.csv" # sin optimizar (no balanceado)
df_b_sgd = CLASSIFICATION_REPORTS / "classification_report_b_sgd.csv" # sin optimizar (balanceado)
df_optimizado_sb_sgd = CLASSIFICATION_REPORTS / "classification_report_opt_sb_sgd.csv" # optimizado (no balanceado)
df_optimizado_b_sgd = CLASSIFICATION_REPORTS / "classification_report_opt_b_sgd.csv" # optimizado (balanceado)

# Rutas de Resultados de Optimización de Modelos

resultados_sgd = OPTIMIZATION_RESULTS / "resultados_sgd.json"
PIPELINE_STATE = OPTIMIZATION_RESULTS / "pipeline_state.json"
PIPELINE_LOGS = OPTIMIZATION_RESULTS / "pipeline_logs"
SGD_CHECKPOINTS = OPTIMIZATION_RESULTS / "sgd_checkpoints"

STOP_FILE = OPTIMIZATION_RESULTS / "pipeline.stop"



def ruta_limitada(output_path, limit):
    # Devuelve una ruta alternativa para una salida de procesamiento limitada
    output_path = Path(output_path)
    if limit is None:
        return output_path
    if limit <= 0:
        raise ValueError("limit debe ser un entero positivo")
    return output_path.with_name(
        f"{output_path.stem}_limit_{limit}{output_path.suffix}"
    )

def cargar_negaciones():
    if Path(negaciones_path).exists():
        with open(negaciones_path, "r", encoding="utf-8") as file:
            contenido = file.read().strip()
            return json.loads(contenido) if contenido else []

    return []

def cargar_resultados_sgd():
    if Path(resultados_sgd).exists():
        with open(resultados_sgd, "r", encoding="utf-8") as file:
            contenido = file.read().strip()
            return json.loads(contenido) if contenido else {}

    return {}
