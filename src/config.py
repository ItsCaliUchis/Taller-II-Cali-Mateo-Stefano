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
SGD_REPORTS = CLASSIFICATION_REPORTS / "sgd"
RNA_REPORTS = CLASSIFICATION_REPORTS / "rna"
LINEAR_SVC_REPORTS = CLASSIFICATION_REPORTS / "linear_svc"
SVM_REPORTS = CLASSIFICATION_REPORTS / "svm"
OPTIMIZATION_RESULTS = PROJECT_ROOT / "optimization_results"
SVM_CHECKPOINTS = OPTIMIZATION_RESULTS / "svm"
DRIVE_FOLDER_ID = "1wFGB-jh0yyq-igyCQPYAzm-hwppNYLyu"
DRIVE_FOLDER_URL = f"https://drive.google.com/drive/folders/{DRIVE_FOLDER_ID}?usp=sharing"
DRIVE_REMOTE_MANIFEST = DATA / "ListadoRemoto.json"
DRIVE_DOWNLOAD_MANIFEST = DATA / "ListadoDescarga.json"
MODELS = PROJECT_ROOT / "models"
LINEAR_SVC_MODELS = MODELS / "linear_svc"
SVM_MODELS = MODELS / "svm"
RNA_DRIVE_BUNDLE = DATA / "rna_modelos_finales.zip"

DATA.mkdir(exist_ok=True)
CLASSIFICATION_REPORTS.mkdir(exist_ok=True)
OPTIMIZATION_RESULTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

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

# Reportes de SGD.
df_sb_sgd = SGD_REPORTS / "classification_report_sb_sgd.csv"
df_b_sgd = SGD_REPORTS / "classification_report_b_sgd.csv"
df_optimizado_sb_sgd = SGD_REPORTS / "classification_report_opt_sb_sgd.csv"
df_optimizado_b_sgd = SGD_REPORTS / "classification_report_opt_b_sgd.csv"

# Rutas de reportes de LinearSVC, separados por variante.
LINEAR_SVC_BASELINE_REPORTS = LINEAR_SVC_REPORTS / "sin_optimizacion"
LINEAR_SVC_OPTIMIZED_REPORTS = LINEAR_SVC_REPORTS / "con_optimizacion"

# CSV base
df_sin_scaler_linear_svc = (
    LINEAR_SVC_BASELINE_REPORTS / "classification_report_sin_escalado.csv"
)
df_con_scaler_linear_svc = (
    LINEAR_SVC_BASELINE_REPORTS / "classification_report_con_escalado.csv"
)

# CSV optimizados
df_opt_sin_scaler_linear_svc = (
    LINEAR_SVC_OPTIMIZED_REPORTS / "classification_report_sin_escalado.csv"
)
df_opt_con_scaler_linear_svc = (
    LINEAR_SVC_OPTIMIZED_REPORTS / "classification_report_con_escalado.csv"
)

# Gráficos base
grafico_matrices_sin_scaler_linear_svc = (
    LINEAR_SVC_BASELINE_REPORTS / "matrices_sin_escalado.png"
)
grafico_matrices_con_scaler_linear_svc = (
    LINEAR_SVC_BASELINE_REPORTS / "matrices_con_escalado.png"
)
grafico_curvas_sin_scaler_linear_svc = (
    LINEAR_SVC_BASELINE_REPORTS / "curvas_sin_escalado.png"
)
grafico_curvas_con_scaler_linear_svc = (
    LINEAR_SVC_BASELINE_REPORTS / "curvas_con_escalado.png"
)

# Gráficos optimizados
grafico_matrices_opt_sin_scaler_linear_svc = (
    LINEAR_SVC_OPTIMIZED_REPORTS / "matrices_sin_escalado.png"
)
grafico_matrices_opt_con_scaler_linear_svc = (
    LINEAR_SVC_OPTIMIZED_REPORTS / "matrices_con_escalado.png"
)
grafico_curvas_opt_sin_scaler_linear_svc = (
    LINEAR_SVC_OPTIMIZED_REPORTS / "curvas_sin_escalado.png"
)
grafico_curvas_opt_con_scaler_linear_svc = (
    LINEAR_SVC_OPTIMIZED_REPORTS / "curvas_con_escalado.png"
)

# Rutas de Resultados de Optimización de Modelos

resultados_sgd = OPTIMIZATION_RESULTS / "resultados_sgd.json"
PIPELINE_STATE = OPTIMIZATION_RESULTS / "pipeline_state.json"
PIPELINE_PLAN = OPTIMIZATION_RESULTS / "plan.json"
PIPELINE_LOGS = OPTIMIZATION_RESULTS / "pipeline_logs"
SGD_CHECKPOINTS = OPTIMIZATION_RESULTS / "sgd_checkpoints"
RNA_CHECKPOINTS = OPTIMIZATION_RESULTS / "rna_checkpoints"
RNA_RESULTS = OPTIMIZATION_RESULTS / "rna_results"
TRAINING_CACHE = OPTIMIZATION_RESULTS / "training_cache"

# Entrenamiento y artefactos SVM optimizado.
SVM_RESULTS = SVM_REPORTS
SVM_CACHE = TRAINING_CACHE / "svm_cache"
SVM_CHECKPOINT = SVM_CHECKPOINTS / "resultados_svm.json"
SVM_SUMMARY = SVM_RESULTS / "resultados_svm.csv"
SVM_ARTIFACTS = SVM_MODELS

svm_dataset = dataset_clean
svm_embeddings = {
    "BoW": bow_vect,
    "TF-IDF Bigrama": idf_bigram,
    "TF-IDF Monogram": idf_monogram,
    "FastText": fasttext_embeddings,
}


def rna_report_dir(embedding):
    """Crea y devuelve la carpeta de reportes de una representación RNA."""
    if embedding not in {"bow", "idf-monogram", "idf-bigram", "fasttext"}:
        raise ValueError(f"Representación RNA desconocida: {embedding}")
    directory = RNA_REPORTS / embedding
    directory.mkdir(parents=True, exist_ok=True)
    return directory

STOP_FILE = OPTIMIZATION_RESULTS / "pipeline.stop"

# Rutas de almacenado de modelos

models_sgd = MODELS / "sgd_opt"
sgd_balanced = models_sgd / "balanced"
sgd_nbalanced = models_sgd / "not_balanced"

for directory in (
    SGD_REPORTS,
    LINEAR_SVC_BASELINE_REPORTS,
    LINEAR_SVC_OPTIMIZED_REPORTS,
    SVM_REPORTS,
    SVM_CHECKPOINTS,
    SVM_CACHE,
    LINEAR_SVC_MODELS,
    SVM_MODELS,
    sgd_balanced,
    sgd_nbalanced,
):
    directory.mkdir(parents=True, exist_ok=True)



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
