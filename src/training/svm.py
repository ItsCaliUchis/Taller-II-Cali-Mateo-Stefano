
"""Optimización y evaluación reanudable de LinearSVC sobre embeddings."""


# ============================================================
# 1. IMPORTS
# ============================================================

import argparse
import gc
import json
import time
import hashlib
import platform

from pathlib import Path
from contextlib import contextmanager

import numpy as np
import pandas as pd
import psutil
import joblib

from scipy.sparse import load_npz

from sklearn.base import clone
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    ParameterGrid,
)

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

import sklearn
import scipy
import re

from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

from joblib import Memory, cpu_count
from scipy.sparse import issparse
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm

import config
from utils import ejecutar_etapa, ejecutar_grid_search
from .data import load_sentiment_labels
from .graphics import graficar_matrices_confusion


# ============================================================
# 2. DIAGNÓSTICO DEL PROCESADOR
# ============================================================

def diagnosticar_procesador():
    cpu = platform.processor() or "No identificado por platform"

    fisicos = psutil.cpu_count(logical=False)
    logicos = psutil.cpu_count(logical=True)
    disponibles = cpu_count()

    memoria = psutil.virtual_memory()

    recomendados = max(
        1,
        min(8, fisicos or disponibles, disponibles)
    )

    print(f"Procesador: {cpu}")
    print(f"CPU físicos: {fisicos}")
    print(f"CPU lógicos: {logicos}")
    print(f"CPU disponibles para este proceso: {disponibles}")

    print(f"RAM total: {memoria.total / 1024**3:.2f} GiB")
    print(f"RAM disponible: {memoria.available / 1024**3:.2f} GiB")

    print(f"\nWorkers sugeridos: {recomendados}")

    return recomendados


WORKERS_SUGERIDOS = None


# ============================================================
# 3. CONFIGURACIÓN
# ============================================================

DATASET_PATH = config.svm_dataset
EMBEDDING_PATHS = config.svm_embeddings
CACHE_DIR = config.SVM_CACHE
CHECKPOINT_PATH = config.SVM_CHECKPOINT
CSV_PATH = config.SVM_SUMMARY
ARTIFACTS_DIR = config.SVM_ARTIFACTS

LABEL_COLUMN = "label"

TRAIN_SIZE = 0.80
SPLIT_RANDOM_STATE = 102

CV_SPLITS = 5
CV_RANDOM_STATE = 42

MAX_ITER = 5000

N_JOBS = 4
RESUME = True

# ============================================================
# 4. VALIDACIÓN DE ARCHIVOS Y ETIQUETAS
# ============================================================

def validar_archivos():
    requeridos = [
        DATASET_PATH,
        *EMBEDDING_PATHS.values()
    ]

    faltantes = [
        str(path)
        for path in requeridos
        if not path.exists()
    ]

    if faltantes:
        raise FileNotFoundError(
            "No se encontraron estos archivos:\n"
            + "\n".join(faltantes)
        )

    print(f"Archivos verificados: {len(requeridos)}")


y = None


# ============================================================
# 5. INSPECCIÓN Y CARGA DE EMBEDDINGS
# ============================================================

def inspeccionar_embedding(nombre, matriz):
    memoria = (
        matriz.data.nbytes + matriz.indices.nbytes + matriz.indptr.nbytes
        if issparse(matriz)
        else matriz.nbytes
    )

    nnz = (
        matriz.nnz
        if hasattr(matriz, "nnz")
        else np.count_nonzero(matriz)
    )

    print(
        f"{nombre}: "
        f"shape={matriz.shape}, "
        f"nnz={nnz:,}, "
        f"dtype={matriz.dtype}, "
        f"memoria={memoria / 1024**2:.1f} MiB"
    )


def cargar_embeddings(
    rutas=EMBEDDING_PATHS,
    cantidad_muestras=None
):
    embeddings = {}

    for nombre, ruta in rutas.items():
        print(f"\nCargando {nombre}...")

        matriz = load_npz(ruta)

        if not issparse(matriz):
            raise TypeError(f"{nombre}: se esperaba una matriz dispersa SciPy.")

        # LinearSVC/liblinear opera en doble precisión. Normalizar aquí a CSR
        # evita conversiones repetidas dentro de cada fold y cada candidato.
        matriz = matriz.tocsr().astype(np.float64, copy=False)
        matriz.sum_duplicates()
        matriz.eliminate_zeros()
        matriz.sort_indices()

        if (
            cantidad_muestras is not None
            and matriz.shape[0] != cantidad_muestras
        ):
            raise ValueError(
                f"{nombre}: {matriz.shape[0]} filas, "
                f"pero el dataset tiene {cantidad_muestras} muestras."
            )

        inspeccionar_embedding(nombre, matriz)

        embeddings[nombre] = matriz

    return embeddings


embeddings = None


# ============================================================
# 6. ESPACIO DE BÚSQUEDA
# ============================================================

C_VALUES = [0.01, 0.1, 1, 10]

HIPERPARAMETROS = [
    {
        "linearsvc__C": C_VALUES,
        "linearsvc__penalty": ["l2"],
        "linearsvc__loss": ["squared_hinge"],
    },
    {
        "linearsvc__C": C_VALUES,
        "linearsvc__penalty": ["l2"],
        "linearsvc__loss": ["hinge"],
    },
    {
        "linearsvc__C": C_VALUES,
        "linearsvc__penalty": ["l1"],
        "linearsvc__loss": ["squared_hinge"],
    },
]

CV = StratifiedKFold(
    n_splits=CV_SPLITS,
    shuffle=True,
    random_state=CV_RANDOM_STATE,
)

# ============================================================
# 7. PIPELINES
# ============================================================

memory = Memory(
    location=str(CACHE_DIR),
    verbose=0
)


def crear_clasificadores(memory_cache=memory):

    sin_balancear = make_pipeline(
        LinearSVC(
            max_iter=MAX_ITER,
            random_state=42,
            class_weight=None,
            dual="auto",
        ),
        memory=memory_cache,
    )

    balanceado = make_pipeline(
        StandardScaler(with_mean=False),
        LinearSVC(
            max_iter=MAX_ITER,
            random_state=42,
            class_weight="balanced",
            dual="auto",
        ),
        memory=memory_cache,
    )

    return {
        "sin balancear": sin_balancear,
        "balanceado": balanceado,
    }


CLASIFICADORES = crear_clasificadores()


# ============================================================
# 8. UTILIDADES
# ============================================================

def format_time(seconds):
    seconds = max(0, int(seconds))

    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    return f"{minutes:02d}:{seconds:02d}"


@contextmanager
def tqdm_joblib(progress_bar):

    callback_original = (
        joblib.parallel.BatchCompletionCallBack
    )

    class TqdmBatchCompletionCallback(callback_original):

        def __call__(self, *args, **kwargs):
            progress_bar.update(n=self.batch_size)

            return super().__call__(
                *args,
                **kwargs
            )

    joblib.parallel.BatchCompletionCallBack = (
        TqdmBatchCompletionCallback
    )

    try:
        yield progress_bar

    finally:
        joblib.parallel.BatchCompletionCallBack = (
            callback_original
        )

        progress_bar.close()

# ============================================================
# PERSISTENCIA DE ARTEFACTOS
# ============================================================

def slugify(value):
    """
    Convierte un nombre en un identificador seguro
    para utilizarlo como nombre de directorio.
    """

    value = value.lower().strip()
    value = value.replace(" ", "_")
    value = value.replace("-", "_")

    value = re.sub(
        r"[^a-z0-9_]",
        "",
        value
    )

    return value


def obtener_directorio_artefactos(
    nombre_embedding,
    nombre_variante,
):
    directorio = (
        ARTIFACTS_DIR
        / slugify(nombre_embedding)
        / slugify(nombre_variante)
    )

    directorio.mkdir(
        parents=True,
        exist_ok=True
    )

    return directorio


def guardar_json_atomico(data, ruta):
    """
    Evita dejar un JSON incompleto si el proceso
    se interrumpe durante la escritura.
    """

    ruta.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temporal = ruta.with_suffix(".tmp")

    with temporal.open(
        "w",
        encoding="utf-8"
    ) as archivo:

        json.dump(
            data,
            archivo,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    temporal.replace(ruta)


def guardar_modelo_atomico(modelo, ruta):
    """
    Persiste el pipeline completo mediante Joblib.
    """

    ruta.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temporal = ruta.with_name(
        ruta.stem + ".tmp" + ruta.suffix
    )

    joblib.dump(
        modelo,
        temporal,
        compress=3,
    )

    temporal.replace(ruta)


def artefactos_completos(directorio):
    """
    Determina si ya existe un artefacto inferible.
    """

    requeridos = [
        directorio / "model.joblib",
        directorio / "metadata.json",
        directorio / "evaluation.json",
        directorio / "evaluation.csv",
        directorio / "confusion_matrix.csv",
        directorio / "confusion_matrix.png",
        directorio / "test_predictions.csv",
    ]

    return all(
        ruta.exists()
        for ruta in requeridos
    )

# ============================================================
# 9. FIRMA DE CONFIGURACIÓN
# ============================================================

def construir_firma_configuracion():

    archivos = {}

    for ruta in [
        DATASET_PATH,
        *EMBEDDING_PATHS.values()
    ]:
        stat = ruta.stat()

        archivos[str(ruta.resolve())] = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }

    config = {
        "version": 2,
        "files": archivos,
        "label_column": LABEL_COLUMN,
        "train_size": TRAIN_SIZE,
        "split_random_state": SPLIT_RANDOM_STATE,
        "cv_splits": CV_SPLITS,
        "cv_random_state": CV_RANDOM_STATE,
        "max_iter": MAX_ITER,
        "parameter_grid": HIPERPARAMETROS,
        "scoring": "f1_macro",
        "input_dtype": "float64",
        "input_format": "csr",
    }

    serializado = json.dumps(
        config,
        sort_keys=True,
        default=str
    )

    return hashlib.sha256(
        serializado.encode("utf-8")
    ).hexdigest()


# ============================================================
# 10. CHECKPOINTS
# ============================================================

def cargar_checkpoint():

    firma = construir_firma_configuracion()

    if not RESUME or not CHECKPOINT_PATH.exists():
        return {
            "signature": firma,
            "experiments": {}
        }

    with CHECKPOINT_PATH.open(
        "r",
        encoding="utf-8"
    ) as archivo:

        checkpoint = json.load(archivo)

    if checkpoint.get("signature") != firma:
        raise ValueError(
            "El checkpoint pertenece a otra configuración "
            "o a otros archivos. Renombrá/eliminá el checkpoint "
            "si querés iniciar una búsqueda nueva."
        )

    print(
        "\nCheckpoint encontrado:",
        len(checkpoint.get("experiments", {})),
        "experimentos guardados."
    )

    return checkpoint


def guardar_checkpoint(checkpoint):

    temporal = CHECKPOINT_PATH.with_suffix(".tmp")

    with temporal.open(
        "w",
        encoding="utf-8"
    ) as archivo:

        json.dump(
            checkpoint,
            archivo,
            ensure_ascii=False,
            indent=2,
            default=str
        )

    temporal.replace(CHECKPOINT_PATH)

    filas = []

    for clave, resultado in checkpoint["experiments"].items():

        filas.append({
            "experimento": clave,
            "embedding": resultado["embedding"],
            "variante": resultado["variante"],
            "best_f1_macro_cv": resultado["best_f1_macro_cv"],
            "best_params": json.dumps(
                resultado["best_params"],
                ensure_ascii=False
            ),
            "elapsed_seconds": resultado["elapsed_seconds"],
        })

    pd.DataFrame(filas).sort_values(
        "experimento"
    ).to_csv(
        CSV_PATH,
        index=False
    )

    print(f"\nCheckpoint actualizado: {CHECKPOINT_PATH}")


def _fit_busqueda_con_progreso(busqueda, X_train, y_train, progreso, nombre):
    """Muestra actividad continua y avance por cada fit completado."""
    def ajustar():
        with threadpool_limits(limits=1):
            with tqdm_joblib(progreso):
                return busqueda.fit(X_train, y_train)

    return ejecutar_etapa(f"Búsqueda CV: {nombre}", ajustar)


# ============================================================
# 11. OPTIMIZACIÓN Y PERSISTENCIA DE MODELOS
# ============================================================

def optimizar_modelo(
    nombre_embedding,
    embedding,
    y,
    nombre_variante,
    clasificador,
    checkpoint,
    n_jobs=N_JOBS,
):

    clave = f"{nombre_embedding}::{nombre_variante}"

    directorio = obtener_directorio_artefactos(
        nombre_embedding,
        nombre_variante,
    )

    ruta_modelo = directorio / "model.joblib"
    ruta_metadata = directorio / "metadata.json"

    ruta_evaluacion = directorio / "evaluation.json"
    ruta_confusion = directorio / "confusion_matrix.csv"
    ruta_predicciones = directorio / "test_predictions.csv"
    ruta_cv = directorio / "cv_results.csv"

    # --------------------------------------------------------
    # 1. Verificar si ya existe un artefacto válido
    # --------------------------------------------------------

    resultado_previo = checkpoint["experiments"].get(clave)

    if artefactos_completos(directorio):

        try:
            with ruta_metadata.open(
                "r",
                encoding="utf-8",
            ) as archivo:

                metadata_existente = json.load(archivo)

            firma_valida = (
                metadata_existente.get("configuration_signature")
                == checkpoint["signature"]
            )

        except (OSError, json.JSONDecodeError):
            firma_valida = False

        if firma_valida and resultado_previo is not None:

            print(
                f"\n[OMITIDO] {clave}: "
                "los artefactos ya existen y corresponden "
                "a la configuración actual."
            )

            return resultado_previo

        print(
            f"\n[REGENERANDO] {clave}: "
            "los artefactos no corresponden a la configuración actual."
        )

    # --------------------------------------------------------
    # 2. Preparar la separación train/test
    # --------------------------------------------------------

    inicio = time.perf_counter()

    print("\n" + "=" * 72)
    print(f"Procesando {nombre_embedding} — {nombre_variante}")
    print("=" * 72)

    indices = np.arange(len(y))

    indices_train, indices_test = train_test_split(
        indices,
        train_size=TRAIN_SIZE,
        shuffle=True,
        stratify=y,
        random_state=SPLIT_RANDOM_STATE,
    )

    X_train = embedding[indices_train]
    X_test = embedding[indices_test]

    y = pd.Series(y, name=LABEL_COLUMN).reset_index(drop=True)
    y_train = y.iloc[indices_train]
    y_test = y.iloc[indices_test]

    print(
        f"Train: {X_train.shape[0]:,} muestras | "
        f"Test: {X_test.shape[0]:,} muestras"
    )

    print(
        f"Features: {X_train.shape[1]:,} | "
        f"Formato: {type(embedding).__name__}"
    )

    # --------------------------------------------------------
    # 3. Recuperar o ejecutar la búsqueda
    # --------------------------------------------------------

    resultado_previo = checkpoint["experiments"].get(clave)

    busqueda = None

    if resultado_previo is not None:

        # Ya se optimizó anteriormente.
        # Se reutilizan los mejores hiperparámetros.

        print(
            "\n[RECUPERADO] Se reutilizan los mejores "
            "hiperparámetros del checkpoint."
        )

        best_params = resultado_previo["best_params"]
        best_score = resultado_previo["best_f1_macro_cv"]

        duracion_busqueda = resultado_previo["elapsed_seconds"]

    else:

        # Todavía no existe una búsqueda para este experimento.

        candidatos = len(ParameterGrid(HIPERPARAMETROS))
        folds = CV.get_n_splits(X_train, y_train)

        total_fits = candidatos * folds

        print(
            f"Candidatos: {candidatos} | "
            f"Folds: {folds} | "
            f"Entrenamientos: {total_fits} | "
            f"Workers: {n_jobs}",
            flush=True,
        )

        progreso = tqdm(
            total=total_fits,
            desc=f"{nombre_embedding} | {nombre_variante}",
            unit="fit",
            dynamic_ncols=True,
            smoothing=0.1,
        )
        inicio_busqueda = time.perf_counter()

        busqueda = ejecutar_grid_search(
            X_train,
            y_train,
            clasificador,
            HIPERPARAMETROS,
            CV,
            n_jobs=n_jobs,
            pre_dispatch=n_jobs,
            verbose=0,
            fit_callback=lambda search, X, labels: _fit_busqueda_con_progreso(
                search, X, labels, progreso, f"{nombre_embedding} | {nombre_variante}"
            ),
        )

        duracion_busqueda = (
            time.perf_counter() - inicio_busqueda
        )

        best_params = busqueda.best_params_
        best_score = float(busqueda.best_score_)

        # ----------------------------------------------------
        # 3.1. Persistir resultados de todos los candidatos
        # ----------------------------------------------------

        resultados_cv = pd.DataFrame(
            busqueda.cv_results_
        )

        # Los parámetros se convierten a JSON para facilitar
        # su lectura desde otras herramientas.

        resultados_cv["params"] = resultados_cv[
            "params"
        ].apply(
            lambda params: json.dumps(
                params,
                ensure_ascii=False,
                default=str,
            )
        )

        resultados_cv.to_csv(
            ruta_cv,
            index=False,
        )

        # ----------------------------------------------------
        # 3.2. Actualizar checkpoint
        # ----------------------------------------------------

        resultado_previo = {
            "embedding": nombre_embedding,
            "variante": nombre_variante,
            "best_params": best_params,
            "best_f1_macro_cv": best_score,
            "elapsed_seconds": float(duracion_busqueda),
        }

        checkpoint["experiments"][clave] = resultado_previo

        guardar_checkpoint(checkpoint)

        print(
            f"\nMejor F1-macro CV: {best_score:.5f}"
        )

        print(
            f"Mejores parámetros: {best_params}"
        )

    # --------------------------------------------------------
    # 4. Entrenar el modelo definitivo
    # --------------------------------------------------------

    print("\nEntrenando clasificador definitivo...")

    modelo_final = clone(clasificador)

    modelo_final.set_params(
        **best_params
    )

    def ajustar_modelo_final():
        with threadpool_limits(limits=1):
            return modelo_final.fit(X_train, y_train)

    ejecutar_etapa(
        f"Entrenamiento final: {nombre_embedding} | {nombre_variante}",
        ajustar_modelo_final,
    )

    # Se elimina la referencia al caché del pipeline.
    # El modelo ya está entrenado y no necesita conservar
    # el directorio de caché para realizar inferencias.

    modelo_final.memory = None

    # --------------------------------------------------------
    # 5. Evaluar sobre el conjunto de test
    # --------------------------------------------------------

    print("Evaluando sobre test...")

    y_pred = modelo_final.predict(X_test)

    clases = modelo_final.classes_.tolist()

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        y_pred,
    )

    precision_macro = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    recall_macro = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    f1_macro = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    reporte = classification_report(
        y_test,
        y_pred,
        labels=clases,
        output_dict=True,
        zero_division=0,
    )

    # --------------------------------------------------------
    # 6. Persistir evaluación
    # --------------------------------------------------------

    evaluacion = {
        "dataset": {
            "total_samples": int(len(y)),
            "train_samples": int(len(y_train)),
            "test_samples": int(len(y_test)),
            "train_size": TRAIN_SIZE,
            "split_random_state": SPLIT_RANDOM_STATE,
        },

        "metrics": {
            "accuracy": float(accuracy),
            "balanced_accuracy": float(balanced_accuracy),
            "precision_macro": float(precision_macro),
            "recall_macro": float(recall_macro),
            "f1_macro": float(f1_macro),
            "f1_macro_cv": float(best_score),
        },

        "classification_report": reporte,
    }

    guardar_json_atomico(
        evaluacion,
        ruta_evaluacion,
    )
    pd.DataFrame([{
        **evaluacion["dataset"],
        **evaluacion["metrics"],
    }]).to_csv(directorio / "evaluation.csv", index=False)

    # --------------------------------------------------------
    # 7. Persistir matriz de confusión
    # --------------------------------------------------------

    matriz_confusion = confusion_matrix(
        y_test,
        y_pred,
        labels=clases,
    )

    df_confusion = pd.DataFrame(
        matriz_confusion,
        index=clases,
        columns=clases,
    )

    df_confusion.index.name = "real"

    df_confusion.to_csv(
        ruta_confusion,
        index=True,
    )

    ruta_matriz_png = directorio / "confusion_matrix.png"
    ejecutar_etapa(
        f"Graficando matriz de confusión: {nombre_embedding} | {nombre_variante}",
        graficar_matrices_confusion,
        [matriz_confusion],
        [f"{nombre_embedding} — {nombre_variante}"],
        clases,
        titulo="Matriz de confusión SVM",
        save_path=ruta_matriz_png,
        show=False,
    )

    # --------------------------------------------------------
    # 8. Persistir predicciones de test
    # --------------------------------------------------------

    df_predicciones = pd.DataFrame({
        "original_index": indices_test,
        "y_true": y_test,
        "y_pred": y_pred,
        "correct": y_test.to_numpy() == y_pred,
    })

    df_predicciones.to_csv(
        ruta_predicciones,
        index=False,
    )

    # --------------------------------------------------------
    # 9. Persistir pipeline completo
    # --------------------------------------------------------

    guardar_modelo_atomico(
        modelo_final,
        ruta_modelo,
    )

    # --------------------------------------------------------
    # 10. Metadata del artefacto
    # --------------------------------------------------------

    es_sparse = hasattr(embedding, "tocsr")

    metadata = {
        "artifact_version": 1,
        "configuration_signature": checkpoint["signature"],

        "model": {
            "type": "sklearn.pipeline.Pipeline",
            "estimator": "LinearSVC",
            "embedding": nombre_embedding,
            "variant": nombre_variante,
            "best_params": best_params,
        },

        "input": {
            "n_features": int(embedding.shape[1]),
            "dtype": str(embedding.dtype),
            "sparse": bool(es_sparse),
            "format": (
                embedding.getformat()
                if es_sparse
                else "dense"
            ),
        },

        "output": {
            "classes": clases,
            "prediction_method": "predict",
            "score_method": "decision_function",
            "probability_output": False,
        },

        "training": {
            "total_samples": int(len(y)),
            "train_samples": int(len(y_train)),
            "test_samples": int(len(y_test)),
            "train_size": TRAIN_SIZE,
            "split_random_state": SPLIT_RANDOM_STATE,
            "cv_splits": CV_SPLITS,
            "cv_random_state": CV_RANDOM_STATE,
            "scoring": "f1_macro",
            "max_iter": MAX_ITER,
            "best_f1_macro_cv": float(best_score),
            "elapsed_search_seconds": float(duracion_busqueda),
        },

        "evaluation": {
            "accuracy": float(accuracy),
            "balanced_accuracy": float(balanced_accuracy),
            "precision_macro": float(precision_macro),
            "recall_macro": float(recall_macro),
            "f1_macro": float(f1_macro),
        },

        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },

        "files": {
            "model": "model.joblib",
            "evaluation": "evaluation.json",
            "evaluation_table": "evaluation.csv",
            "confusion_matrix": "confusion_matrix.csv",
            "confusion_matrix_image": "confusion_matrix.png",
            "test_predictions": "test_predictions.csv",
            "cv_results": (
                "cv_results.csv"
                if ruta_cv.exists()
                else None
            ),
        },
    }

    guardar_json_atomico(
        metadata,
        ruta_metadata,
    )

    # --------------------------------------------------------
    # 11. Finalización
    # --------------------------------------------------------

    duracion_total = time.perf_counter() - inicio

    print("\n" + "-" * 72)
    print(f"Artefactos generados: {clave}")
    print(f"F1-macro CV: {best_score:.5f}")
    print(f"F1-macro TEST: {f1_macro:.5f}")
    print(f"Accuracy TEST: {accuracy:.5f}")
    print(f"Tiempo total: {format_time(duracion_total)}")
    print(f"Directorio: {directorio.resolve()}")
    print("-" * 72, flush=True)

    # Liberar memoria de los datos y del modelo.
    del indices_train, indices_test
    del indices
    del df_predicciones
    del df_confusion
    del matriz_confusion
    del reporte
    del evaluacion
    del metadata

    gc.collect()

    if busqueda is not None:
        del busqueda

    gc.collect()

    return resultado_previo


# ============================================================
# 12. EJECUCIÓN PRINCIPAL
# ============================================================

def main(n_jobs=N_JOBS, argv=None):

    global y, embeddings

    parser = argparse.ArgumentParser(
        description="Optimiza y evalúa clasificadores LinearSVC."
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=n_jobs,
        help="Procesos paralelos para la búsqueda (por defecto, hasta 8 según CPU disponible).",
    )
    args = parser.parse_args(argv)
    n_jobs = args.jobs

    if n_jobs is None:
        n_jobs = diagnosticar_procesador()

    if n_jobs < 1:
        parser.error("--jobs debe ser un entero positivo.")

    inicio_total = time.perf_counter()

    ejecutar_etapa("Validando archivos de entrada", validar_archivos)
    y = ejecutar_etapa("Cargando etiquetas", load_sentiment_labels, DATASET_PATH)
    print(f"\nTotal de muestras: {len(y):,}")
    embeddings = ejecutar_etapa(
        "Cargando representaciones", cargar_embeddings,
        EMBEDDING_PATHS, len(y),
    )
    checkpoint = cargar_checkpoint()

    total_experimentos = (
        len(embeddings) * len(CLASIFICADORES)
    )

    completados_iniciales = len(
        checkpoint["experiments"]
    )

    print("\nINICIANDO OPTIMIZACIÓN DE HIPERPARÁMETROS")
    print(f"Configuración: {n_jobs} procesos paralelos")
    print("Estrategia: GridSearchCV + StratifiedKFold")
    print("Métrica: F1-macro")
    print(f"Candidatos por experimento: {len(ParameterGrid(HIPERPARAMETROS))}")
    print(f"Fits CV por experimento: {len(ParameterGrid(HIPERPARAMETROS)) * CV_SPLITS}")

    print(
        "Experimentos completados previamente: "
        f"{completados_iniciales}/{total_experimentos}"
    )

    for nombre_embedding, matriz in embeddings.items():

        for nombre_variante, clasificador in CLASIFICADORES.items():

            optimizar_modelo(
                nombre_embedding=nombre_embedding,
                embedding=matriz,
                y=y,
                nombre_variante=nombre_variante,
                clasificador=clasificador,
                checkpoint=checkpoint,
                n_jobs=n_jobs,
            )

    duracion_total = time.perf_counter() - inicio_total

    print("\n" + "=" * 72)
    print("OPTIMIZACIÓN FINALIZADA")

    print(
        f"Experimentos guardados: "
        f"{len(checkpoint['experiments'])}/{total_experimentos}"
    )

    print(
        f"Tiempo de esta ejecución: "
        f"{format_time(duracion_total)}"
    )

    print(f"JSON: {CHECKPOINT_PATH}")
    print(f"CSV: {CSV_PATH}")

    print("=" * 72)

    return checkpoint


# ============================================================
# 13. CONSULTAR RESULTADOS
# ============================================================

def mostrar_resultados(ruta=CSV_PATH):

    if not ruta.exists():
        print("Todavía no hay resultados guardados.")
        return pd.DataFrame()

    df = pd.read_csv(ruta)

    if df.empty:
        print("El archivo de resultados está vacío.")
        return df

    return (
        df.sort_values(
            "best_f1_macro_cv",
            ascending=False
        )
        .reset_index(drop=True)
    )


# ============================================================
# 14. ENTRY POINT
# ============================================================

if __name__ == "__main__":

    resultados = main()

    tabla_resultados = mostrar_resultados()

    print("\nRESULTADOS ORDENADOS POR F1-MACRO")
    print(tabla_resultados.to_string(index=False))
