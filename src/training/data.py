"""Carga común de objetivos y matrices usadas por los entrenadores."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

import config


CLASS_NAMES = ("negative", "neutral", "positive")
LABEL_MAPPING_VERSION = "amazon-five-to-three-v1"
FIVE_TO_THREE = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2}


def _label_cache_path(dataset_path, column):
    stat = dataset_path.stat()
    signature = f"{dataset_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{column}|{LABEL_MAPPING_VERSION}"
    cache_id = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:20]
    return config.TRAINING_CACHE / f"sentiment_labels_{cache_id}.npy"


def load_sentiment_labels(dataset_path=None, *, limit=None, as_indices=False):
    """Carga etiquetas ya agrupadas o deriva 3 clases desde la columna label.

    Solo lee la columna de etiquetas. El CSV de origen nunca se modifica y la
    derivación se cachea según tamaño/mtime del dataset y versión del mapeo.
    """
    dataset_path = Path(dataset_path or config.dataset_clean)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"No existe el dataset: {dataset_path}")
    if limit is not None and limit <= 0:
        raise ValueError("limit debe ser un entero positivo.")

    columns = set(pd.read_csv(dataset_path, nrows=0).columns)
    if "sentiment" in columns:
        column = "sentiment"
        grouped = True
    elif "label_3class" in columns:
        column = "label_3class"
        grouped = True
    elif "label" in columns:
        column = "label"
        grouped = False
    else:
        raise ValueError("El dataset debe incluir 'label', 'sentiment' o 'label_3class'.")

    cache_path = _label_cache_path(dataset_path, column)
    labels = None
    if cache_path.is_file():
        try:
            labels = np.load(cache_path, allow_pickle=False)
        except (OSError, ValueError):
            cache_path.unlink(missing_ok=True)

    if labels is None:
        source = pd.read_csv(dataset_path, usecols=[column])[column]
        if grouped:
            normalized = source.astype(str).str.strip().str.lower()
            class_to_index = {name: index for index, name in enumerate(CLASS_NAMES)}
            mapped = normalized.map(class_to_index)
            numeric = pd.to_numeric(source, errors="coerce")
            numeric_classes = numeric.map({0: 0, 1: 1, 2: 2})
            mapped = mapped.fillna(numeric_classes)
        else:
            numeric = pd.to_numeric(source, errors="coerce")
            mapped = numeric.map(FIVE_TO_THREE)

        if mapped.isna().any():
            invalid = source[mapped.isna()].drop_duplicates().tolist()[:10]
            raise ValueError(f"Etiquetas desconocidas en '{column}': {invalid}")
        labels = mapped.to_numpy(dtype=np.int64)
        config.TRAINING_CACHE.mkdir(parents=True, exist_ok=True)
        temporary_path = cache_path.with_suffix(".tmp")
        with temporary_path.open("wb") as cache_file:
            np.save(cache_file, labels, allow_pickle=False)
        temporary_path.replace(cache_path)

    if limit is not None:
        if limit > len(labels):
            raise ValueError(f"--limit {limit} supera las {len(labels)} etiquetas disponibles.")
        labels = labels[:limit]

    if as_indices:
        return labels.astype(np.int64, copy=False)
    return np.asarray(CLASS_NAMES, dtype=str)[labels]


def load_dense_embeddings(name, *, limit=None):
    """Carga un NPZ denso de BERT o ELMo y recorta su prefijo opcional."""
    paths = {"bert": config.bert_vect, "elmo": config.elmo_vect}
    if name not in paths:
        raise ValueError(f"Representación no compatible: {name}. Use bert o elmo.")
    path = config.ruta_limitada(paths[name], limit)
    if not path.is_file():
        suffix = f" con limit={limit}" if limit is not None else ""
        raise FileNotFoundError(f"No existe la matriz {path.name}{suffix}; ejecute primero el embedding {name}.")

    with np.load(path, allow_pickle=False) as archive:
        if "embeddings" not in archive:
            raise ValueError(f"{path.name} no contiene el arreglo 'embeddings'.")
        matrix = archive["embeddings"]
    if matrix.ndim != 2 or not np.issubdtype(matrix.dtype, np.number):
        raise ValueError(f"La matriz {path.name} debe ser bidimensional y numérica.")
    if limit is not None and matrix.shape[0] != limit:
        raise ValueError(f"{path.name} tiene {matrix.shape[0]} filas; se esperaban {limit}.")
    if not np.isfinite(matrix).all():
        raise ValueError(f"La matriz {path.name} contiene valores NaN o infinitos.")
    return np.asarray(matrix, dtype=np.float32), path
