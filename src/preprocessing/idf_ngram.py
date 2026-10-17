
import argparse
import csv
from pathlib import Path

import config
import pandas as pd
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

from .utils import extractor_reviews, ejecutar_etapa


def obtener_limit():
    parser = argparse.ArgumentParser(
        description="Generación de matrices TF-IDF de n-gramas."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cantidad máxima de reseñas a procesar. Por defecto, todas."
    )

    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit debe ser un entero positivo.")

    return args.limit


def obtener_ruta_salida(output_path, limit):
    output_path = Path(output_path)

    if limit is None:
        return output_path

    return output_path.with_name(
        f"{output_path.stem}_limit_{limit}{output_path.suffix}"
    )


def obtener_ruta_vocabulario(matrix_path):
    matrix_path = Path(matrix_path)
    return matrix_path.with_name(
        f"{matrix_path.stem}_vocab.csv"
    )


def guardar_vocabulario(vocabulary, output_path):
    with open(output_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["index", "term"])

        for index, term in enumerate(vocabulary):
            writer.writerow([index, term])


def main(
    ngram_range: tuple[int, int],
    output_path: str,
    nombre: str,
    limit: int | None = None
):
    dataset_path = Path(config.dataset_clean)

    if not dataset_path.exists():
        print(
            f'No existe el dataset en la ruta "{dataset_path}". '
            'Ejecute primero el comando "extractor".'
        )
        return

    output_file = obtener_ruta_salida(output_path, limit)

    matrix_file = output_file.with_suffix(".npz")
    vocabulary_file = obtener_ruta_vocabulario(matrix_file)

    print(
        f"\nConfiguración: {nombre} | "
        f"Límite: {limit if limit is not None else 'sin límite'}\n",
        flush=True
    )

    print("\nSe está cargando el dataset\n", flush=True)

    df = ejecutar_etapa(
        "Carga del dataset",
        pd.read_csv,
        dataset_path,
        nrows=limit
    )

    print(
        f"\nSe han cargado {len(df)} registros. "
        f"Se procede a generar los n-gramas TF-IDF ({nombre}).\n",
        flush=True
    )

    reviews = ejecutar_etapa(
        "Extracción de reseñas",
        extractor_reviews,
        df
    )

    vect = TfidfVectorizer(ngram_range=ngram_range)

    matrix = ejecutar_etapa(
        f"Vectorización TF-IDF ({nombre})",
        vect.fit_transform,
        reviews
    )

    vocabulary = vect.get_feature_names_out()

    print(
        f"\nVocabulario aprendido: {len(vocabulary)} términos.",
        flush=True
    )

    try:
        ejecutar_etapa(
            "Exportación de matriz dispersa NPZ",
            save_npz,
            matrix_file,
            matrix
        )

        ejecutar_etapa(
            "Exportación de vocabulario CSV",
            guardar_vocabulario,
            vocabulary,
            vocabulary_file
        )

        print(
            f"\nMatriz dispersa guardada en: {matrix_file}"
            f"\nVocabulario guardado en: {vocabulary_file}",
            flush=True
        )

    except Exception as e:
        print(f"Tipo: {type(e).__name__}")
        print(f"Mensaje: {e}")


def monograma():
    main(
        ngram_range=(1, 1),
        output_path=config.idf_monogram,
        nombre="monogramas",
        limit=obtener_limit()
    )


def bigrama():
    main(
        ngram_range=(2, 2),
        output_path=config.idf_bigram,
        nombre="bigramas",
        limit=obtener_limit()
    )


if __name__ == "__main__":
    bigrama()
