from pathlib import Path
import argparse
import config
import pandas as pd
import numpy as np

from gensim import downloader
from nltk.tokenize import word_tokenize as wt
from scipy.sparse import csr_matrix, save_npz
from utils import extractor_reviews, ejecutar_etapa, guardar_vocabulario

MODELO_FASTTEXT = config.MODELO_FASTTEXT
RUTA_SALIDA = config.fasttext_embeddings


def obtener_limit():
    parser = argparse.ArgumentParser(
        description="generacion de embeddings de palabras con FastText"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=config.FASTTEXT_NROWS,
        help="cantidad máxima de reseñas a procesar (todas)"
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


def main():
    limit = obtener_limit()
    ruta_salida = obtener_ruta_salida(RUTA_SALIDA, limit)

    if not Path(config.dataset_clean).exists():
        print(
            f'no existe el dataset en la ruta "{config.dataset_clean}". '
            'ejecute primero los comandos "extractor" y "clean".'
        )
        return

    print("\nse esta cargando el dataset.\n", flush=True)

    df = ejecutar_etapa(
        "carga del dataset",
        pd.read_csv,
        config.dataset_clean,
        nrows=limit
    )

    print(
        f"reseñas cargadas: {len(df)}",
        flush=True
    )

    reviews = ejecutar_etapa(
        "extracción de reseñas",
        extractor_reviews,
        df
    )

    #generacion del vocabulario
    sentences = [
        wt(review) for review in reviews
    ]

    vocabulario = sorted(set(
    token
    for review in sentences
    for token in review
    ))

    print("\nSe procede a cargar el modelo FastText.\n", flush=True)

    modelo = ejecutar_etapa(
        "carga del modelo FastText",
        downloader.load,
        MODELO_FASTTEXT
    )

    print(
        "\nSe procede a generar los embeddings por palabra.\n",
        flush=True
    )

    vectores = []

    for palabra in vocabulario:
        try:
            vector = modelo.get_vector(palabra)
        except (KeyError, ValueError):
            print(f"palabra no encontrada en el modelo: {palabra}")
            vector = np.zeros(modelo.vector_size, dtype=np.float32)

        vectores.append(vector)

    #construccion de la matriz dispersa
    if vectores:
        matriz_embeddings = csr_matrix(
            np.asarray(vectores, dtype=np.float32)
        )
    else:
        matriz_embeddings = csr_matrix(
            (0, modelo.vector_size),
            dtype=np.float32
        )

    print(
        f"dimensiones de la matriz: {matriz_embeddings.shape}",
        flush=True
    )

    #exportacion del vocabulario
    ejecutar_etapa(
        "exportación del vocabulario CSV",
        guardar_vocabulario,
        vocabulario,
        config.fasttext_vocab
    )

    #exportacion de la matriz dispersa a npz
    ejecutar_etapa(
        "exportacion de la matriz dispersa npz",
        save_npz,
        ruta_salida,
        matriz_embeddings
    )

    print(
        f"\nembeddings FastText guardados en: {ruta_salida}",
        flush=True
    )
    

if __name__ == "__main__":
    main()