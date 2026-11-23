from pathlib import Path
import argparse
import config
import pandas as pd
import numpy as np
import joblib
from fasttext_vectorizer import FastTextVectorizer

from gensim import downloader
from nltk.tokenize import word_tokenize as wt
from scipy.sparse import csr_matrix, save_npz
from utils import extractor_reviews, ejecutar_etapa, guardar_vocabulario, _ascii_console_text

MODELO_FASTTEXT = config.MODELO_FASTTEXT
RUTA_SALIDA = config.fasttext_embeddings


def obtener_limit():
    parser = argparse.ArgumentParser(
        description="generacion de embeddings por resena con FastText"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=config.FASTTEXT_NROWS,
        help="cantidad maxima de resenas a procesar (todas)"
    )

    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit debe ser un entero positivo.")

    return args.limit


def obtener_ruta_salida(output_path, limit):
    return config.ruta_limitada(output_path, limit)


def main():
    limit = obtener_limit()
    ruta_salida = obtener_ruta_salida(RUTA_SALIDA, limit)

    if not Path(config.dataset_clean).exists():
        print(
            f'no existe el dataset "{Path(config.dataset_clean).name}". '
            'ejecute primero los comandos "extractor" y "clean".'
        )
        return 1

    print("\nse esta cargando el dataset.\n", flush=True)

    df = ejecutar_etapa(
        "carga del dataset",
        pd.read_csv,
        config.dataset_clean,
        nrows=limit
    )

    print(
        f"resenas cargadas: {len(df)}",
        flush=True
    )

    reviews = ejecutar_etapa(
        "extraccion de resenas",
        extractor_reviews,
        df
    )
    
    if len(reviews) != len(df):
        raise ValueError(
            "la cantidad de resenas procesadas no coincide "
            "con la cantidad de filas "
        )

    #generacion de los tokens por reseña
    sentences = [
        wt(review) for review in reviews
    ]

    #generacion del vocabulario
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

    vectorizador = FastTextVectorizer(modelo) # ya contiene representaciones del modelo (está preentrenado)

    print(
        "\nSe procede a generar los embeddings por palabra.\n",
        flush=True
    )

    """ Generador de un vector por reseña
    """
    matriz_reseñas = vectorizador.transform(reviews) 
    matriz_embeddings = csr_matrix(matriz_reseñas)

    """ Almacenamiento de las representaciones de las reviews
    """
    vectorizador.matriz_reseñas_ = matriz_embeddings
    vectorizador.vocabulario_ = vocabulario

    joblib.dump(vectorizador, config.fasttext_vectorizer)

    #exportacion del vocabulario
    ejecutar_etapa(
        "exportacion del vocabulario CSV",
        guardar_vocabulario,
        vocabulario,
        config.ruta_limitada(config.fasttext_vocab, limit)
    )

    #exportacion de la matriz de embeddings por reseña
    ejecutar_etapa(
        "exportacion de la matriz dispersa npz",
        save_npz,
        ruta_salida,
        matriz_embeddings
    )

    print(
        f"\nembeddings FastText guardados en: {ruta_salida.name}",
        flush=True
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
