import nltk
import config
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize as wt
from itertools import cycle
from threading import Event, Thread
from time import perf_counter
import argparse
import csv
from pathlib import Path
from scipy.sparse import load_npz
import pandas as pd
import subprocess

import psutil


def ejecutar_etapa(nombre, funcion, *args, **kwargs):
    inicio = perf_counter()
    detener = Event()
    proceso = psutil.Process()
    animacion = cycle("|/-\\")

    def monitorizar():
        while not detener.wait(1):
            elapsed = perf_counter() - inicio

            try:
                ram_mb = proceso.memory_info().rss / (1024 ** 2)
                estado_ram = f"RAM: {ram_mb:.0f} MB"
            except psutil.Error:
                estado_ram = "RAM: no disponible"

            print(
                f"\r[{nombre}] {next(animacion)} "
                f"{elapsed:.0f}s | {estado_ram}",
                end="",
                flush=True,
            )

    print(f"\n[{nombre}] Iniciando...", flush=True)

    monitor = Thread(target=monitorizar, daemon=True)
    monitor.start()

    try:
        resultado = funcion(*args, **kwargs)

    except BaseException:
        detener.set()
        monitor.join()

        elapsed = perf_counter() - inicio
        print(
            f"\r[{nombre}] Falló tras {elapsed:.2f}s.",
            flush=True,
        )
        raise

    else:
        detener.set()
        monitor.join()

        elapsed = perf_counter() - inicio
        ram_mb = proceso.memory_info().rss / (1024 ** 2)

        print(
            f"\r[{nombre}] Finalizado en {elapsed:.2f}s "
            f"| RAM final: {ram_mb:.0f} MB",
            flush=True,
        )

        return resultado

def extractor_reviews(df):
    negaciones = set(config.cargar_negaciones())
    stop_words = set(stopwords.words("english")) - negaciones

    vec_reviews = []

    for review in df["reviews"]:
        review_split = wt(review.lower())

        review_split = [
            word for word in review_split
            if word not in stop_words and word.isalpha()
        ]

        review_normalize = " ".join(review_split)
        vec_reviews.append(review_normalize)

    return vec_reviews

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


def verificar_recursos_nltk():
    recursos = {
        "tokenizers/punkt_tab": "punkt_tab",
        "corpora/stopwords": "stopwords",
    }

    for ruta, nombre in recursos.items():
        try:
            nltk.data.find(ruta)
        except LookupError:
            print(f"Descargando recurso NLTK: {nombre}...")
            nltk.download(nombre, quiet=True)

def verificar_embeddings():
    X_bow = None
    X_bigram = None
    X_monogram = None
    X_ft = None

    # BOW ======================
    if Path(config.bow_vect).exists():
        X_bow = load_npz(config.bow_vect)

    if not Path(config.bow_vect).exists():

        print(f'no existen los embeddings de bow en la ruta "{config.bow_vect}"')
        opcion = input("¿Desea cargar los embeddings de bow? [s/n]: ").lower()
        print("\n")

        if opcion == "s":
            subprocess.run(["poetry", "run", "bow"], check=True)
            X_bow = load_npz(config.bow_vect)          
        else:
            print("Se continua el proceso sin los embeddings bow")
            print("\n")

    # IDF BIGRAM =============================

    if Path(config.idf_bigram).exists():
        X_bigram = load_npz(config.idf_bigram)

    if not Path(config.idf_bigram).exists():

        print(f'no existen los embeddings de bow en la ruta "{config.idf_bigram}"')
        opcion = input("¿Desea cargar los embeddings de tf-idf bigrama? [s/n]: ").lower()
        print("\n")

        if opcion == "s":
            subprocess.run(["poetry", "run", "idfbigram"], check=True)
            X_bigram = load_npz(config.idf_bigram)
        else:
            print("Se continua el proceso sin los embeddings de tf-idf bigrama")
            print("\n")

    # IDF MONOGRAM ====================

    if Path(config.idf_monogram).exists():
        X_monogram = load_npz(config.idf_monogram)

    if not Path(config.idf_monogram).exists():

        print(f'no existen los embeddings de bow en la ruta "{config.idf_monogram}"')
        opcion = input("¿Desea cargar los embeddings de tf-idf monogram? [s/n]: ").lower()
        print("\n")

        if opcion == "s":
            subprocess.run(["poetry", "run", "idfmonogram"], check=True)
            X_monogram = load_npz(config.idf_monogram)
        else:
            print("Se continua el proceso sin los embeddings de tf-idf monogram")
            print("\n")


    if Path(config.fasttext_embeddings).exists():
        X_ft = load_npz(config.fasttext_embeddings)

    if not Path(config.fasttext_embeddings).exists():

        print(f'no existen los embeddings de fasttext en la ruta "{config.fasttext_embeddings}"')
        opcion = input("¿Desea cargar los embeddings de fasttext? [s/n]: ").lower()
        print("\n")

        if opcion == "s":
            subprocess.run(["poetry", "run", "fasttext"], check=True)
            X_ft = load_npz(config.fasttext_embeddings)
        else:
            print("Se continua el proceso sin los embeddings de fasttext")
            print("\n")

    return X_bow, X_bigram, X_monogram, X_ft