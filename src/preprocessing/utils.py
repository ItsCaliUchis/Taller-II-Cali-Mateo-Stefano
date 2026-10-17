import config
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize as wt
from itertools import cycle
from threading import Event, Thread
from time import perf_counter

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
            if word not in stop_words
        ]

        review_normalize = " ".join(review_split)
        vec_reviews.append(review_normalize)

    return vec_reviews