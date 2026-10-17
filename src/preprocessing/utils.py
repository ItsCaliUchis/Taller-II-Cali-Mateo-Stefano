from time import perf_counter
import config
from nltk.corpus import stopwords

def ejecutar_etapa(nombre, funcion, *args, **kwargs):
    inicio = perf_counter()
    print(f"\n[{nombre}] Iniciando...", flush=True)

    resultado = funcion(*args, **kwargs)

    elapsed = perf_counter() - inicio
    print(f"[{nombre}] Finalizado en {elapsed:.2f} segundos.", flush=True)

    return resultado

def extractor_reviews(df):
    negaciones = set(config.cargar_negaciones())
    stop_words = set(stopwords.words("english")) - negaciones

    vec_reviews = []

    for review in df["reviews"]:
        review_split = review.lower().split()

        review_split = [
            word for word in review_split
            if word not in stop_words
        ]

        review_normalize = " ".join(review_split)
        vec_reviews.append(review_normalize)

    return vec_reviews