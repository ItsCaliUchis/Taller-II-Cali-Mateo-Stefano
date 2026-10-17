from pathlib import Path

import config
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .utils import extractor_reviews, ejecutar_etapa

def main():
    if not Path(config.dataset_clean).exists():
        print(
            f'No existe el dataset en la ruta "{config.dataset_clean}". '
            'Ejecute primero el comando "extractor".'
        )
        return

    print("\nSe está cargando el dataset\n", flush=True)

    df = ejecutar_etapa(
        "Carga del dataset",
        pd.read_csv,
        config.dataset_clean
    )

    print(
        f"\nSe ha cargado el dataset con {len(df)} registros. "
        "Se procede a generar los bigramas TF-IDF.\n",
        flush=True
    )

    reviews = ejecutar_etapa(
        "Extracción de reseñas",
        extractor_reviews,
        df
    )

    vect = TfidfVectorizer(ngram_range=(2, 2))

    matrix = ejecutar_etapa(
        "Vectorización TF-IDF de bigramas",
        vect.fit_transform,
        reviews
    )

    vocabulary = vect.get_feature_names_out()

    print(
        f"\nVocabulario aprendido: {len(vocabulary)} términos.",
        flush=True
    )

    try:
        tfidf = ejecutar_etapa(
            "Construcción del DataFrame disperso",
            pd.DataFrame.sparse.from_spmatrix,
            matrix,
            columns=vocabulary
        )

        ejecutar_etapa(
            "Exportación CSV",
            tfidf.to_csv,
            config.idf_bigram,
            index=False
        )

        print(
            f"\nMatriz TF-IDF guardada en: {config.idf_bigram}",
            flush=True
        )

    except Exception as e:
        print(f"Tipo: {type(e).__name__}")
        print(f"Mensaje: {e}")


if __name__ == "__main__":
    main()
