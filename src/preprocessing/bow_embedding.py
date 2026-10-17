import config
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer as CV
import numpy as np
from scipy.sparse import save_npz
from .utils import extractor_reviews, ejecutar_etapa, obtener_limit, guardar_vocabulario, verificar_recursos_nltk

def main():
    if not Path(config.dataset_clean).exists():
        print(
            f'no existe el dataset en la ruta "{config.dataset_clean}". '
            'ejecute primero el comando "extractor" y despues "clean".'
        )
        return
    verificar_recursos_nltk()

    print("\nse esta cargando el dataset\n", flush=True)

    limit = obtener_limit()

    df = ejecutar_etapa(
        "carga del dataset",
        pd.read_csv,
        config.dataset_clean,
        nrows=limit
    )
    
    vect = CV()
    matrix = vect.fit_transform(extractor_reviews(df)) # aprende el vocabulario y crea los embeddings
    vocab = vect.get_feature_names_out()

    print(f"El vocabulario aprendido es: {vect.get_feature_names_out()} con tamaño {len(vect.get_feature_names_out())}")

    ejecutar_etapa(
        "Exportando vocabulario",
        guardar_vocabulario,
        vocab,
        config.bow_vocab
    )

    bow_columns = vect.get_feature_names_out()

    try:
        save_npz(config.bow_vect, matrix)

    except Exception as e:
        print(f"Tipo: {type(e).__name__}")
        print(f"Mensaje: {e}")


    if __name__ == '__main__':
        main()