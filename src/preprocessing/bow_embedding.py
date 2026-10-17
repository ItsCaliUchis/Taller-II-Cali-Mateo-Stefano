import config
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer as CV
import numpy as np
from scipy.sparse import csr_matrix
from nltk.corpus import stopwords


def extractor_reviews(df):
    negaciones = config.cargar_negaciones()
    STOPWORDS = list(set((stopwords.words("english")))-set(negaciones))
    vec_reviews = []
    for i, review in df["reviews"].items():
        review_split = review.split()

        review_split = [
            word for word in review_split
            if word not in STOPWORDS
        ]

        review_normalize = " ".join(review_split)
        vec_reviews.append(review_normalize)
    return vec_reviews

def main():
    if Path(config.dataset_clean).exists():
        print('\nSe esta cargando el dataset\n')
        df = pd.read_csv(config.dataset_clean)
        print(f'\nSe ha cargado el dataset con {len(df)} registros, se procede a explorar el dataset\n')
    else:
        print(f'No existe el dataset en la ruta "{config.dataset_clean}", ejecute primero el comando "extractor" para descargar el dataset y luego ejecute este comando para explorar el dataset')
      
    vect = CV()
    matrix = vect.fit_transform(extractor_reviews(df)) # aprende el vocabulario y crea los embeddings

    print(f"El vocabulario aprendido es: {vect.get_feature_names_out()} con tamaño {len(vect.get_feature_names_out())}")

    bow_columns = vect.get_feature_names_out()
    try:
        bow = pd.DataFrame.sparse.from_spmatrix(
            matrix,
            columns=bow_columns) # guarda resultado en dataframe
        bow.to_csv(config.bow_vect, index=False)

    except Exception as e:
        print(f"Tipo: {type(e).__name__}")
        print(f"Mensaje: {e}")


    if __name__ == '__main__':
        main()