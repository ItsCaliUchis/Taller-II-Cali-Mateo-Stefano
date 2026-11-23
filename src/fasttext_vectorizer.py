
import numpy as np
from nltk.tokenize import word_tokenize as wt


class FastTextVectorizer:

    def __init__(self, modelo):
        self.modelo = modelo

    def transform(self, reviews):

        matriz = np.zeros(
            (len(reviews), self.modelo.vector_size),
            dtype=np.float32
        )

        for i, review in enumerate(reviews):

            tokens = wt(review)

            if not tokens:
                continue

            suma_vectores = np.zeros(
                self.modelo.vector_size,
                dtype=np.float32
            )

            for palabra in tokens:
                try:
                    vector = self.modelo.get_vector(palabra)
                except (KeyError, ValueError):
                    vector = np.zeros(
                        self.modelo.vector_size,
                        dtype=np.float32
                    )

                suma_vectores += vector

            matriz[i] = suma_vectores / len(tokens)

        return matriz
