import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"


from pathlib import Path
import config 
import numpy as np
import pandas as pd
from scipy.sparse import save_npz
from nltk.tokenize import word_tokenize as wt
from .utils import extractor_reviews, ejecutar_etapa

print("LA VECTORIZACION ELMO SE ESTA EJECUTANDO")

def load_elmo():
    from simple_elmo import ElmoModel
    model = ElmoModel()

    model.load(str(config.elmo_load))
    return model
    
def main():
    if not Path(config.dataset_clean).exists():
        print(
            f'no existe el dataset en la ruta "{config.dataset_clean}". '
            'ejecute primero el comando "extractor" y despues "clean".'
        )
        return
    print("\nse esta cargando el dataset\n", flush=True)

    df = ejecutar_etapa(
        "carga del dataset",
        pd.read_csv,
        config.dataset_clean
    )

    reviews = ejecutar_etapa(               # devuelve lista de reviews como strings
        "extraccion de reseñas",
        extractor_reviews,
        df
    )

    sentences = [
        wt(review) for review in reviews    # se hace tokenizacion para poder usar model.get_elmo_vector_average
    ]

    vocabulario = sorted(set(
    token
    for review in sentences
    for token in review
    ))

    print(f"\nLa cardinalidad del vocabulario es de {len(vocabulario)}")
    
    print(
        "\nse procede a cargar el modelo ELMo.\n",
        flush=True
    )

    modelo = ejecutar_etapa(
        "carga del modelo ELMo",
        load_elmo
    )

    print("\nse procede a generar los embeddings.\n", flush=True)

    def embedding(sentences):
        embeddings = modelo.get_elmo_vector_average(sentences) # embedding por review
        embeddings = np.asarray(embeddings, dtype=np.float32)

        np.savez_compressed(
        config.elmo_vect,
        embeddings = embeddings
        )

        return embeddings

    vect = ejecutar_etapa(
        "Generación de embeddings",
        embedding,
        sentences,
        flush=True
    )

    if vect is None:
        print("No se pudieron generar los embeddings.")
        return

    try:
        with np.load(config.elmo_vect) as data:
            embeddings = data["embeddings"]

        print("Embeddings guardados correctamente.")
        print("Dimensiones:", embeddings.shape)

    except (OSError, KeyError, ValueError) as e:
        print(f"No se pudieron cargar los embeddings: {e}")

if __name__ == "__main__":
    main()
