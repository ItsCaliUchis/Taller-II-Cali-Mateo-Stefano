from pathlib import Path
import config 
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize as wt
from scipy.sparse import save_npz
from .utils import extractor_reviews, ejecutar_etapa, obtener_limit, guardar_vocabulario, verificar_recursos_nltk
from transformers import BertTokenizer, BertModel
import tensorflow
import torch

def load_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return tokenizer

def load_bert():
    model = BertModel.from_pretrained("bert-base-uncased")
    return model

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

    reviews = ejecutar_etapa(   
        "extraccion de reseñas",
        extractor_reviews,
        df
    )

    sentences = [
    wt(review) for review in reviews
    ]

    vocabulario = sorted(set(
    token
    for review in sentences
    for token in review
    ))

    print(f"\nLa cardinalidad del vocabulario es de {len(vocabulario)}")

    ejecutar_etapa(
        "Exportación del vocabulario CSV",
        guardar_vocabulario,
        vocabulario,
        config.vocabulario_bert
    )

    tokenizer = ejecutar_etapa(
        "Se carga el tokenizador",
        load_tokenizer
    )

    inputs = [tokenizer(review, return_tensors='pt') for review in reviews]

    modelo = ejecutar_etapa(
        "Se carga el modelo BERT",
        load_bert
    )

    def embedding(inputs):
        modelo.eval()
        embeddings = []

        with torch.no_grad():
            for item in inputs:
                outputs = modelo(**item)
                vector = outputs.last_hidden_state[:, 0, :]
                embeddings.append(vector.squeeze(0).cpu().numpy())

        embeddings = np.asarray(embeddings, dtype=np.float32)

        np.savez_compressed(
            config.bert_vect,
            embeddings=embeddings
        )

        return embeddings

    vect = ejecutar_etapa(
        "Generación de embeddings",
        embedding,
        inputs
    )

    if vect is None:
        print("No se pudieron generar los embeddings.")
        return

    try:
        with np.load(config.bert_vect) as data:
            embeddings = data["embeddings"]

        print("Embeddings guardados correctamente.")
        print("Dimensiones:", embeddings.shape)

    except (OSError, KeyError, ValueError) as e:
        print(f"No se pudieron cargar los embeddings: {e}")

    if __name__ == "__main__":
        main()