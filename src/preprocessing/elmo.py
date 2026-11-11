from pathlib import Path
import argparse
import config 
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize as wt
import requests
from utils import extractor_reviews, ejecutar_etapa, guardar_vocabulario, verificar_recursos_nltk, _ascii_console_text

pesos = "https://vectors.nlpl.eu/repository/20/194.zip"

def load_elmo(*, download=False):
    import os
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
    from simple_elmo import ElmoModel
    model = ElmoModel()

    if not Path(str(config.elmo_load)).exists():
        if not download:
            respuesta = input("Los pesos no se encuentran instalados. Desea instalar? [s/n]: ")
            if respuesta.lower() != "s":
                print("\nPrograma de vectorizacion terminado")
                return

        temporal = Path(config.elmo_load).with_suffix(".download")
        try:
            with requests.get(pesos, stream=True, timeout=(10, 120)) as descarga:
                descarga.raise_for_status()
                with temporal.open("wb") as archivo:
                    for chunk in descarga.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            archivo.write(chunk)
            temporal.replace(config.elmo_load)
        except requests.RequestException as error:
            temporal.unlink(missing_ok=True)
            print(f"\nLa descarga no fue exitosa: {_ascii_console_text(error)}")
            return

        model.load(str(config.elmo_load))
        return model

    else:
        print(f"Los pesos se encuentran instalados en {Path(config.elmo_load).name}")
        model.load(str(config.elmo_load))
    
    return model
    
def main():
    parser = argparse.ArgumentParser(description="Genera embeddings ELMo por resenas.")
    parser.add_argument("--limit", type=int, default=None, help="Cantidad maxima de resenas.")
    parser.add_argument("--download-model", action="store_true", help="Descarga los pesos si no estan instalados, sin preguntar.")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit debe ser un entero positivo")

    if not Path(config.dataset_clean).exists():
        print(
            f'no existe el dataset "{Path(config.dataset_clean).name}". '
            'ejecute primero el comando "extractor" y despues "clean".'
        )
        return 1
    verificar_recursos_nltk()

    print("\nse esta cargando el dataset\n", flush=True)

    limit = args.limit

    df = ejecutar_etapa(
        "carga del dataset",
        pd.read_csv,
        config.dataset_clean,
        nrows=limit
    )

    reviews = ejecutar_etapa(               # devuelve lista de reviews como strings
        "extraccion de resenas",
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

    output_path = config.ruta_limitada(config.elmo_vect, limit)
    vocab_path = config.ruta_limitada(config.vocabulario_elmo, limit)
    ejecutar_etapa(
        "Exportacion del vocabulario CSV",
        guardar_vocabulario,
        vocabulario,
        vocab_path
    )
    
    print(
        "\nse procede a cargar el modelo ELMo.\n",
        flush=True
    )

    modelo = load_elmo(download=args.download_model)

    if modelo is None:
        print("No se pudo cargar el modelo ELMo. Se cancela la vectorizacion.")
        return 1

    print("\nse procede a generar los embeddings.\n", flush=True)

    def embedding(sentences):
        embeddings = modelo.get_elmo_vector_average(sentences) # embedding por review
        embeddings = np.asarray(embeddings, dtype=np.float32)

        np.savez_compressed(
        output_path,
        embeddings = embeddings
        )

        return embeddings

    vect = ejecutar_etapa(
        "Generacion de embeddings",
        embedding,
        sentences
    )

    if vect is None:
        print("No se pudieron generar los embeddings.")
        return 1

    try:
        with np.load(output_path) as data:
            embeddings = data["embeddings"]

        print("Embeddings guardados correctamente.")
        print("Dimensiones:", embeddings.shape)

    except (OSError, KeyError, ValueError) as e:
        print(f"No se pudieron cargar los embeddings: {_ascii_console_text(e)}")
        return 1

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
