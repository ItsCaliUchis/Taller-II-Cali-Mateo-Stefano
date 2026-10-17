from pathlib import Path
import re
import config
import pandas as pd
from gensim import downloader
from .utils import extractor_reviews, ejecutar_etapa

print("EL ARCHIVO FASTTEXT SE ESTA EJECUTANDO")

MODELO_FASTTEXT = "fasttext-wiki-news-subwords-300"
RUTA_SALIDA = config.DATA / "fasttext_embeddings.csv"

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
        config.dataset_clean,
        nrows=100
    )

    reviews = ejecutar_etapa(
        "extraccion de reseñas",
        extractor_reviews,
        df
    )

    print(
        "\nse procede a cargar el modelo FastText.\n",
        flush=True
    )

    modelo = ejecutar_etapa(
        "carga del modelo FastText",
        downloader.load,
        MODELO_FASTTEXT
    )

    print("\nse procede a generar los embeddings por palabra.\n", flush=True)
    palabras = set()

    for review in reviews:
        tokens = re.findall(r"\b\w+\b", review.lower())
        palabras.update(tokens)

    palabras = sorted(palabras)

    print(
        f"palabras unicas encontradas: {len(palabras)}",
        flush=True
    )

    resultados= []

    for palabra in palabras:
        if palabra not in modelo.key_to_index:
            print(f"palabra no encontrada en el modelo: {palabra}")
            continue
        
        vector = modelo[palabra]
        fila = {"palabra": palabra}
        fila.update({
            f"dim_{i+1}": float(valor)
            for i, valor in enumerate(vector)
        })

        resultados.append(fila)
    embeddings = ejecutar_etapa(
        "construccion de la tabla de embeddings",
        pd.DataFrame,
        resultados
    )

    ejecutar_etapa(
        "Exportación CSV",
        embeddings.to_csv,
        RUTA_SALIDA,
        index=False
    )

    print(
        f"\nembeddings FastText guardados en: {RUTA_SALIDA}",
        flush=True
    )


if __name__ == "__main__":
    main()