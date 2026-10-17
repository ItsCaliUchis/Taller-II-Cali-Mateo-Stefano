from pathlib import Path
import re
import argparse
import config
import pandas as pd
from gensim import downloader
from .utils import extractor_reviews, ejecutar_etapa

MODELO_FASTTEXT = config.MODELO_FASTTEXT
RUTA_SALIDA = config.fasttext_embeddings


def obtener_limit():
    parser = argparse.ArgumentParser(
        description="generacion de embeddings de palabras con FastText"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=config.FASTTEXT_NROWS,
        help="cantidad máxima de reseñas a procesar (todas)"
    )

    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit debe ser un entero positivo.")

    return args.limit


def obtener_ruta_salida(output_path, limit):
    output_path = Path(output_path)

    if limit is None:
        return output_path

    return output_path.with_name(
        f"{output_path.stem}_limit_{limit}{output_path.suffix}"
    )


def main():
    limit = obtener_limit()
    ruta_salida = obtener_ruta_salida(RUTA_SALIDA, limit)

    if not Path(config.dataset_clean).exists():
        print(
            f'no existe el dataset en la ruta "{config.dataset_clean}". '
            'ejecute primero los comandos "extractor" y "clean".'
        )
        return

    print("\nse esta cargando el dataset.\n", flush=True)

    df = ejecutar_etapa(
        "carga del dataset",
        pd.read_csv,
        config.dataset_clean,
        nrows=limit
    )

    print(
        f"reseñas cargadas: {len(df)}",
        flush=True
    )

    reviews = ejecutar_etapa(
        "extracción de reseñas",
        extractor_reviews,
        df
    )

    print("\nSe procede a cargar el modelo FastText.\n", flush=True)

    modelo = ejecutar_etapa(
        "carga del modelo FastText",
        downloader.load,
        MODELO_FASTTEXT
    )

    print(
        "\nSe procede a generar los embeddings por palabra.\n",
        flush=True
    )

    palabras = set()

    for review in reviews:
        tokens = re.findall(r"\b\w+\b", review.lower())
        palabras.update(tokens)

    palabras = sorted(palabras)

    print(
        f"palabras únicas encontradas: {len(palabras)}",
        flush=True
    )

    resultados = []

    for palabra in palabras:
        if palabra not in modelo.key_to_index:
            print(f"palabra no encontrada en el vocabulario: {palabra}")
            continue

        vector = modelo[palabra]

        fila = {"palabra": palabra}
        fila.update({
            f"dim_{i + 1}": float(valor)
            for i, valor in enumerate(vector)
        })

        resultados.append(fila)

    embeddings = ejecutar_etapa(
        "construcción de la tabla de embeddings",
        pd.DataFrame,
        resultados
    )

    ejecutar_etapa(
        "exportación CSV",
        embeddings.to_csv,
        ruta_salida,
        index=False
    )

    print(
        f"\nembeddings FastText guardados en: {ruta_salida}",
        flush=True
    )


if __name__ == "__main__":
    main()
