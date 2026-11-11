import argparse
from pathlib import Path
import pandas as pd
import config

def cleanup(df):
    df = df.drop(columns=['label_text'])

    reviews_limpias = []
    for review in df['text']:
        reviews_limpias.append(review.strip().lower())

    df['reviews'] = reviews_limpias
    df = df.drop(columns=['text'])

    if 'Unnamed: 0' in df.columns:
        df = df[['Unnamed: 0', 'id', 'reviews', 'label']]
    else:
        df = df[['id', 'reviews', 'label']]

    print(df.head(20).to_string().encode("ascii", "replace").decode("ascii"))
    df.to_csv(config.dataset_clean, index=False)

def main():
    argparse.ArgumentParser(
        description='Limpia el dataset raw y genera el dataset preparado.'
    ).parse_args()

    if Path(config.dataset_raw).exists():
        print('\nSe esta cargando el dataset\n')
        df = pd.read_csv(config.dataset_raw)
        print(f'\nSe ha cargado el dataset con {len(df)} registros, ahora se procede a limpiar el dataset\n')
        cleanup(df)
        print('\nSe ha guardado el dataset limpio en la carpeta data')
        return 0
    else:
        print(f'No existe el dataset "{Path(config.dataset_raw).name}". Ejecute primero "extractor" para descargarlo.')
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
