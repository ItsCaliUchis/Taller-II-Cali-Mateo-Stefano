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

    print(df.head(20))
    df.to_csv(config.dataset_clean, index=False)

if Path(config.dataset_raw).exists():
    print(f'\nSe esta cargando el dataset\n')
    df = pd.read_csv(config.dataset_raw)
    print(f'\nSe ha cargado el dataset con {len(df)} registros, ahora se procede a limpiar el dataset\n')
    cleanup(df)
    print(f'\nSe ha guardado el dataset limpio en la carpeta data')
else:
    print(f'No existe el dataset en la ruta "{config.dataset_raw}", ejecute primero el script "extractor/extractor.py" para descargar el dataset y luego ejecute este script para limpiar el dataset')