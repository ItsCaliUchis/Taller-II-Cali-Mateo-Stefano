import argparse
from pathlib import Path
import pandas as pd
import config

def verificar_labels(df):
    diferentes = df[df['label'] != df['label_text']]
    if diferentes.empty:
        print("\nLabel y label_text son coincidentes")
    return None

def exploratorio(df):
    print("===============================================")
    print("Columnas del dataset:")
    print(df.columns.tolist()) # ver que columnas tiene el dataset; no hay distincion entre label y label-text
    print("\nPrimeras 10 filas:")
    print(df.head(10))
    print("\nValores nulos por columna:")
    print(df.isnull().sum()) # no hay nulos tampoco
    print("===============================================\n")

    # Revisar si hay reviews que no estan en ingles - Primeros 2 caracteres del id son el idioma de la review
    cumplen = df[df['id'].astype(str).str[:2] != 'en']['id']
    if not cumplen.empty: 
        for cumple in cumplen:
            print(cumple)
    else:
        print("Todas las reviews son en ingles")
    verificar_labels(df)

def main():
    argparse.ArgumentParser(
        description='Realiza un análisis exploratorio del dataset raw.'
    ).parse_args()

    if Path(config.dataset_raw).exists():
        print('\nSe esta cargando el dataset\n')
        df = pd.read_csv(config.dataset_raw)
        print(f'\nSe ha cargado el dataset con {len(df)} registros, se procede a explorar el dataset\n')
        exploratorio(df)
    else:
        print(f'No existe el dataset en la ruta "{config.dataset_raw}", ejecute primero el comando "extractor" para descargar el dataset y luego ejecute este comando para explorar el dataset')
