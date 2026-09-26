import pandas as pd
import config

df = pd.read_csv(config.dataset_raw)

def verificar_labels(df):
    diferentes = df[df['label'] != df['label_text']]
    if diferentes.empty:
        print("Label y label_text son coincidentes")
    return None

def exploratorio(df):
    print(df.columns.tolist()) # ver que columnas tiene el dataset; no hay distincion entre label y label-text
    print(df.head(10))
    print(df.isnull().sum()) # no hay nulos tampoco

    # Revisar si hay reviews que no estan en ingles - Primeros 2 caracteres del id son el idioma de la review
    cumplen = df[df['id'].astype(str).str[:2] != 'en']['id']
    if not cumplen.empty: 
        for cumple in cumplen:
            print(cumple)
    else:
        print("Todas las reviews son en ingles")
    verificar_labels(df)


exploratorio(df)