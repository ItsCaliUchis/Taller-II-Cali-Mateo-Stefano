import pandas as pd

path = "data cleanup/raw/dataset.csv"

df = pd.read_csv(path)

def verificar_labels(df):
    diferentes = df[df['label'] != df['label_text']]
    if diferentes.empty:
        print("Label y label_text son coincidentes")
    return None

def exploratorio(df):
    print(df.columns.tolist()) # ver que columnas tiene el dataset; no hay distincion entre label y label-text
    print(df.head(10))
    print(df.isnull().sum()) # no hay nulos tampoco
    # extracto de logica para ver si hay reviews que no estan en ingles
    cumplen = df[df['id'].astype(str).str[:2] != 'en']['id']
    if not cumplen.empty: 
        for cumple in cumplen:
            print(cumple)
    else:
        print("Todas las reviews son en ingles")
    verificar_labels(df)


exploratorio(df)