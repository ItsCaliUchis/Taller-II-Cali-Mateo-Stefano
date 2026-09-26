import pandas as pd
import config

df = pd.read_csv(config.dataset_raw)

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

cleanup(df)