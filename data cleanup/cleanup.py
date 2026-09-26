import pandas as pd

path = "data cleanup/raw/dataset.csv"

df = pd.read_csv(path)

def cleanup(df):
    df = df.drop(columns=['label_text'])
    reviews_limpias = []
    for review in df['text']:
        reviews_limpias.append(review.strip().lower())

    df['reviews'] = reviews_limpias
    df = df.drop(columns=['text'])
    df = df[['Unnamed: 0','id','reviews','label']]
    print(df.head(20))
    df.to_csv("data cleanup/clean/dataset_limpio.csv")
    
cleanup(df)