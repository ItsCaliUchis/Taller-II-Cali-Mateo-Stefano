import pandas as pd

path = "data cleanup/raw/dataset.csv"

df = pd.read_csv(path)

for reviews in df['text'][:10]:
    print(reviews)
    print("\n")