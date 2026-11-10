from utils import verificar_embeddings
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X_bow, X_bigram, X_monogram, X_ft = verificar_embeddings()
embeddings = [X_bow, X_bigram, X_monogram, X_ft]
nombres = ["BoW", "Bigram", "Monogram", "FastText"]

def aplicar_pca(embedding):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(embedding)
    return X_pca, pca.explained_variance_ratio_


def escalado(embeddings):

    scaler = StandardScaler(with_mean=False)

    embeddings_escalados = [
        scaler.fit_transform(embedding)
        for embedding in embeddings] # se escalan los embeddings
    
    resultados = [
        aplicar_pca(embedding)
        for embedding in embeddings_escalados
    ]

    embeddings_pca = [r[0] for r in resultados]
    varianza = [r[1] for r in resultados]

    return embeddings_pca, varianza

embeddings_pca, varianza = escalado(embeddings)

varianza_acumulada = [
    v.sum()
    for v in varianza
]


for nombre, v, acumulada in zip(nombres, varianza, varianza_acumulada):
    print(f"{nombre}:")
    print(f"  PC1: {v[0]*100:.2f}%")
    print(f"  PC2: {v[1]*100:.2f}%")
    print(f"  Acumulada: {acumulada*100:.2f}%")

for nombre, embedding in zip(nombres, embeddings):
    print(nombre, embedding.shape)
    print("Media:", embedding.mean())
    print("Varianza:", embedding.multiply(embedding).mean() - embedding.mean()**2)
    print("\n")

"""CONCLUSION: No se puede hacer un Boundary Decision Display dado que
el modelo aprenderia de dos dimensiones de cada embedding las cuales por medio de un analisis exploratorio,
conservan muy poca varianza"""