"""Análisis exploratorio PCA de las representaciones disponibles."""

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def aplicar_pca(embedding):
    """Proyecta una matriz de características sobre dos componentes."""
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(embedding)
    return X_pca, pca.explained_variance_ratio_


def escalado(embeddings):
    """Escala cada matriz dispersa y calcula su proyección PCA."""
    scaler = StandardScaler(with_mean=False)
    embeddings_escalados = [scaler.fit_transform(embedding) for embedding in embeddings]
    resultados = [aplicar_pca(embedding) for embedding in embeddings_escalados]
    return [r[0] for r in resultados], [r[1] for r in resultados]


def main():
    """Carga los embeddings y muestra varianza y estadísticas descriptivas."""
    from utils import verificar_embeddings

    embeddings = list(verificar_embeddings())
    nombres = ["BoW", "Bigram", "Monogram", "FastText"]
    _, varianza = escalado(embeddings)

    for nombre, v in zip(nombres, varianza):
        print(f"{nombre}:")
        print(f"  PC1: {v[0] * 100:.2f}%")
        print(f"  PC2: {v[1] * 100:.2f}%")
        print(f"  Acumulada: {v.sum() * 100:.2f}%")

    for nombre, embedding in zip(nombres, embeddings):
        print(nombre, embedding.shape)
        print("Media:", embedding.mean())
        print("Varianza:", embedding.multiply(embedding).mean() - embedding.mean() ** 2)


if __name__ == "__main__":
    main()
