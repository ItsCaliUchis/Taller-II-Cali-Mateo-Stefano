import pandas as pd
import config
from utils import ejecutar_etapa, verificar_embeddings
from pathlib import Path
import subprocess
import numpy as np

# ===== SCIKIT LEARN ==============
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

if not Path(config.dataset_clean).exists:
    print(f"La ruta para '{config.dataset_clean}' no existe")


df = ejecutar_etapa(
        "Cargando dataset",
        pd.read_csv,
        config.dataset_clean
    )

df["sentiment"] = df["label"].map({
    0: "negative",
    1: "negative",
    2: "neutral",
    3: "positive",
    4: "positive"
})

y = df["sentiment"]

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=102)

def modelo():
    clf = SGDClassifier(loss="log_loss", penalty="l2", max_iter=1000) # usamos 1000 para mayor convergencia
    return clf

def entrenar(X_modelo,y,clf):

    X_train, X_test, y_train, y_test = train_test_split(
        X_modelo,
        y,
        train_size=0.80,
        test_size=0.20,
        shuffle=True,
        stratify=y,
        random_state=102
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["negative", "neutral", "positive"]
    )


    print(classification_report(y_test, y_pred))
    print("\n")


# CURVA DE APRENDIZAJE

    train_sizes, train_scores, validation_scores = learning_curve(
        clf,
        X_train,
        y_train,
        train_sizes=np.linspace(0.1, 1.0, 5),
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1
    )

    train_mean = np.mean(train_scores, axis=1)
    validation_mean = np.mean(validation_scores, axis=1)

    train_std = np.std(train_scores, axis=1)
    validation_std = np.std(validation_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, "o-", label="Training score")
    plt.plot(train_sizes, validation_mean, "o-", label="Validation score")

    plt.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.15
    )

    plt.fill_between(
        train_sizes,
        validation_mean - validation_std,
        validation_mean + validation_std,
        alpha=0.15
    )

    plt.xlabel("Cantidad de muestras de entrenamiento")
    plt.ylabel("F1-score macro")
    plt.title(f"Curva de aprendizaje - SGDClassifier")

    plt.legend()
    plt.grid()
    plt.show()

    return cm

def graficar(matrices, nombres):

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18,5))
    
    for ax, cm, nombre in zip(axes, matrices, nombres):
        disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["negative", "neutral", "positive"]
        )

        disp.plot(cmap="Blues", ax=ax)
        ax.set_title(nombre) 

    plt.tight_layout(pad=2)
    plt.show()


def main():

    if not Path(config.dataset_clean).exists():
        print(f"El archivo ubicado en ruta '{config.dataset_clean}' no existe. Ejecutar comando 'extractor' y luego 'clean'")
        return


    X_bow, X_bigram, X_monogram = verificar_embeddings()
    """print("Dimensiones de X:", X_ft.shape)
    print("Dimensiones de y:", y.shape)

    assert X_ft.shape[0] == len(y) # La cantidad de reviews y etiquetas no coincide"""
    

    embeddings = [X_bow, X_bigram, X_monogram]
    nombres = ["BOW", "TF-IDF Bigrama", "TF-IDF Monograma"]
    matrices = []

    for embedding, nombre in zip(embeddings,nombres):
        print(f"\nEntrenando con {nombre}")
        clf = modelo()
        cm = entrenar(embedding,y,clf)
        matrices.append(cm)

    graficar(matrices, nombres)

if __name__ == "__main__":
    main()