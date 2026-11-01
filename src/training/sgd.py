import pandas as pd
import config
from utils import ejecutar_etapa, verificar_embeddings
from pathlib import Path
import subprocess
import numpy as np

# ===== SCIKIT LEARN ==============
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
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

def entrenar(X_modelo,y):
    X_train, X_test, y_train, y_test = train_test_split(
        X_modelo,
        y,
        train_size=0.80,
        test_size=0.20,
        shuffle=True,
        stratify=y,
        random_state=102
    )

    clf = SGDClassifier(loss="log_loss", penalty="l2", max_iter=100).fit(X, y)
    y_pred = clf.predict(X_test)

    cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["negative", "neutral", "positive"]
    )

    disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["negative", "neutral", "positive"]
    )

    print(classification_report(y_test, y_pred))

    disp.plot(cmap="Blues")
    plt.show()


def main():

    if not Path(config.dataset_clean).exists():
        print(f"El archivo ubicado en ruta '{config.dataset_clean}' no existe. Ejecutar comando 'extractor' y luego 'clean'")
        return

    X_bow, X_bigram, X_monogram = verificar_embeddings()
    entrenar(X_bow,y)

if __name__ == "__main__":
    main()