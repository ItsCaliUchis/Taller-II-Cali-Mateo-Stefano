import pandas as pd
import config
from utils import ejecutar_etapa, verificar_embeddings
from pathlib import Path
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from .graphics import graficar_curva_aprendizaje, graficar_matrices_confusion

# ===== SCIKIT LEARN ==============
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.base import clone

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

def entrenar(X_modelo,y,clf,nombre_modelo):

    clf = clone(clf)

    X_train, X_test, y_train, y_test = train_test_split(
        X_modelo,
        y,
        train_size=0.80,
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

    graficar_curva_aprendizaje(
        train_sizes,
        train_scores,
        validation_scores,
        titulo=f"Curva de aprendizaje - {nombre_modelo}",
        metrica="F1-score macro"
    )

    return cm


def main():

    if not Path(config.dataset_clean).exists():
        print(f"El archivo ubicado en ruta '{config.dataset_clean}' no existe. Ejecutar comando 'extractor' y luego 'clean'")
        return


    X_bow, X_bigram, X_monogram, X_ft = verificar_embeddings()
    
    #print("Dimensiones de X:", X_ft.shape)
    #print("Dimensiones de y:", y.shape)

    #assert X_ft.shape[0] == len(y) La cantidad de reviews y etiquetas no coincide

    clf = SGDClassifier(loss="log_loss", penalty="l2", max_iter=1000)
    clf_balanceado = make_pipeline(StandardScaler(with_mean=False), SGDClassifier(loss="log_loss", penalty="l2", max_iter=1000))

    embeddings = [X_bow, X_bigram, X_monogram, X_ft]
    nombres = ["BOW", "TF-IDF Bigrama", "TF-IDF Monograma", "FastText"]
    matrices_sb = [] # sb = sin balancear
    matrices_b = [] # b = balanceadas

    for embedding, nombre in zip(embeddings,nombres):
        print(f"\nEntrenando SGD sin escalado: {nombre}")
        cm = entrenar(embedding,y,clf,nombre_modelo=f"SGD sin escalado {nombre}")
        matrices_sb.append(cm)
        
    for embedding, nombre in zip(embeddings,nombres):
        print(f"\nEntrenando SGD con StandardScaler: {nombre}")  
        cm = entrenar(embedding,y,clf_balanceado,nombre_modelo=f"SGD escalado {nombre}")
        matrices_b.append(cm)

    graficar_matrices_confusion(
        matrices_sb,
        nombres,
        ["negative", "neutral", "positive"],
        titulo="SGD sin escalado"
    )

    graficar_matrices_confusion(
        matrices_b,
        nombres,
        ["negative", "neutral", "positive"],
        titulo="SGD con StandardScaler"
    )

if __name__ == "__main__":
    main()