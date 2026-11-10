
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import ConfusionMatrixDisplay


def graficar_matriz_confusion(
    cm,
    nombres_clases,
    titulo="Matriz de confusión",
    ax=None
):

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=nombres_clases
    )

    disp.plot(
        cmap="Blues",
        ax=ax,
        colorbar=False
    )

    if ax is not None:
        ax.set_title(titulo)
    else:
        plt.title(titulo)
        plt.show()


def graficar_matrices_confusion(
    matrices,
    nombres,
    nombres_clases,
    titulo="Comparación de modelos"
):

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(14, 10),
        squeeze=False
    )

    for ax, cm, nombre in zip(
        axes.flat,
        matrices,
        nombres
    ):

        graficar_matriz_confusion(
            cm,
            nombres_clases,
            titulo=nombre,
            ax=ax
        )

    fig.suptitle(titulo, fontsize=16)

    plt.tight_layout(pad=2)
    plt.show()


def graficar_curva_aprendizaje(
    train_sizes,
    train_scores,
    validation_scores,
    titulo="Curva de aprendizaje",
    metrica="Score",
    ax=None
):

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    train_mean = np.mean(train_scores, axis=1)
    validation_mean = np.mean(validation_scores, axis=1)

    train_std = np.std(train_scores, axis=1)
    validation_std = np.std(validation_scores, axis=1)

    ax.plot(
        train_sizes,
        train_mean,
        "o-",
        label="Training score"
    )

    ax.plot(
        train_sizes,
        validation_mean,
        "o-",
        label="Validation score"
    )

    ax.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.15
    )

    ax.fill_between(
        train_sizes,
        validation_mean - validation_std,
        validation_mean + validation_std,
        alpha=0.15
    )

    ax.set_xlabel("Cantidad de muestras de entrenamiento")
    ax.set_ylabel(metrica)
    ax.set_title(titulo)

    ax.legend()
    ax.grid()

    return ax

def graficar_curvas_aprendizaje(
    ln,
    nombres,
    titulo="Comparación de modelos"
):

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(14, 10),
        squeeze=False
    )

    for ax, learning_curve, nombre in zip(
        axes.flat,
        ln,
        nombres
    ):

        train_sizes, train_scores, validation_scores = learning_curve

        graficar_curva_aprendizaje(
            train_sizes,
            train_scores,
            validation_scores,
            metrica="F1-score macro",
            titulo=f"Curva de aprendizaje - {nombre}",
            ax=ax
        )

    fig.suptitle(titulo, fontsize=16)

    plt.tight_layout(pad=2, rect=[0, 0, 1, 0.96])
    plt.show()