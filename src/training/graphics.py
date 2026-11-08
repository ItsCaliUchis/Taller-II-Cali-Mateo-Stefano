
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

    n = len(matrices)

    if n == 0:
        return

    ncols = min(2, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(7 * ncols, 5 * nrows),
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


    for ax in axes.flat[n:]:
        ax.set_visible(False)

    fig.suptitle(titulo, fontsize=16)

    plt.tight_layout(w_pad=2)
    plt.show()


def graficar_curva_aprendizaje(
    train_sizes,
    train_scores,
    validation_scores,
    titulo="Curva de aprendizaje",
    metrica="Score"
):

    train_mean = np.mean(train_scores, axis=1)
    validation_mean = np.mean(validation_scores, axis=1)

    train_std = np.std(train_scores, axis=1)
    validation_std = np.std(validation_scores, axis=1)

    plt.figure(figsize=(10, 6))

    plt.plot(
        train_sizes,
        train_mean,
        "o-",
        label="Training score"
    )

    plt.plot(
        train_sizes,
        validation_mean,
        "o-",
        label="Validation score"
    )

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
    plt.ylabel(metrica)
    plt.title(titulo)

    plt.legend()
    plt.grid()
    plt.show()
