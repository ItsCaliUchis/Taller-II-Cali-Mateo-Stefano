
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

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
    titulo="Comparación de modelos",
    *,
    save_path=None,
    show=True,
):
    count = max(1, len(matrices))
    ncols = min(2, count)
    nrows = int(np.ceil(count / ncols))
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

    fig.suptitle(titulo, fontsize=16)
    for ax in list(axes.flat)[count:]:
        ax.set_visible(False)
    plt.tight_layout(pad=2)
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


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
    titulo="Comparación de modelos",
    *,
    save_path=None,
    show=True,
):
    count = max(1, len(ln))
    ncols = min(2, count)
    nrows = int(np.ceil(count / ncols))
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(7 * ncols, 5 * nrows),
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
    for ax in list(axes.flat)[count:]:
        ax.set_visible(False)
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def graficar_historial_rna(history, titulo, save_path, *, show=False):
    """Guarda pérdida y F1 macro de entrenamiento/validación por época."""
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for axis, metric, ylabel in (
        (axes[0], "loss", "Loss"),
        (axes[1], "macro_f1", "F1 macro"),
    ):
        axis.plot(epochs, history[f"train_{metric}"], "o-", label="Train")
        axis.plot(epochs, history[f"val_{metric}"], "o-", label="Validation")
        axis.set_xlabel("Época")
        axis.set_ylabel(ylabel)
        axis.set_title(ylabel)
        axis.grid(True, alpha=0.3)
        axis.legend()
    fig.suptitle(titulo)
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
