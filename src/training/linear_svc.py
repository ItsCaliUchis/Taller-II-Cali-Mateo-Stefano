from pathlib import Path
import argparse
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import config

from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.base import clone

from utils import (
    ejecutar_etapa,
    verificar_embeddings,
    modelo_sin_scaler,
    modelo_con_scaler,
    optimizar_sin_scaler,
    optimizar_con_scaler,
)

from .graphics import (
    graficar_matriz_confusion,
    graficar_curva_aprendizaje,
)


def graficar_y_guardar_matrices(
    matrices,
    nombres,
    labels,
    titulo,
    ruta_salida,
):
    #arma, guarda y muestra una tabla 2x2 de matrices de confusion
    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(14, 10),
        squeeze=False,
    )

    for ax, matriz, nombre in zip(axes.flat, matrices, nombres):
        graficar_matriz_confusion(
            matriz,
            labels,
            titulo=nombre,
            ax=ax,
        )

    fig.suptitle(titulo, fontsize=16)
    fig.tight_layout(pad=2, rect=[0, 0, 1, 0.96])
    fig.savefig(ruta_salida, dpi=150, bbox_inches="tight")

    plt.show()
    plt.close(fig)


def graficar_y_guardar_curvas(
    curvas,
    nombres,
    titulo,
    ruta_salida,
):
    #arma, guarda y muestra una tabla 2x2 de curvas de aprendizaje
    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(14, 10),
        squeeze=False,
    )

    for ax, curva, nombre in zip(axes.flat, curvas, nombres):
        train_sizes, train_scores, validation_scores = curva

        graficar_curva_aprendizaje(
            train_sizes,
            train_scores,
            validation_scores,
            titulo=f"Curva de aprendizaje - {nombre}",
            metrica="F1-score macro",
            ax=ax,
        )

    fig.suptitle(titulo, fontsize=16)
    fig.tight_layout(pad=2, rect=[0, 0, 1, 0.96])
    fig.savefig(ruta_salida, dpi=150, bbox_inches="tight")

    plt.show()
    plt.close(fig)


def crear_dataframe_reportes(nombres, reportes):
    return pd.DataFrame(
        {"Classification Report": reportes},
        index=pd.Index(nombres, name="Embedding"),
    )


def obtener_argumentos():
    parser = argparse.ArgumentParser(
        description="Entrenamiento de LinearSVC"
    )

    parser.add_argument(
        "--optimizar",
        action="store_true",
        help="ejecutar la optimización de hiperparámetros con Grid Search",
    )

    return parser.parse_args()


def main():
    args = obtener_argumentos()

    # 1. comprobar que exista el dataset limpio
    if not Path(config.dataset_clean).exists():
        print(
            f"El archivo '{config.dataset_clean}' no existe. "
            "Ejecutar primero extractor y luego clean."
        )
        return

    # 2. cargar el dataset
    df = ejecutar_etapa(
        "Cargando dataset",
        pd.read_csv,
        config.dataset_clean,
    )

    # 3. mapear las etiquetas a tres sentimientos
    df["sentiment"] = df["label"].map({
        0: "negative",
        1: "negative",
        2: "neutral",
        3: "positive",
        4: "positive",
    })

    y = df["sentiment"]

    if y.isna().any():
        etiquetas_sin_mapeo = df.loc[y.isna(), "label"].unique()
        raise ValueError(
            "Hay etiquetas sin correspondencia en el mapeo: "
            f"{etiquetas_sin_mapeo}"
        )

    print("\nDistribución de sentimientos:")
    print(y.value_counts())

    # 4. cargar los embeddings
    X_bow, X_bigram, X_monogram, X_ft = ejecutar_etapa(
        "Cargando y verificando embeddings",
        verificar_embeddings,
    )

    embeddings = [
        X_bow,
        X_bigram,
        X_monogram,
        X_ft,
    ]

    nombres = [
        "BOW",
        "TF_IDF Bigrama",
        "TF_IDF Monograma",
        "FastText",
    ]

    labels = [
        "negative",
        "neutral",
        "positive",
    ]

    """Utilizamos Slicing para que excluyamos a Bigram"""
    embeddings_sin_bigram = embeddings[:1] + embeddings[2:] 
    nombres_sin_bigram = nombres[:1] + nombres[2:]

    # 5. verificar que las matrices coincidan con las etiquetas
    for embedding, nombre in zip(embeddings, nombres):
        if embedding is None:
            raise ValueError(
                f"No se pudo cargar el embedding {nombre}."
            )

        print(f"Dimensiones de {nombre}: {embedding.shape}")

        if embedding.shape[0] != len(y):
            raise ValueError(
                f"La cantidad de filas de {nombre} ({embedding.shape[0]}) "
                f"no coincide con la cantidad de etiquetas ({len(y)})."
            )

    print(f"\nDimensiones de y: {y.shape}")


    # 6. validación cruzada: 3 folds
    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42,
    )

    # 7. clasificador base LinearSVC (BoW, Monogram, FastText)
    clf = LinearSVC(
        max_iter=10000,
        random_state=42,
        dual=False,
        tol=1e-3
    )

    clf_BIGRAM = LinearSVC(
        max_iter=3000,
        random_state=42,
        dual=True,
        tol=1e-3
    )

    # =========================================================
    # ENTRENAMIENTO BASE SIN ESCALADO
    # =========================================================
    """
    resultados_sin_scaler_1 = ejecutar_etapa(
        "Entrenando modelos sin escalado [BoW, Monogram, FastText]",
        modelo_sin_scaler,
        embeddings_sin_bigram,
        clf,
        nombres_sin_bigram,
        y,
        cv,
        output_dir = config.svm_nbalanced
    )

    #Para entrenar a TF-IDF Bigrama, utilizamos dual para resolver el problema de optimización

    resultados_sin_scaler_2 = ejecutar_etapa(
        "Entrenando modelos sin escalado [Bigram]",
        modelo_sin_scaler,
        [embeddings[1]],
        clf_BIGRAM,
        [nombres[1]],
        y,
        cv,
        output_dir = config.svm_nbalanced
    )

    cr_sb = [
        resultados_sin_scaler_1["cr_sb"][0],
        resultados_sin_scaler_2["cr_sb"][0],
        resultados_sin_scaler_1["cr_sb"][1],
        resultados_sin_scaler_1["cr_sb"][2],
    ]

    matrices_sb = (
        resultados_sin_scaler_1["matrices"] + resultados_sin_scaler_2["matrices"]
    )

    ln_sb = [
        resultados_sin_scaler_1["ln_sb"][0],
        resultados_sin_scaler_2["ln_sb"][0],
        resultados_sin_scaler_1["ln_sb"][1],
        resultados_sin_scaler_1["ln_sb"][2],
    ]

    df_sb = crear_dataframe_reportes(nombres, cr_sb)
    df_sb.to_csv(
        config.df_sin_scaler_linear_svc,
        index=True,
    )
    """

    # =========================================================
    # ENTRENAMIENTO BASE CON ESCALADO
    # =========================================================
    # with_mean=False permite trabajar con matrices dispersas
    clf_balanceado = make_pipeline(
        StandardScaler(with_mean=False),
        clone(clf),
    )

    clf_balanceado_BIGRAM = make_pipeline(
        StandardScaler(with_mean=False),
        clone(clf_BIGRAM)
    )

    resultados_con_scaler_1 = ejecutar_etapa(
        "Entrenando modelos con escalado [BoW, Monogram, FastText]",
        modelo_con_scaler,
        embeddings_sin_bigram,
        clf_balanceado,
        nombres_sin_bigram,
        y,
        cv,
        output_dir = config.svm_balanced
    )

    resultados_con_scaler_2 = ejecutar_etapa(
        "Entrenando modelos con escalado [Bigram]",
        modelo_con_scaler,
        [embeddings[1]],
        clf_balanceado_BIGRAM,
        [nombres[1]],
        y,
        cv,
        output_dir = config.svm_balanced
    )

    cr_b = [
        resultados_con_scaler_1["cr_b"][0],
        resultados_con_scaler_2["cr_b"][0],
        resultados_con_scaler_1["cr_b"][1],
        resultados_con_scaler_1["cr_b"][2],
    ]

    matrices_b = (
        resultados_con_scaler_1["matrices"] + resultados_con_scaler_2["matrices"]
    )

    ln_b = [
        resultados_con_scaler_1["ln_b"][0],
        resultados_con_scaler_2["ln_b"][0],
        resultados_con_scaler_1["ln_b"][1],
        resultados_con_scaler_1["ln_b"][2],
    ]

    df_b = crear_dataframe_reportes(nombres, cr_b)
    df_b.to_csv(
        config.df_con_scaler_linear_svc,
        index=True,
    )
    # =========================================================
    # GRÁFICOS DEL ENTRENAMIENTO BASE
    # =========================================================

    """
    ejecutar_etapa(
        "Graficando matrices de confusión sin escalado",
        graficar_y_guardar_matrices,
        matrices_sb,
        nombres,
        labels,
        "LinearSVC sin escalado",
        config.grafico_matrices_sin_scaler_linear_svc,
    )
    """
    ejecutar_etapa(
        "Graficando matrices de confusión con escalado",
        graficar_y_guardar_matrices,
        matrices_b,
        nombres,
        labels,
        "LinearSVC con StandardScaler",
        config.grafico_matrices_con_scaler_linear_svc,
    )

    """
    ejecutar_etapa(
        "Graficando curvas de aprendizaje sin escalado",
        graficar_y_guardar_curvas,
        ln_sb,
        nombres,
        "Curvas de aprendizaje LinearSVC sin escalado",
        config.grafico_curvas_sin_scaler_linear_svc,
    )
    """
    ejecutar_etapa(
        "Graficando curvas de aprendizaje con escalado",
        graficar_y_guardar_curvas,
        ln_b,
        nombres,
        "Curvas de aprendizaje LinearSVC con StandardScaler",
        config.grafico_curvas_con_scaler_linear_svc,
    )
    # sin --optimizar, el programa termina después del entrenamiento base
    if not args.optimizar:
        print(
            "\nEntrenamiento base finalizado. "
            "Para ejecutar Grid Search, agregá --optimizar."
        )
        return

    # =========================================================
    # OPTIMIZACIÓN DE HIPERPARÁMETROS
    # =========================================================

    c_values = [0.01, 0.1, 1, 10]

    hiperparametros = [
        # penalty L2 + squared_hinge admite dual=True o dual=False
        {
            "linearsvc__C": c_values,
            "linearsvc__penalty": ["l2"],
            "linearsvc__loss": ["squared_hinge"],
            "linearsvc__dual": [True, False],
        },

        # hinge requiere penalty L2 y dual=True
        {
            "linearsvc__C": c_values,
            "linearsvc__penalty": ["l2"],
            "linearsvc__loss": ["hinge"],
            "linearsvc__dual": [True],
        },

        # penalty L1 requiere squared_hinge y dual=False
        {
            "linearsvc__C": c_values,
            "linearsvc__penalty": ["l1"],
            "linearsvc__loss": ["squared_hinge"],
            "linearsvc__dual": [False],
        },
    ]

    # optimizar sin escalado y sin balanceo
    clf_opt = make_pipeline(
        LinearSVC(
            max_iter=5000,
            random_state=42,
        )
    )
    clf_opt.set_params(
        linearsvc__class_weight=None,
    )

    # optimizar con escalado y balanceo
    clf_balanceado_opt = clone(clf_balanceado)
    clf_balanceado_opt.set_params(
        linearsvc__class_weight="balanced",
    )

    best_params_sb = ejecutar_etapa(
        "Optimizando LinearSVC sin escalado",
        optimizar_sin_scaler,
        hiperparametros,
        embeddings,
        clf_opt,
        nombres,
        y,
        cv,
    )

    best_params_b = ejecutar_etapa(
        "Optimizando LinearSVC con escalado",
        optimizar_con_scaler,
        hiperparametros,
        embeddings,
        clf_balanceado_opt,
        nombres,
        y,
        cv,
    )

    print("\n===== MEJORES PARÁMETROS SIN ESCALADO =====")
    print(best_params_sb)

    print("\n===== MEJORES PARÁMETROS CON ESCALADO =====")
    print(best_params_b)

    # =========================================================
    # EVALUAR LOS MEJORES MODELOS OPTIMIZADOS
    # =========================================================

    resultados_opt_sb = {}

    for embedding, nombre in zip(embeddings, nombres):
        modelo = clone(clf_opt)
        modelo.set_params(**best_params_sb[nombre])

        resultados_opt_sb[nombre] = ejecutar_etapa(
            f"Evaluando {nombre} optimizado sin escalado",
            modelo_sin_scaler,
            [embedding],
            modelo,
            [nombre],
            y,
            cv,
        )

    cr_opt_sb = [
        resultados_opt_sb[nombre]["cr_sb"][0]
        for nombre in nombres
    ]

    matrices_opt_sb = [
        resultados_opt_sb[nombre]["matrices"][0]
        for nombre in nombres
    ]

    ln_opt_sb = [
        resultados_opt_sb[nombre]["ln_sb"][0]
        for nombre in nombres
    ]

    df_opt_sb = crear_dataframe_reportes(nombres, cr_opt_sb)
    df_opt_sb.to_csv(
        config.df_opt_sin_scaler_linear_svc,
        index=True,
    )

    resultados_opt_b = {}

    for embedding, nombre in zip(embeddings, nombres):
        modelo = clone(clf_balanceado_opt)
        modelo.set_params(**best_params_b[nombre])

        resultados_opt_b[nombre] = ejecutar_etapa(
            f"Evaluando {nombre} optimizado con escalado",
            modelo_con_scaler,
            [embedding],
            modelo,
            [nombre],
            y,
            cv,
        )

    cr_opt_b = [
        resultados_opt_b[nombre]["cr_b"][0]
        for nombre in nombres
    ]

    matrices_opt_b = [
        resultados_opt_b[nombre]["matrices"][0]
        for nombre in nombres
    ]

    ln_opt_b = [
        resultados_opt_b[nombre]["ln_b"][0]
        for nombre in nombres
    ]

    df_opt_b = crear_dataframe_reportes(nombres, cr_opt_b)
    df_opt_b.to_csv(
        config.df_opt_con_scaler_linear_svc,
        index=True,
    )

    # =========================================================
    # GRÁFICOS DE LOS MODELOS OPTIMIZADOS
    # =========================================================

    ejecutar_etapa(
        "Graficando matrices optimizadas sin escalado",
        graficar_y_guardar_matrices,
        matrices_opt_sb,
        nombres,
        labels,
        "LinearSVC optimizado sin escalado",
        config.grafico_matrices_opt_sin_scaler_linear_svc,
    )

    ejecutar_etapa(
        "Graficando matrices optimizadas con escalado",
        graficar_y_guardar_matrices,
        matrices_opt_b,
        nombres,
        labels,
        "LinearSVC optimizado con StandardScaler",
        config.grafico_matrices_opt_con_scaler_linear_svc,
    )

    ejecutar_etapa(
        "Graficando curvas optimizadas sin escalado",
        graficar_y_guardar_curvas,
        ln_opt_sb,
        nombres,
        "Curvas LinearSVC optimizado sin escalado",
        config.grafico_curvas_opt_sin_scaler_linear_svc,
    )

    ejecutar_etapa(
        "Graficando curvas optimizadas con escalado",
        graficar_y_guardar_curvas,
        ln_opt_b,
        nombres,
        "Curvas LinearSVC optimizado con StandardScaler",
        config.grafico_curvas_opt_con_scaler_linear_svc,
    )

    # =========================================================
    # GUARDAR LOS MODELOS FINALES
    # =========================================================

    ruta_modelos = Path(config.DATA) / "models"
    ruta_modelos.mkdir(parents=True, exist_ok=True)

    # modelos finales sin escalado
    for embedding, nombre in zip(embeddings, nombres):
        modelo = clone(clf_opt)
        modelo.set_params(**best_params_sb[nombre])
        modelo.fit(embedding, y)

        slug = nombre.lower().replace(" ", "_").replace("-", "_")
        ruta = ruta_modelos / (
            f"linear_svc_{slug}_sin_escalado.pkl"
        )

        joblib.dump(modelo, ruta)
        print(f"Modelo guardado en: {ruta}")

    # modelos finales con escalado y balanceo
    for embedding, nombre in zip(embeddings, nombres):
        modelo = clone(clf_balanceado_opt)
        modelo.set_params(**best_params_b[nombre])
        modelo.fit(embedding, y)

        slug = nombre.lower().replace(" ", "_").replace("-", "_")
        ruta = ruta_modelos / (
            f"linear_svc_{slug}_con_escalado_balanceado.pkl"
        )

        joblib.dump(modelo, ruta)
        print(f"Modelo guardado en: {ruta}")


if __name__ == "__main__":
    main()