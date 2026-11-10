import pandas as pd
import config
from utils import ejecutar_etapa, verificar_embeddings, modelo_con_scaler, modelo_sin_scaler, optimizar_sin_scaler, optimizar_con_scaler
from pathlib import Path
from .graphics import graficar_curvas_aprendizaje, graficar_matrices_confusion

# ===== SCIKIT LEARN ==============
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
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

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def main():

    if not Path(config.dataset_clean).exists():
        print(f"El archivo ubicado en ruta '{config.dataset_clean}' no existe. Ejecutar comando 'extractor' y luego 'clean'")
        return

    X_bow, X_bigram, X_monogram, X_ft = verificar_embeddings()

# Dejamos pocos parametros estandar para despues ir optimizando con los modelos
    
    clf = SGDClassifier(max_iter=1000, random_state=42)
    clf_balanceado = make_pipeline(
        StandardScaler(with_mean=False), # se usa with_mean = False porque todas las matrices son esparcidas
        clf)

    embeddings = [X_bow, X_bigram, X_monogram, X_ft]
    nombres = ["BOW", "TF-IDF Bigrama", "TF-IDF Monograma", "FastText"]
    labels = ["negative", "neutral", "positive"]


    resultados_sin_scaler = ejecutar_etapa(
        "Entrenando modelos sin escalado",
        modelo_sin_scaler,
        embeddings,
        clf,
        nombres,
        y,
        cv
    )

    cr_sb = resultados_sin_scaler["cr_sb"] # classification report sin balancear (sin scaler)

    matrices_sb = resultados_sin_scaler["matrices"] # confussion matrices sin balancear

    ln_sb = resultados_sin_scaler["ln_sb"] # learning curves sin balancear

    df_sb_sgd = pd.DataFrame({
        "Embedding": nombres,
        "Classification Report": cr_sb
    })
    
    df_sb_sgd.to_csv(config.df_sb_sgd, index=False)

    resultados_con_scaler = ejecutar_etapa(
        "Entrenando modelos con escalado",
        modelo_con_scaler,
        embeddings,
        clf_balanceado,
        nombres,
        y,
        cv
    )

    cr_b = resultados_con_scaler["cr_b"] # classification report con scaler (balanceado)

    matrices_b = resultados_con_scaler["matrices"] # confussion matrices balanceadas

    ln_b = resultados_con_scaler["ln_b"] # learning curves balanceadas

    df_b_sgd = pd.DataFrame({
        "Embedding": nombres,
        "Classification Report": cr_b
    })

    df_b_sgd.to_csv(config.df_b_sgd, index=False)


    ejecutar_etapa(
        "Graficando Confussion Matrices sin escalado",
        graficar_matrices_confusion,
        matrices_sb,
        nombres,
        labels,
        titulo="SGD sin escalado"
    )

    ejecutar_etapa(
        "Graficando Confussion Matrices con escalado",
        graficar_matrices_confusion,
        matrices_b,
        nombres,
        labels,
        titulo="SGD con StandardScaler"
    )

    ejecutar_etapa(
        "Graficando Learning Curves sin escalado",
        graficar_curvas_aprendizaje,
        ln_sb,
        nombres,
        titulo = "Curvas de aprendizaje sin escalado"
    )

    ejecutar_etapa(
        "Graficando Learning Curves con escalado",
        graficar_curvas_aprendizaje,
        ln_b,
        nombres,
        titulo = "Curvas de aprendizaje con StandardScaler"
    )

# ====== MODULO DE OPTIMIZACION DE MODELO ======

    hiperparametros = [
    {
        "sgdclassifier__loss": ["hinge", "log_loss", "modified_huber"],
        "sgdclassifier__penalty": ["l2", "elasticnet"],
        "sgdclassifier__alpha": [1e-5, 1e-4, 1e-3],
        "sgdclassifier__max_iter": [1000, 3000],
        "sgdclassifier__learning_rate": ["optimal"]
    },

    {
        "sgdclassifier__loss": ["hinge", "log_loss", "modified_huber"],
        "sgdclassifier__penalty": ["l2", "elasticnet"],
        "sgdclassifier__alpha": [1e-5, 1e-4, 1e-3],
        "sgdclassifier__max_iter": [1000],
        "sgdclassifier__learning_rate": ["constant", "adaptive"],
        "sgdclassifier__eta0": [0.001, 0.01]
    }
    ]

    clf_opt = make_pipeline(SGDClassifier(max_iter=1000, random_state=42))
    clf_opt.set_params(sgdclassifier__class_weight = None)

    clf_balanceado_opt = clone(clf_balanceado)
    clf_balanceado_opt.set_params(sgdclassifier__class_weight = "balanced")

    resultados_opt_sin_scaler = ejecutar_etapa(
        "Optimizando modelos sin scaler",
        optimizar_sin_scaler,
        hiperparametros,
        embeddings,
        clf_opt,
        nombres,
        y,
        cv
    )

    best_params_sb = resultados_opt_sin_scaler["best_params"]
    best_models_sb = resultados_opt_sin_scaler["best_models"]

    resultados_opt_con_scaler = ejecutar_etapa(
        "Optimizando modelos con scaler",
        optimizar_con_scaler,
        hiperparametros,
        embeddings,
        clf_balanceado_opt,
        nombres,
        y,
        cv
    )

    best_params_b = resultados_opt_con_scaler["best_params"]
    best_models_b = resultados_opt_con_scaler["best_models"]

    print("====== RESULTADOS DE OPTIMIZACION SIN ESCALADO ======")
    print(best_params_sb)
    print("\n")
    print(best_models_sb)

    print("\n")

    print("===== RESULTADOS DE OPTIMIZACION CON ESCALADO ======")
    print(best_params_b)
    print("\n")
    print(best_models_b)


if __name__ == "__main__":
    main()