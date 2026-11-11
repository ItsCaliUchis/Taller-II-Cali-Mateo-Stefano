import argparse
import joblib
import pandas as pd
import config
from utils import ejecutar_etapa, verificar_embeddings, modelo_con_scaler, modelo_sin_scaler, optimizar_sin_scaler, optimizar_con_scaler
from pathlib import Path

# ===== BIBLIOTECA SCIKIT-LEARN ==============
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.base import clone

def main():
    parser = argparse.ArgumentParser(description="Entrena y evalua clasificadores SGD.")
    parser.add_argument("--jobs", type=int, default=1, help="Procesos para cada curva de aprendizaje (default: 1).")
    parser.add_argument("--full-evaluation", action="store_true", help="Usa 5 particiones y 5 tamanos por curva; el modo estandar usa 3 y 3.")
    parser.add_argument("--no-plots", action="store_true", help="No abre graficos; util para ejecucion desatendida.")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs debe ser un entero positivo")

    if not Path(config.dataset_clean).exists():
        print(f"No existe el archivo '{Path(config.dataset_clean).name}'. Ejecute 'extractor' y luego 'clean'.")
        return 1

    df = ejecutar_etapa("Cargando dataset", pd.read_csv, config.dataset_clean)
    y = df["label"].map({0: "negative", 1: "negative", 2: "neutral", 3: "positive", 4: "positive"})
    if y.isna().any():
        raise ValueError("El dataset contiene etiquetas fuera del rango esperado 0..4.")
    cv_folds = 5 if args.full_evaluation else 3
    curve_points = 5 if args.full_evaluation else 3
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    X_bow, X_bigram, X_monogram, X_ft = verificar_embeddings()

    input_paths = [
        config.dataset_clean,
        config.bow_vect,
        config.idf_bigram,
        config.idf_monogram,
        config.fasttext_embeddings,
    ]
    input_signature = [
        (str(path), Path(path).stat().st_size, Path(path).stat().st_mtime_ns)
        for path in input_paths
    ]
    run_id = joblib.hash((input_signature, cv_folds, curve_points))[:16]
    checkpoint_dir = config.OPTIMIZATION_RESULTS / "sgd_checkpoints" / run_id

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
        cv,
        checkpoint_dir=checkpoint_dir,
        checkpoint_prefix="baseline_unscaled",
        n_jobs=args.jobs,
        curve_points=curve_points,
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
        cv,
        checkpoint_dir=checkpoint_dir,
        checkpoint_prefix="baseline_scaled",
        n_jobs=args.jobs,
        curve_points=curve_points,
    )

    cr_b = resultados_con_scaler["cr_b"] # classification report con scaler (balanceado)

    matrices_b = resultados_con_scaler["matrices"] # confussion matrices balanceadas

    ln_b = resultados_con_scaler["ln_b"] # learning curves balanceadas

    df_b_sgd = pd.DataFrame({
        "Embedding": nombres,
        "Classification Report": cr_b
    })

    df_b_sgd.to_csv(config.df_b_sgd, index=False)


    if not args.no_plots:
        from .graphics import graficar_curvas_aprendizaje, graficar_matrices_confusion

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
    """

# ====== MODULO DE OPTIMIZACION DE MODELO (Los resultados estan en la carpeta "optimization_results")======

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

    best_params_sb = ejecutar_etapa(
        "Optimizando modelos sin scaler",
        optimizar_sin_scaler,
        hiperparametros,
        embeddings,
        clf_opt,
        nombres,
        y,
        cv
    )

    best_params_b = ejecutar_etapa(
        "Optimizando modelos con scaler",
        optimizar_con_scaler,
        hiperparametros,
        embeddings,
        clf_balanceado_opt,
        nombres,
        y,
        cv
    )

    print("====== RESULTADOS DE OPTIMIZACION SIN ESCALADO ======")
    print(best_params_sb)
    print("\n")

    print("===== RESULTADOS DE OPTIMIZACION CON ESCALADO ======")
    print(best_params_b)
    print("\n")
    """

    clf_bow = SGDClassifier(
            alpha = 1e-05,
            eta0 = 0.01,
            learning_rate = "constant",
            loss = "log_loss",
            max_iter = 1000,
            penalty = "elasticnet",
            random_state = 42
    )

    clf_bigram = SGDClassifier(
            alpha = 1e-05,
            learning_rate = "optimal",
            loss = "modified_huber",
            max_iter = 1000,
            penalty = "l2",
            random_state = 42
    )

    clf_monogram = SGDClassifier(
            alpha = 1e-05,
            learning_rate = "optimal",
            loss = "modified_huber",
            max_iter = 1000,
            penalty = "l2",
            random_state = 42  
    )

    clf_ft = SGDClassifier(
            alpha = 1e-05,
            learning_rate = "optimal",
            loss = "log_loss",
            max_iter = 1000,
            penalty = "l2",
            random_state = 42     
    )

    clf_opt_sin_scaler = [clf_bow, clf_bigram, clf_monogram, clf_ft]

    clf_bow_balanceado = make_pipeline(
        StandardScaler(with_mean=False),
        SGDClassifier(
            alpha = 0.001,
            learning_rate = "optimal",
            loss = "log_loss",
            max_iter = 1000,
            penalty = "elasticnet",
            class_weight = "balanced",
            random_state = 42
        )
    )

    clf_bigram_balanceado = make_pipeline(
        StandardScaler(with_mean=False),
        SGDClassifier(
            alpha = 0.001,
            eta0 = 0.01,
            learning_rate = "adaptive",
            loss = "log_loss",
            max_iter = 1000,
            penalty = "elasticnet",
            class_weight = "balanced",
            random_state = 42
        )
    )

    clf_monogram_balanceado = make_pipeline(
        StandardScaler(with_mean=False),
        SGDClassifier(
            alpha = 0.001,
            learning_rate = "optimal",
            loss = "log_loss",
            max_iter = 1000,
            penalty = "elasticnet",
            class_weight = "balanced",
            random_state = 42           
        )
    )

    clf_ft_balanceado = make_pipeline(
        StandardScaler(with_mean=False),
        SGDClassifier(
            alpha = 0.0001,
            eta0 = 0.01,
            learning_rate = "adaptive",
            loss = "log_loss",
            max_iter = 1000,
            penalty = "elasticnet",
            class_weight = "balanced",
            random_state = 42            
        )
    )

    clf_opt_con_scaler = [clf_bow_balanceado, clf_bigram_balanceado, clf_monogram_balanceado, clf_ft_balanceado]

    resultados_1 = {}

    for embedding, nombre, clf in zip(embeddings, nombres, clf_opt_sin_scaler):
        resultado = modelo_sin_scaler(
            [embedding],
            clf,
            [nombre],
            y,
            cv,
            checkpoint_dir=checkpoint_dir,
            checkpoint_prefix="tuned_unscaled",
            n_jobs=args.jobs,
            curve_points=curve_points,
        )

        resultados_1[nombre] = resultado


    cr_opt_sb = [resultados_1[nombre]["cr_sb"][0] for nombre in nombres]

    matrices_opt_sb = [resultados_1[nombre]["matrices"][0] for nombre in nombres]

    ln_opt_sb = [resultados_1[nombre]["ln_sb"][0] for nombre in nombres]

    df_optimizado_sb_sgd = pd.DataFrame({
        "Embedding": nombres,
        "Classification Report": cr_opt_sb
    })

    df_optimizado_sb_sgd.to_csv(
        config.df_optimizado_sb_sgd,
        index=False
    )

    resultados_2 = {}

    for embedding, nombre, clf in zip(embeddings, nombres, clf_opt_con_scaler):
        resultado = modelo_con_scaler(
            [embedding],
            clf,
            [nombre],
            y,
            cv,
            checkpoint_dir=checkpoint_dir,
            checkpoint_prefix="tuned_scaled",
            n_jobs=args.jobs,
            curve_points=curve_points,
        )

        resultados_2[nombre] = resultado


    cr_opt_b = [resultados_2[nombre]["cr_b"][0] for nombre in nombres]

    matrices_opt_b = [resultados_2[nombre]["matrices"][0] for nombre in nombres]

    ln_opt_b = [resultados_2[nombre]["ln_b"][0] for nombre in nombres]

    df_optimizado_b_sgd = pd.DataFrame({
        "Embedding": nombres,
        "Classification Report": cr_opt_b
    })

    df_optimizado_b_sgd.to_csv(
        config.df_optimizado_b_sgd,
        index=False
    )

    if not args.no_plots:
        from .graphics import graficar_curvas_aprendizaje, graficar_matrices_confusion

        ejecutar_etapa(
        "Graficando Confussion Matrices sin escalado (OPTIMIZADO)",
        graficar_matrices_confusion,
        matrices_opt_sb,
        nombres,
        labels,
        titulo="SGD sin escalado"
        )

        ejecutar_etapa(
        "Graficando Confussion Matrices con escalado (OPTIMIZADO)",
        graficar_matrices_confusion,
        matrices_opt_b,
        nombres,
        labels,
        titulo="SGD con StandardScaler"
        )

        ejecutar_etapa(
        "Graficando Learning Curves sin escalado (OPTIMIZADO)",
        graficar_curvas_aprendizaje,
        ln_opt_sb,
        nombres,
        titulo = "Curvas de aprendizaje sin escalado"
        )

        ejecutar_etapa(
        "Graficando Learning Curves con escalado (OPTIMIZADO)",
        graficar_curvas_aprendizaje,
        ln_opt_b,
        nombres,
        titulo = "Curvas de aprendizaje con StandardScaler"
        )

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
