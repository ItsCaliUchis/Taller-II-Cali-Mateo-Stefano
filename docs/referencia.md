# API pública del proyecto

Esta página recoge los valores y funciones usados desde otros módulos como interfaz compartida. Los helpers que solo sirven a una función del mismo módulo se omiten. Los puntos de entrada `main()` y sus opciones están descritos en [Comandos CLI](comandos.md); las funciones específicas de cada etapa se explican en [Arquitectura](arquitectura.md).

## Rutas y configuración — `src/config.py`

| Nombre | Uso |
|---|---|
| `PROJECT_ROOT`, `DATA`, `CLASSIFICATION_REPORTS`, `RNA_REPORTS`, `OPTIMIZATION_RESULTS` | Rutas base del repositorio. `config.rna_report_dir(embedding)` crea y devuelve la carpeta de reportes RNA correspondiente a BOW, TF-IDF mono/bigrama o FastText. |
| Rutas de archivos (`dataset_raw`, `dataset_clean`, `bow_vect`, `bow_vocab`, `idf_bigram`, `idf_monogram`, `fasttext_embeddings`, `fasttext_vocab`, `bert_vect`, `vocabulario_bert`, `elmo_vect`, `vocabulario_elmo`, `negaciones_path`) | Ubicaciones canónicas de entradas y salidas completas. Las ejecuciones limitadas derivan rutas con `ruta_limitada`; no requieren variables separadas. |
| `ruta_limitada(output_path, limit)` | Retorna la ruta base si `limit` es `None`; con límite positivo, agrega `_limit_N` al nombre antes de la extensión. Rechaza límites no positivos. Es la política compartida de nombres de salida del preprocesamiento. |
| `cargar_negaciones()` | Lee la lista de negaciones; retorna `[]` si el archivo no existe o está vacío. |
| `cargar_resultados_sgd()` | Lee el JSON de optimización configurado; retorna `{}` si falta o está vacío. Está disponible, pero el flujo SGD actual no la usa. |

## Funciones compartidas — `src/utils.py`

| Función | Consumidores | Comportamiento |
|---|---|---|
| `ejecutar_etapa(nombre, funcion, *args, **kwargs)` | Preprocesamiento y entrenamiento | Ejecuta una operación mostrando tiempo y RAM; retorna el valor de la operación y vuelve a lanzar sus excepciones. |
| `extractor_reviews(df)` | Vectorizadores | Limpia y tokeniza la columna `reviews`, elimina stopwords inglesas salvo las negaciones configuradas y retorna una lista de textos. |
| `obtener_limit()` | Comandos de vectorización | Lee `--limit` de la CLI; permite ausencia o entero positivo. |
| `obtener_ruta_salida(path, limit)` | TF-IDF | Fachada de compatibilidad que delega el nombre limitado a `config.ruta_limitada`. |
| `obtener_ruta_vocabulario(matrix_path)` | TF-IDF | Deriva el nombre del CSV de vocabulario a partir de la ruta de la matriz. |
| `guardar_vocabulario(vocabulary, output_path)` | Vectorizadores | Escribe CSV con columnas `index` y `term`. |
| `verificar_recursos_nltk()` | Vectorizadores | Comprueba y descarga si faltan los recursos `punkt_tab` y `stopwords`. |
| `verificar_embeddings()` | Entrenamiento | Carga BoW, TF-IDF y FastText completos; ejecuta los comandos generadores si faltan. No carga embeddings limitados, BERT ni ELMo. |
| `entrenar(X_modelo, y, clf, cv)` | Evaluación SGD | Clona, divide, ajusta y evalúa un clasificador; retorna matriz de confusión, curva de aprendizaje y reporte. |
| `modelo_sin_scaler(...)`, `modelo_con_scaler(...)` | `training.sgd` | Evalúan clasificadores sobre varias matrices y agrupan reportes, matrices de confusión y curvas. El estimador recibido determina si se escala o balancea. |
| `optimizar_sin_scaler(...)`, `optimizar_con_scaler(...)` | Optimización SGD | Ejecutan `GridSearchCV` por representación. Están implementadas, pero desconectadas del flujo activo del comando `sgd`. |

## Visualización — `src/training/graphics.py`

Funciones consumidas por SGD:

- `graficar_matrices_confusion(...)` muestra una grilla de matrices para comparar representaciones.
- `graficar_curvas_aprendizaje(...)` muestra curvas de aprendizaje para comparar modelos.
- `graficar_matriz_confusion(...)` y `graficar_curva_aprendizaje(...)` permiten dibujar una visualización individual y también son utilizadas internamente por las funciones de grilla.

## Estados y compatibilidad

- `config.ruta_limitada` se usa para las salidas de todos los vectorizadores. Los nombres base existentes se conservan; matriz y vocabulario reciben el mismo `_limit_N`.
- `utils.obtener_ruta_salida` se conserva porque los comandos TF-IDF la usan; su lógica reside en `config` para evitar duplicar la convención.
- La API documentada describe el uso entre módulos actual. No implica que todos los helpers sean contratos estables para consumidores externos al repositorio.
