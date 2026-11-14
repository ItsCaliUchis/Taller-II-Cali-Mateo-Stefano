# Arquitectura y datos

## Pipeline

```text
API de reseñas → dataset_raw.csv → exploración/limpieza → dataset_clean.csv
               → vectorización BoW, TF-IDF, FastText, BERT o ELMo
               → evaluación SGD (BoW, TF-IDF y FastText)
```

## Paquetes

- `src/extract/extractor.py`: descarga páginas de hasta 1000 reseñas.
- `src/clean/exploratorio.py`: imprime esquema, primeras filas, nulos, idioma inferido del ID y consistencia entre etiquetas.
- `src/clean/cleanup.py`: transforma `text` en `reviews`, normaliza espacios extremos/minúsculas y conserva `id`, `reviews`, `label` (y `Unnamed: 0` si estaba presente).
- `src/preprocessing/`: vectorizadores BoW, TF-IDF por n-gramas, FastText, BERT y ELMo.
- `src/training/sgd.py`: evaluación de clasificadores SGD y generación de reportes y gráficos.
- `src/training/rna.py`: entrenamiento MLP por minibatches sobre BOW, TF-IDF y FastText; la primera capa aprovecha matrices dispersas y Optuna busca hiperparámetros con pruning.
- `src/training/graphics.py`: funciones de visualización.
- `src/training/pca_exploratorio.py`: experimento PCA; la carga de datos ocurre al invocar `main`, no al importar.
- `src/utils.py`: limpieza lingüística, control de ejecución, carga de matrices y entrenamiento/evaluación compartidos.
- `src/config.py`: rutas centrales y carga de JSON de negaciones/resultados.
- `src/list_tools.py`: CRUD interactivo del catálogo CLI y sincronización de `pyproject.toml`.

## Archivos y formatos

| Archivo | Uso |
|---|---|
| `data/dataset_raw.csv` | Datos directos de la API |
| `data/dataset_clean.csv` | Dataset preparado; `label` preserva los valores originales 0–4 |
| `data/negaciones.json` | Palabras que se preservan al retirar stopwords inglesas |
| `data/bow_vect.npz`, `data/idf_*.npz` | Matrices dispersas SciPy |
| `data/fasttext_embeddings.npz` | Promedio de vectores de palabras por reseña, almacenado como CSR |
| `data/bert_vect.npz`, `data/elmo_vect.npz` | NPZ comprimido con clave `embeddings` (matriz densa) |
| `*_vocab.csv` | Columnas `index` y `term` del vocabulario asociado |
| `classification_reports/` | Reportes CSV de los modelos SGD |
| `optimization_results/` | Ubicación configurada para resultados de optimización |

Las rutas parten de la raíz del proyecto, detectada desde el módulo, `TALLER_PROJECT_ROOT` o el directorio actual del repositorio. Los archivos se escriben en `data/` salvo reportes e información de optimización. `config.py` crea esas carpetas al importarse y centraliza la derivación de nombres limitados en `ruta_limitada(path, limit)`. Cada ruta base (matriz y vocabulario) recibe el sufijo `_limit_N`, por lo que la convención queda alineada sin añadir variables de configuración para cada límite posible.

## Limpieza lingüística

`utils.extractor_reviews` toma la columna `reviews`, aplica minúsculas y tokenización NLTK, elimina stopwords inglesas excepto negaciones declaradas en `negaciones.json`, y descarta tokens no alfabéticos. Los recursos `punkt_tab` y `stopwords` se descargan si faltan cuando el vectorizador invoca `verificar_recursos_nltk`.

## Etiquetas y evaluación

SGD mapea 0 y 1 a `negative`, 2 a `neutral`, 3 y 4 a `positive`. Divide 80/20 con estratificación y semilla 102. Usa validación cruzada estratificada de cinco particiones para las curvas de aprendizaje y F1 macro como métrica. El código ejecuta modelos base y una segunda evaluación con parámetros fijados manualmente por embedding. El escenario que pasa por `modelo_con_scaler` añade escalado, pero no configura `class_weight="balanced"`. GridSearch no corre en el flujo activo.
