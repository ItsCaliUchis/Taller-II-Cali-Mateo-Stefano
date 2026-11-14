# Comandos de línea de comandos

Todos se ejecutan con `poetry run`. El catálogo canónico está en `src/commands.info.json`; `list_tools.py` sincroniza scripts activos y obsoletos hacia `pyproject.toml` y paquetes hacia la configuración Poetry.

| Comando | Entrada | Opciones y efecto |
|---|---|---|
| `extractor` | `extract.extractor:main` | `--size N`: hasta N×1000 filas (el último bloque se recorta); `--overwrite`/`--no-confirm`: omite confirmación si ya existe el CSV. |
| `explore` | `clean.exploratorio:main` | Sin opciones; analiza `dataset_raw.csv`. |
| `clean` | `clean.cleanup:main` | Sin opciones; lee raw y escribe dataset limpio. |
| `bow` | `preprocessing.bow_embedding:main` | `--limit N`; genera BoW NPZ y vocabulario CSV. |
| `idfMonogram` | `preprocessing.idf_ngram:monograma` | `--limit N`; TF-IDF unigramas. |
| `idfBigram` | `preprocessing.idf_ngram:bigrama` | `--limit N`; TF-IDF bigramas. |
| `fasttext` | `preprocessing.fasttext:main` | `--limit N`; vectores FastText promediados por reseña. |
| `bert` | `preprocessing.bert:main` | `--limit N`; vector CLS de `bert-base-uncased`. |
| `elmo` | `preprocessing.elmo:main` | `--limit N`; vector promedio ELMo; pregunta antes de descargar pesos si no existen. |
| `sgd` | `training.sgd:main` | Sin opciones; evalúa modelos base y con parámetros fijados manualmente, genera reportes CSV y abre gráficos interactivos. Requiere dataset limpio y los cuatro embeddings completos; genera automáticamente los faltantes. |
| `rna` | `training.rna:main` | MLP sobre BOW, TF-IDF monograma/bigrama y FastText; `--embedding all` ejecuta los cuatro por defecto. Evalúa con/sin escalado y optimiza hiperparámetros con Optuna pruning. Las matrices dispersas sobre 100.000 columnas se reducen con chi2 usando solo train (`--max-features`). Admite `--device auto|cuda|cpu`, `--optimization-trials`, `--batch-size`, `--overwrite` y `--restart`. |
| `schedule` | `schedule:main` | `start`/`run` ejecutan `optimization_results/plan.json`; admiten `--plan RUTA`, `--overwrite`, `--workers N`, `--sgd-jobs N`, `--full-evaluation` y `--skip-contextual`. `plan`, `status`, `stop` y `gui` administran la cola y la ejecución. |
| `drive` | `drive_downloader.cli:main` | `list`, `--download`, `--overwrite`, `gui` y `manifest`; compara por SHA-256, actualiza SHA locales y genera `ListadoRemoto.json` desde `ListadoDescarga.json`. |
| `list` | `list_tools:main` | `--all`, `--add`, `--modify NAME`, `--remove NAME`, `--sync`; `--include` solo con add/modify. |

La validación de `--limit` exige entero positivo. Omitirlo procesa todas las filas y escribe el nombre estándar. En modo limitado, los nombres son por ejemplo `idf_bigram_limit_500.npz` y `idf_bigram_limit_500_vocab.csv`.

## Consideraciones

- El limitador de extracción se expresa en miles, distinto del `--limit` de los vectorizadores que expresa filas.
- Los embeddings limitados quedan separados de los completos. `sgd` actualmente no selecciona automáticamente matrices `*_limit_N`; espera que las matrices tengan la misma cantidad de filas que las etiquetas.
- `bert` requiere acceso a los archivos del modelo de Hugging Face si aún no están en caché. BERT procesa secuencias con los valores predeterminados del tokenizador y puede fallar con reseñas que excedan el máximo admitido.
- FastText descarga el modelo `fasttext-wiki-news-subwords-300` mediante Gensim si no está cacheado. ELMo descarga pesos mediante HTTP tras confirmación.
- `list --add` y `list --modify` son flujos interactivos. El catálogo guarda descripción, estado, entrypoint, include, compatibilidad con ayuda y metadata opcional de schedule: salidas, dependencias, grupo, argumentos por defecto y argumento de overwrite.

`rna` utiliza las mismas etiquetas de sentimiento que SGD (0–1 negativa, 2 neutral, 3–4 positiva). Si `dataset_clean.csv` no tiene una columna de etiqueta de tres clases, crea un mapa cacheado en `optimization_results/training_cache/` sin modificar el CSV fuente. Los reportes quedan organizados en `classification_reports/rna/<representacion>/`. Los checkpoints por corrida se guardan en `optimization_results/rna_checkpoints/`; `--overwrite` regenera resultados y reanuda un checkpoint compatible, mientras que `--restart` fuerza entrenamiento desde cero. Si se usa `--limit`, las matrices y etiquetas deben compartir la misma selección y orden de filas.

Para uso desatendido, inicia con `poetry run schedule start`. El plan se guarda localmente en `optimization_results/plan.json`; la GUI permite editar selección, orden y argumentos. `--workers N` solo paraleliza comandos consecutivos con el mismo grupo. Cada ejecución nueva limpia logs y estado del scheduler, sin borrar artefactos de datos. Consulta [Optimización](optimizacion.md).
