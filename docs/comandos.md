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
| `list` | `list_tools:main` | `--all`, `--add`, `--modify NAME`, `--remove NAME`, `--sync`; `--include` solo con add/modify. |

La validación de `--limit` exige entero positivo. Omitirlo procesa todas las filas y escribe el nombre estándar. En modo limitado, los nombres son por ejemplo `idf_bigram_limit_500.npz` y `idf_bigram_limit_500_vocab.csv`.

## Consideraciones

- El limitador de extracción se expresa en miles, distinto del `--limit` de los vectorizadores que expresa filas.
- Los embeddings limitados quedan separados de los completos. `sgd` actualmente no selecciona automáticamente matrices `*_limit_N`; espera que las matrices tengan la misma cantidad de filas que las etiquetas.
- `bert` requiere acceso a los archivos del modelo de Hugging Face si aún no están en caché. BERT procesa secuencias con los valores predeterminados del tokenizador y puede fallar con reseñas que excedan el máximo admitido.
- FastText descarga el modelo `fasttext-wiki-news-subwords-300` mediante Gensim si no está cacheado. ELMo descarga pesos mediante HTTP tras confirmación.
- `list --add` y `list --modify` son flujos interactivos. El catálogo guarda descripción, estado, entrypoint, include y compatibilidad con ayuda.
