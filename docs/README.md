# Documentación del proyecto

## Propósito

Pipeline académico para análisis de sentimiento de reseñas de Amazon. Descarga registros desde una API, los explora y limpia, convierte reseñas a matrices numéricas y evalúa clasificadores SGD en tres clases (`negative`, `neutral`, `positive`). Incluye además una CLI para mantener los comandos de Poetry sincronizados con un catálogo JSON.

## Recorrido recomendado

1. Instalar dependencias con `poetry install` (Python 3.13 o superior).
2. `poetry run extractor [--size N] [--overwrite]` crea `data/dataset_raw.csv`.
3. `poetry run explore` inspecciona el dataset raw; `poetry run clean` crea `data/dataset_clean.csv`.
4. Ejecutar los vectorizadores requeridos: `bow`, `idfMonogram`, `idfBigram`, `fasttext`, opcionalmente `bert` o `elmo`.
5. `poetry run sgd` entrena y evalúa cuatro matrices: BoW, TF-IDF bigrama, TF-IDF monograma y FastText.

Cada vectorizador acepta `--limit N` para procesar las primeras N filas del dataset limpio. Los artefactos limitados llevan `_limit_N` en su nombre (incluido el CSV de vocabulario) y no reemplazan las salidas completas. Ejemplo: `poetry run idfBigram --limit 500`.

## Documentos

- [Arquitectura, datos y flujo](arquitectura.md)
- [Comandos CLI](comandos.md)
- [Referencia de funciones y estado](referencia.md)
- [Comportamientos, límites y pendientes](estado.md)

## Estado en breve

Las rutas se definen en `src/config.py` relativas al repositorio, no al directorio de ejecución. Los vectorizadores BERT y ELMo guardan matrices densas dentro de archivos NPZ; el resto guarda matrices dispersas SciPy. SGD actualmente consume únicamente matrices completas y BoW/TF-IDF/FastText. Evalúa modelos base y configuraciones con parámetros fijados manualmente; GridSearch está disponible en utilidades, pero no forma parte de la ejecución activa del comando `sgd`.
