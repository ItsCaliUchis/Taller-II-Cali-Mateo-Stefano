# Documentación del proyecto

## Propósito

Pipeline académico para análisis de sentimiento de reseñas de Amazon. Descarga registros desde una API, los explora y limpia, convierte reseñas a matrices numéricas y evalúa clasificadores SGD en tres clases (`negative`, `neutral`, `positive`). Incluye además una CLI para mantener los comandos de Poetry sincronizados con un catálogo JSON.

## Recorrido recomendado

1. Instalar dependencias con `poetry install` (Python 3.13 o superior).
2. `poetry run extractor [--size N] [--overwrite]` crea `data/dataset_raw.csv`.
3. `poetry run explore` inspecciona el dataset raw; `poetry run clean` crea `data/dataset_clean.csv`.
4. Ejecutar los vectorizadores requeridos: `bow`, `idfMonogram`, `idfBigram`, `fasttext`, opcionalmente `bert` o `elmo`.
5. `poetry run sgd` entrena y evalúa cuatro matrices: BoW, TF-IDF bigrama, TF-IDF monograma y FastText.

Para ejecutar el plan guardado en segundo plano, usa `poetry run schedule start`; `optimization_results/plan.json` determina selección, orden y argumentos. La GUI permite editar esa cola. `--workers N` paraleliza solo comandos consecutivos del mismo grupo; las salidas existentes se conservan, salvo que indiques `--overwrite`. Consulta [Optimización y ejecución programada](optimizacion.md).

Para listar y descargar los artefactos publicados en Drive, usa `poetry run drive list`, `poetry run drive --download` o `poetry run drive gui`. Usa `poetry run drive manifest` para generar el manifiesto remoto y actualizar los SHA locales del inventario; consulta [Descargas desde Drive](drive_downloader.md) para su formato.

Cada vectorizador acepta `--limit N` para procesar las primeras N filas del dataset limpio. Los artefactos limitados llevan `_limit_N` en su nombre (incluido el CSV de vocabulario) y no reemplazan las salidas completas. Ejemplo: `poetry run idfBigram --limit 500`.

## Documentos

- [Arquitectura, datos y flujo](arquitectura.md)
- [Comandos CLI](comandos.md)
- [Referencia de funciones y estado](referencia.md)
- [Comportamientos, límites y pendientes](estado.md)
- [Descargas desde Drive](drive_downloader.md)
- [Optimización y ejecución programada](optimizacion.md)

## Estado en breve

Las rutas se resuelven contra la raíz del repositorio incluso al usar los comandos instalados por Poetry. Los artefactos de datos `.csv`, `.npz` y `.zip` están configurados para Git LFS; los detalles y las cuotas están en [Optimización](optimizacion.md). Los vectorizadores BERT y ELMo guardan matrices densas dentro de archivos NPZ; el resto guarda matrices dispersas SciPy. SGD consume únicamente matrices completas y BoW/TF-IDF/FastText. Evalúa modelos base y configuraciones con parámetros fijados manualmente; GridSearch está disponible en utilidades, pero no forma parte de la ejecución activa del comando `sgd`.
