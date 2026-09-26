# Taller II — Cali, Mateo y Stefano

Espacio de trabajo para las consignas de **Taller II — segundo cuatrimestre**.

## Estructura del proyecto

```text
Taller-II-Cali-Mateo-Stefano/
├── config.py
├── data/
│   ├── dataset_raw.csv
│   └── dataset_clean.csv
├── extractor/
│   ├── __init__.py
│   └── extractor.py
├── clean_up/
│   ├── __init__.py
│   ├── cleanup.py
│   └── exploratorio.py
├── pyproject.toml
├── poetry.lock
└── README.md
```

## Requisitos

* Python 3.13 o superior
* Poetry

Las dependencias del proyecto se encuentran declaradas en `pyproject.toml` y bloqueadas mediante `poetry.lock`.

## Instalación

Clonar el repositorio y ubicarse en la carpeta del proyecto:

```bash
git clone <URL_DEL_REPOSITORIO>
cd Taller-II-Cali-Mateo-Stefano
```

Instalar las dependencias mediante Poetry:

```bash
poetry install
```

El proyecto utiliza Python 3.13.

## Configuración

Las rutas utilizadas por los scripts se encuentran centralizadas en `config.py`.

Actualmente se manejan:

* `dataset_raw`: dataset obtenido directamente desde la API.
* `dataset_clean`: dataset resultante del proceso de limpieza.

Esto permite modificar la ubicación de los archivos sin tener que cambiar las rutas directamente en cada script.

## Extracción de datos

El módulo `extractor/extractor.py` obtiene las reseñas desde la API:

```text
https://amazon-reviews-api-g5ae.onrender.com/reviews
```

La extracción se realiza mediante solicitudes paginadas utilizando `limit` y `offset`.

Actualmente el extractor:

1. Solicita las reseñas a la API.
2. Obtiene la cantidad total de registros.
3. Descarga los registros por bloques.
4. Muestra el progreso de la descarga.
5. Muestra velocidad de extracción y tiempo estimado restante.
6. Construye un `DataFrame` de Pandas.
7. Guarda el resultado como CSV sin almacenar el índice de Pandas.

Para ejecutar el extractor:

```bash
poetry run python -m extractor.extractor
```

El dataset obtenido se guarda en la ubicación definida por:

```python
config.dataset_raw
```

### Nota sobre `Unnamed: 0`

El extractor utiliza:

```python
dataset.to_csv(config.dataset_raw, index=False)
```

Esto evita guardar el índice interno de Pandas como una columna adicional.

En versiones anteriores se utilizaba `to_csv()` sin `index=False`, lo que provocaba que al volver a leer el CSV apareciera una columna `Unnamed: 0`.

El proceso de limpieza mantiene compatibilidad con datasets antiguos que todavía contengan dicha columna.

## Análisis exploratorio

El módulo `clean_up/exploratorio.py` permite realizar una inspección inicial del dataset.

Para ejecutarlo:

```bash
poetry run python -m clean_up.exploratorio
```

Actualmente se realizan comprobaciones sobre:

* Columnas disponibles.
* Valores faltantes.
* Idioma de las reseñas.
* Correspondencia entre `label` y `label_text`.

## Limpieza del dataset

El módulo `clean_up/cleanup.py` procesa el dataset obtenido de la API.

Actualmente:

1. Elimina la columna `label_text`.
2. Limpia el texto de las reseñas:
   * elimina espacios al principio y al final;
   * convierte el texto a minúsculas.
3. Renombra conceptualmente el contenido de `text` como `reviews`.
4. Elimina la columna original `text`.
5. Conserva las columnas relevantes:
   * `id`
   * `reviews`
   * `label`
6. Guarda el dataset limpio como CSV.

Para ejecutar la limpieza:

```bash
poetry run python -m clean_up.cleanup
```

El resultado se guarda en:

```python
config.dataset_clean
```

## Flujo de trabajo actual

El procesamiento de los datos se realiza en dos etapas principales:

```text
API
 │
 ▼
extractor/extractor.py
 │
 ▼
dataset_raw.csv
 │
 ▼
clean_up/exploratorio.py
 │
 ├── análisis inicial
 │
 ▼
clean_up/cleanup.py
 │
 ▼
dataset_clean.csv
```

El dataset **raw** conserva los datos obtenidos de la API, mientras que el dataset **clean** contiene los datos preparados para las etapas posteriores del proyecto.

## Comandos principales

### Instalar dependencias

```bash
poetry install
```

### Verificar Python

```bash
poetry run python --version
```

### Ejecutar extracción

```bash
poetry run python -m extractor.extractor
```

### Ejecutar análisis exploratorio

```bash
poetry run python -m clean_up.exploratorio
```

### Ejecutar limpieza

```bash
poetry run python -m clean_up.cleanup
```

## Estado actual

* [x] Configuración del entorno con Poetry.
* [x] Extracción de datos desde la API.
* [x] Extracción paginada mediante `limit` y `offset`.
* [x] Indicador de progreso durante la extracción.
* [x] Generación del dataset raw.
* [x] Análisis exploratorio inicial.
* [x] Limpieza básica de las reseñas.
* [x] Generación del dataset limpio.
* [x] Separación de configuración y rutas mediante `config.py`.
* [ ] Próximas etapas de procesamiento/análisis.
