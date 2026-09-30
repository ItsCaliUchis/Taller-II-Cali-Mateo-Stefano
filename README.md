# Taller II — Cali, Mateo y Stefano

Espacio de trabajo para las consignas de **Taller II — segundo cuatrimestre**.

## Estructura del proyecto

```text
Taller-II-Cali-Mateo-Stefano/

├── src/
│   ├── config.py
│   ├── extract/
│   │   ├── __init__.py
│   │   └── extractor.py
│   └── clean/
│       ├── __init__.py
│       ├── cleanup.py
│       └── exploratorio.py
├── data/
│   ├── dataset_raw.csv
│   └── dataset_clean.csv
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

Las rutas utilizadas por los comandos se encuentran centralizadas en
`src/config.py`.

Actualmente se manejan:

* `dataset_raw`: dataset obtenido directamente desde la API.
* `dataset_clean`: dataset resultante del proceso de limpieza.

La carpeta `data/` se crea automáticamente al importar `config.py` si todavía no existe.

Esto permite modificar la ubicación de los archivos sin tener que cambiar las rutas directamente en cada script.

## Comandos del proyecto

El paquete se instala en modo editable en el entorno de Poetry. Después de
`poetry install`, se pueden ejecutar estas tareas desde la raíz del proyecto:

```bash
poetry run extractor
poetry run explore
poetry run clean
```

Los imports del paquete no dependen del directorio actual. Los archivos de
datos se guardan en la carpeta `data/` del repositorio.

## Extracción de datos

El comando `extractor` obtiene las reseñas desde la API:

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
8. Verifica si ya existe un dataset antes de iniciar la extracción.
9. Solicita confirmación antes de sobrescribir un dataset existente, salvo que se indique lo contrario mediante un argumento de CLI.

El extractor acepta los siguientes argumentos:

```bash
--size SIZE
```

Define la cantidad de reseñas solicitadas por petición.

Por defecto:

```text
1000
```

Ejemplo:

```bash
poetry run extractor --size 500
```

También permite evitar la confirmación al sobrescribir el dataset:

```bash
--overwrite
--no-confirm
```

Ambas opciones tienen el mismo efecto.

Ejemplo:

```bash
poetry run extractor --overwrite
```

Para consultar todas las opciones disponibles:

```bash
poetry run extractor --help
```

Si `dataset_raw.csv` ya existe y no se utiliza `--overwrite` ni `--no-confirm`, el extractor solicita confirmación antes de reemplazarlo.

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

El comando `explore` permite realizar una inspección inicial del dataset.

Para ejecutarlo:

```bash
poetry run explore
```

Actualmente se realizan comprobaciones sobre:

* Columnas disponibles.
* Valores faltantes.
* Idioma de las reseñas.
* Correspondencia entre `label` y `label_text`.

## Limpieza del dataset

El comando `clean` procesa el dataset obtenido de la API.

Antes de comenzar, verifica que el dataset raw exista en la ruta definida por `config.dataset_raw`. Si el archivo no existe, el proceso termina mostrando la ruta esperada.

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
poetry run clean
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
 extractor
 │
 ▼
dataset_raw.csv
 │
 ▼
 explore
 │
 ├── análisis inicial
 │
 ▼
 clean
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
poetry run extractor
```

### Ejecutar extracción con opciones

```bash
poetry run extractor --size 500
```

```bash
poetry run extractor --overwrite
```

### Consultar ayuda del extractor

```bash
poetry run extractor --help
```

### Ejecutar análisis exploratorio

```bash
poetry run explore
```

### Ejecutar limpieza

```bash
poetry run clean
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
* [x] Creación automática de la carpeta `data/`.
* [x] Verificación de existencia del dataset raw antes de la limpieza.
* [x] Confirmación antes de sobrescribir el dataset durante la extracción.
* [x] Argumentos de línea de comandos para el extractor.
* [ ] Próximas etapas de procesamiento/análisis.
