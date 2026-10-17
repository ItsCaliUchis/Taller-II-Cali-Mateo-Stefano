# Taller II — Cali, Mateo y Stefano

Espacio de trabajo para las consignas de **Taller II — segundo cuatrimestre**.

## Estructura del proyecto

```text
Taller-II-Cali-Mateo-Stefano/
├── src/
│   ├── config.py
│   ├── list_tools.py
│   ├── commands.info.json
│   ├── extract/
│   │   ├── __init__.py
│   │   └── extractor.py
│   ├── clean/
│   │   ├── __init__.py
│   │   ├── cleanup.py
│   │   └── exploratorio.py
│   └── preprocessing/
│       ├── __init__.py
│       ├── bow_embedding.py
│       ├── idf_ngram.py
│       └── utils.py
├── data/
│   ├── negaciones.json
│   └── Archivos generados durante el procesamiento
├── pyproject.toml
├── poetry.lock
└── README.md

```

Los archivos de salida se generan según las rutas definidas en `src/config.py`. La carpeta `data/` se crea automáticamente al importar el módulo si todavía no existe.

---

## Requisitos e instalación

* **Python 3.13 o superior**
* **Poetry**

Clonar el repositorio y ubicarse en su directorio raíz:

```bash
git clone <URL_DEL_REPOSITORIO>
cd Taller-II-Cali-Mateo-Stefano

```

Instalar las dependencias y el paquete en modo editable:

```bash
poetry install

```

Verificar la versión de Python configurada:

```bash
poetry run python --version

```

---

## Configuración y Gestión de Herramientas CLI

Las rutas de los archivos se centralizan en `src/config.py` (dataset raw, dataset clean, vectores BoW y salidas TF-IDF), permitiendo cambiar ubicaciones sin alterar la lógica de los scripts.

### Gestor de herramientas CLI

El proyecto incorpora un gestor centralizado que actúa como un **CRUD de comandos** (permite registrar, listar, modificar y eliminar herramientas). Su propósito principal es **sincronizar automáticamente el catálogo de comandos con `pyproject.toml`**, evitando la administración manual de cada entrada.

* **Listar comandos expuestos:** Puedes consultar todas las herramientas activas ejecutando:
```bash
poetry run list

```


* **Consultar ayuda:** Cada comando declara si admite la opción de ayuda. Para ver las opciones y parámetros específicos de un comando, puedes hacer:
```bash
poetry run <comando> --help

```



*(Nota: Este diseño permite mantener el README ágil, evitando detallar manualmente todos los argumentos de cada función. Para conocer los argumentos y opciones disponibles, consultar la ayuda de cada comando.).*

---

## Flujo de procesamiento

El pipeline se organiza en tres etapas principales, que utilizan los archivos generados en las etapas anteriores como entrada para el procesamiento siguiente.

```text
API ──> [Extracción] ──> dataset_raw.csv ──> [Exploración / Limpieza] ──> dataset_clean.csv ──> [Vectorización (BoW / TF-IDF)]

```

### 1. Extracción de datos

Obtiene las reseñas desde la API (`https://amazon-reviews-api-g5ae.onrender.com/reviews`) mediante solicitudes paginadas con `limit` y `offset` en bloques de 1000 registros:

* Muestra el progreso de descarga, velocidad y tiempo estimado restante.
* Permite limitar la descarga mediante `--size` (en miles de registros).
* Verifica la existencia previa del dataset y solicita confirmación antes de sobrescribirlo (salvo que se indique `--overwrite` o `--no-confirm`).
* Previene columnas duplicadas de índices (`Unnamed: 0`).

### 2. Análisis exploratorio y Limpieza

* **Análisis exploratorio:** Inspecciona columnas, valores faltantes, idioma y correspondencia de etiquetas (`label` y `label_text`).
* **Limpieza:** Elimina columnas redundantes (`label_text`, `text`), normaliza textos (espacios, minúsculas), renombra campos clave (`text` a `reviews`) y conserva únicamente las columnas relevantes (`id`, `reviews`, `label`).

### 3. Vectorización y Representación de Texto

Las etapas de vectorización reutilizan utilidades comunes (`src/preprocessing/utils.py`), incluyendo el filtrado de *stopwords* en inglés mediante NLTK **preservando las negaciones** configuradas en el proyecto para evitar invertir el significado de las reseñas.

Cuenta con un **Watchdog de ejecución** que observa el rendimiento (tiempo transcurrido y consumo de memoria RAM en la terminal mediante una animación) durante las operaciones de procesamiento.

#### Métodos de Vectorización

* **Bag of Words (BoW):** Conteo de frecuencias mediante `CountVectorizer`, exportado a CSV.
* **TF-IDF de Monogramas y Bigramas:** Ponderación de términos según su frecuencia en cada documento y su distribución global.
* Configuración: Monogramas (`ngram_range=(1, 1)`) y Bigramas (`ngram_range=(2, 2)`).
* **Límite de procesamiento:** Los comandos admiten el argumento opcional `--limit` (entero positivo) para realizar pruebas con una cantidad acotada de reseñas. Cuando se utiliza, el límite se incorpora automáticamente a los nombres de salida para distinguir los resultados parciales de los completos. (Consultar disponibilidad de dicho argumento opcional)


#### Formato de Salida y Uso de Matrices Dispersas

Las matrices de texto suelen contener muchos ceros, ya que cada reseña utiliza solo una pequeña parte del vocabulario global. Exportar una matriz completa a un CSV ancho requeriría escribir cientos de miles de millones de celdas. Para solucionarlo, **las matrices TF-IDF se guardan en formato `.npz` utilizando matrices dispersas de SciPy**, mientras que el vocabulario se exporta por separado en un archivo CSV.

Ejemplo de ejecución completa:

```text
data/
├── idf_monogram.npz
├── idf_monogram_vocab.csv
├── idf_bigram.npz
└── idf_bigram_vocab.csv

```

Ejemplo de ejecución limitada a 1000 reseñas:

```text
data/
├── idf_bigram_limit_1000.npz
└── idf_bigram_limit_1000_vocab.csv

```

El archivo de vocabulario contiene dos columnas: `index` (posición de la columna en la matriz) y `term` (término o bigrama asociado).

---

## Estado actual del proyecto

* [x] Configuración del entorno con Poetry.
* [x] Centralización de rutas mediante `config.py`.
* [x] Creación automática de la carpeta `data/`.
* [x] Extracción paginada desde la API con indicador de progreso, velocidad y tiempo estimado.
* [x] Confirmación antes de sobrescribir el dataset raw.
* [x] Análisis exploratorio inicial.
* [x] Limpieza y generación del dataset limpio.
* [x] Paquete de preprocesamiento con manejo inteligente de negaciones.
* [x] Gestor de herramientas CLI (CRUD para registrar, modificar, consultar y sincronizar comandos con `pyproject.toml`).
* [x] Watchdog de tiempo y uso de RAM para las etapas de procesamiento.
* [x] Vectorización BoW.
* [x] TF-IDF de monogramas y bigramas.
* [x] Argumento `--limit` para pruebas con cantidad acotada de reseñas.
* [x] Exportación de matrices TF-IDF en formato disperso `.npz`.
* [x] Exportación de vocabularios TF-IDF en archivos CSV separados.
* [ ] Incorporación de otros métodos de representación (ELMo, FastText y BERT).
* [ ] Próximas etapas de procesamiento y análisis.