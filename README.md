# Taller II — Cali, Mateo y Stefano

Espacio de trabajo para las consignas de **Taller II — segundo cuatrimestre**.

## Estructura del proyecto

```text
Taller-II-Cali-Mateo-Stefano/
├── src/
│   ├── config.py
│   ├── list_tools.py
│   ├── utils.py
│   ├── commands.info.json
│   ├── extract/
│   │   ├── __init__.py
│   │   └── extractor.py
│   ├── clean/
│   │   ├── __init__.py
│   │   ├── cleanup.py
│   │   └── exploratorio.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── bow_embedding.py
│   │   ├── idf_ngram.py
│   │   ├── bert.py
│   │   ├── elmo.py
│   │   └── fasttext.py
│   └── training/
│      ├── __init__.py
│      ├── graphics.py
│      ├── pca_exploratorio.py
│      └── sgd.py
│
│       
├── data/
│   ├── negaciones.json
│   └── Archivos generados durante el procesamiento
├── classification_reports/
│   └── Archivos de la clase "classification reports" de Scikit Learn generados durante training de modelos
├── optimization_results/
│   └── resultados_sgd.json
├── pyproject.toml
├── poetry.lock
└── README.md

```

Los archivos de salida se generan según las rutas definidas en `src/config.py`. La carpeta `data/` se crea automáticamente al importar el módulo si todavía no existe, al igual que `classification_reports/` y `optimization_results/`

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

### Planificador configurable

`schedule` ejecuta la cola guardada en `optimization_results/plan.json`. Desde `poetry run schedule gui` puedes elegir comandos, cambiar su orden y editar sus argumentos; también puedes ejecutar el plan desde CLI con `poetry run schedule run` o `poetry run schedule start`. `--overwrite` permite regenerar salidas y `--workers N` paraleliza únicamente comandos consecutivos que comparten un grupo declarado en el catálogo. Consulta [la guía del planificador](docs/optimizacion.md).

## Flujo de procesamiento

El pipeline se organiza en tres etapas principales, que utilizan los archivos generados en las etapas anteriores como entrada para el procesamiento siguiente.

```text
API ──> [Extracción] ──> dataset_raw.csv ──> [Exploración / Limpieza] ──> dataset_clean.csv ──> [Vectorización (BoW / TF-IDF)] ──> [Entrenamiento (SGD, ...)] 

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

El proyecto implementa distintos métodos de representación vectorial de texto para transformar las reseñas en datos numéricos que puedan utilizarse en tareas de procesamiento del lenguaje natural (NLP).

- **Bag of Words (BoW):** representa cada documento mediante la frecuencia de aparición de las palabras de su vocabulario, utilizando `CountVectorizer`.
- **N-gramas con TF-IDF:** genera representaciones basadas en la frecuencia de los términos y su relevancia relativa en el corpus. Permite trabajar con monogramas y bigramas mediante `TfidfVectorizer`, configurando `ngram_range=(1, 1)` y `ngram_range=(2, 2)`, respectivamente.
- **BERT:** utiliza `bert-base-uncased` para obtener representaciones contextuales del texto mediante un modelo Transformer preentrenado.
- **ELMo:** genera representaciones contextuales de palabras mediante un modelo de lenguaje basado en redes neuronales recurrentes.
- **FastText:** obtiene representaciones vectoriales de palabras considerando información de subpalabras, lo que permite aprovechar características morfológicas y representar palabras fuera del vocabulario aprendido, según la configuración del modelo.

#### Límite de procesamiento

Los comandos que lo implementan admiten el argumento opcional `--limit`, que permite restringir la cantidad de reseñas procesadas durante las pruebas. El valor debe ser un entero positivo.

Cuando corresponde, el límite se incorpora automáticamente al nombre de los archivos de salida para distinguir los resultados parciales de los completos.

#### Formato de Salida y Uso de Matrices

El formato de exportación depende del método de vectorización:

- **BoW y TF-IDF:** las representaciones se almacenan en los formatos definidos por cada implementación. Las matrices dispersas de TF-IDF se guardan en archivos `.npz` mediante SciPy, y el vocabulario asociado se exporta por separado en CSV.
- **BERT y ELMo:** generan representaciones vectoriales densas a partir de modelos de lenguaje preentrenados.
- **FastText:** permite obtener vectores densos para palabras y, según la implementación, representaciones agregadas de documentos.

Los archivos generados y sus dimensiones dependen del método utilizado, de la configuración del modelo y de la cantidad de reseñas procesadas.

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

### 4. Entrenamiento

El módulo de entrenamiento tiene como objetivo desarrollar, entrenar y evaluar distintos modelos de aprendizaje automático supervisado para Sentiment Analysis.

A partir de diferentes representaciones vectoriales del texto, se experimenta con distintos algoritmos de clasificación para analizar su capacidad de identificar sentimientos negativos, neutrales y positivos en las reseñas.

El módulo contempla distintas etapas experimentales, que incluyen la preparación de los datos, el entrenamiento de modelos base, la evaluación de su rendimiento, la optimización de hiperparámetros y la comparación de resultados. Asimismo, se consideran diferentes configuraciones de los datos, como el escalado de características y el balance de clases, con el propósito de estudiar su influencia sobre el desempeño de los clasificadores.

- **Módulo SGD:**

Este algoritmo permite entrenar modelos de clasificación mediante actualizaciones iterativas de sus parámetros, utilizando muestras individuales o pequeños subconjuntos de datos para aproximar el gradiente de la función de pérdida.

Se trabaja con cuatro representaciones vectoriales del texto:

* BoW (Bag of Words).
* TF-IDF con unigramas.
* TF-IDF con unigramas y bigramas.
* FastText.

1) Preparación de los datos 

Se realiza el mapeo de las etiquetas originales del dataset para agrupar las cinco categorías en tres clases de sentimiento:

0: "negative",
1: "negative",
2: "neutral",
3: "positive",
4: "positive"

2) Evaluación de modelos base: 

En una primera instancia, se entrenan y evalúan los clasificadores con sus configuraciones estándar, sin realizar optimización de hiperparámetros.

Se experimenta con las cuatro representaciones vectoriales, considerando dos configuraciones:

* **Sin escalar:** se utilizan los valores originales de las representaciones vectoriales.
* **Escalado:** se aplica `StandardScaler` para analizar el efecto de la estandarización sobre el rendimiento de los clasificadores.

Se consideran tanto los conjuntos de datos balanceados como los no balanceados.

Para evaluar los resultados se generan:

- Matrices de confusión.
- `Classification Reports`, que incluyen métricas de precision, recall, f1-score y support.
- `Learning Curves`, para estudiar el comportamiento de los modelos según la cantidad de datos utilizados durante el entrenamiento.

Los resultados gráficos se organizan en grillas de 2×2, permitiendo comparar las cuatro representaciones vectoriales.

Además, los `Classification Reports` se almacenan en la carpeta `classification_reports/` del proyecto.

3) Optimización de Hiperparámetros:

Posteriormente, se incorpora un proceso de optimización mediante `GridSearchCV`, con el objetivo de encontrar las combinaciones de hiperparámetros que permiten obtener el mejor rendimiento durante la validación cruzada.

Los mejores parámetros obtenidos se almacenan en un archivo JSON en la carpeta `optimization_results/` para conservar los resultados de la búsqueda y permitir su reutilización sin necesidad de ejecutar nuevamente todo el proceso de optimización.

A partir de estos parámetros se definen nuevos clasificadores optimizados y se repite el procedimiento de entrenamiento y evaluación realizado con los modelos base.

Se mantienen las dos configuraciones experimentales:

* **Modelos optimizados sin escalado**
* **Modelos optimizados con `StandardScaler`**

De esta manera, es posible comparar el rendimiento de los clasificadores antes y después de la optimización

4) Análisis exploratorio mediante PCA

Se incorpora un análisis exploratorio de reducción de dimensionalidad mediante PCA (Principal Component Analysis), con el propósito de estudiar la posibilidad de visualizar las fronteras de decisión de los clasificadores mediante `DecisionBoundaryDisplay`.

Sin embargo, debido a la alta dimensionalidad y dispersión de las representaciones vectoriales, la proyección sobre dos componentes principales conserva una proporción reducida de la varianza original.

Por este motivo, las visualizaciones bidimensionales pueden perder información relevante de los datos y no representar adecuadamente las fronteras de decisión en el espacio original.

Este análisis se conserva como una exploración metodológica y no forma parte de la evaluación principal de los clasificadores.

5) Resultados generados

El módulo produce resultados gráficos y reportes numéricos correspondientes a las dos etapas experimentales.

Se generan ocho imágenes en total:

- Modelos base: 4 imágenes
- Modelos optimizado: 4 imágenes

Las visualizaciones permiten comparar las cuatro representaciones vectoriales, considerando el balance de clases y las configuraciones de escalado y no escalado.


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
* [x] Vectorización Bag of Words (BoW).
* [x] Vectorización TF-IDF de monogramas y bigramas.
* [x] Generación de embeddings mediante BERT, ELMo y FastText.
* [x] Argumento opcional `--limit` para pruebas con una cantidad acotada de reseñas en los comandos compatibles.
* [x] Exportación de matrices de representación en formato disperso `.npz` cuando corresponde.
* [x] Exportación de vocabularios en archivos CSV independientes.
* [x] Entrenamiento de embeddings BoW, TF-IDF y FastText por medio de método SGD.
* [x] Adición de carpetas `classification_reports` y `optimization_results` al proyecto.
* [ ] Próximas etapas de procesamiento y análisis.
