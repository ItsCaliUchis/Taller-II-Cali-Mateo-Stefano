# Comportamientos, límites y estado

## Funcionalidad implementada

- Descarga paginada y controlada de reseñas desde la API configurada.
- Inspección exploratoria y producción del dataset limpio.
- Filtrado de stopwords conservando negaciones configurables.
- Vectorización BoW, TF-IDF unigramas/bigramas, FastText, BERT y ELMo.
- `--limit` positivo en todos los vectorizadores; cada salida limitada (matriz y vocabulario) se guarda bajo un nombre separado con `_limit_N`.
- La política para derivar esas rutas está centralizada en `config.ruta_limitada`; `config.py` continúa definiendo las rutas base de los archivos completos.
- Evaluación SGD base y con parámetros fijados manualmente en cuatro representaciones, reportes CSV, matrices de confusión y curvas de aprendizaje.
- Administración del catálogo de comandos y sincronización de Poetry.

## Parcial, experimental o desconectado

- Los vectores BERT y ELMo se producen, pero SGD no los carga ni evalúa.
- Grid search y helpers de optimización existen, pero el bloque en `sgd.main()` está comentado. El JSON de resultados se configura, pero no se escribe desde ese flujo activo.
- PCA es exploratorio y no alimenta al clasificador ni produce un display de fronteras de decisión.
- El escenario llamado con escalado utiliza `StandardScaler(with_mean=False)`, pero no configura `class_weight="balanced"`; la documentación y nombres históricos pueden sugerir balanceo sin que ocurra.
- `--limit N` de vectorización es independiente para cada ejecución y los embeddings limitados no se consumen automáticamente desde SGD, que carga únicamente nombres completos.

## Riesgos de datos/ejecución conocidos

- La limpieza espera `label_text` y `text`; faltantes generan excepción. El campo de reseñas nulo tampoco se sanea.
- La descarga puede guardar un resultado parcial si recibe un HTTP no exitoso; no dispone de timeout/reintentos.
- BERT procesa cada reseña por separado y usa CLS; requiere modelo descargable/cacheado y puede superar el máximo de tokens del modelo.
- FastText usa vectores cero para palabras no recuperables. BERT/ELMo escriben arrays densos, que pueden consumir mucha memoria.
- Algunas importaciones pesadas (Transformers, PyTorch, TensorFlow/ELMo) condicionan disponibilidad del entorno para invocar esos comandos.

## Encapsulación de efectos secundarios

La importación de `training.sgd` ya no carga el dataset ni lanza trabajo. La importación de `training.pca_exploratorio` ya no llama a `verificar_embeddings`. La importación de `preprocessing.elmo` ya no imprime el mensaje de inicio. Estas acciones ocurren al entrar por `main()` o `load_elmo()` según corresponda.

## Estado de mantenimiento

El estado refleja el código revisado en esta versión y no una garantía de rendimiento. Las funciones marcadas como implementadas aún dependen de que existan datos válidos, dependencias compatibles y acceso a los modelos/servicios externos. Al cambiar comportamiento, mantener actualizados los comandos, la referencia de funciones y esta página.
