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
- Downloader para la carpeta pública y plana de Drive: crea el inventario local y compara archivos usando `ListadoRemoto.json` y SHA-256; `drive bundle-rna` crea un ZIP de los modelos RNA finales y sus reportes para archivado/transferencia.
- Ejecución programada con plan editable, validación de artefactos, omisión segura de salidas existentes, grupos de concurrencia contiguos y checkpoints por representación para SGD.

## Parcial, experimental o desconectado

- RNA y SGD evalúan BoW, TF-IDF y FastText; BERT y ELMo quedan fuera del flujo RNA.
- Grid search y helpers de optimización existen, pero el bloque en `sgd.main()` está comentado. El JSON de resultados se configura, pero no se escribe desde ese flujo activo.
- PCA es exploratorio y no alimenta al clasificador ni produce un display de fronteras de decisión.
- El escenario llamado con escalado utiliza `StandardScaler(with_mean=False)`, pero no configura `class_weight="balanced"`; la documentación y nombres históricos pueden sugerir balanceo sin que ocurra.
- `--limit N` de vectorización es independiente para cada ejecución y los embeddings limitados no se consumen automáticamente desde SGD, que carga únicamente nombres completos.
- El modo estándar de SGD usa 3 folds y 3 tamaños para curvas de aprendizaje; `--full-evaluation` conserva la configuración anterior de 5 por 5 a costa de más tiempo.

## Riesgos de datos/ejecución conocidos

- La limpieza espera `label_text` y `text`; faltantes generan excepción. El campo de reseñas nulo tampoco se sanea.
- Una descarga incompleta no se guarda como dataset válido; la extracción no dispone de timeout/reintentos.
- BERT procesa cada reseña por separado y usa CLS; requiere modelo descargable/cacheado y puede superar el máximo de tokens del modelo.
- FastText usa vectores cero para palabras no recuperables. BERT/ELMo escriben arrays densos, que pueden consumir mucha memoria.
- Algunas importaciones pesadas (Transformers, PyTorch, TensorFlow/ELMo) condicionan disponibilidad del entorno para invocar esos comandos.

## Encapsulación de efectos secundarios

La importación de `training.sgd` ya no carga el dataset ni lanza trabajo. La importación de `training.pca_exploratorio` ya no llama a `verificar_embeddings`. La importación de `preprocessing.elmo` ya no imprime el mensaje de inicio. Estas acciones ocurren al entrar por `main()` o `load_elmo()` según corresponda.

## Estado de mantenimiento

Cada `schedule start` o `schedule run` elimina primero los logs y el estado JSON anteriores del scheduler; conserva el plan, matrices, datasets, checkpoints y otros resultados. El plan se guarda en `optimization_results/plan.json`; la GUI permite seleccionar comandos, ordenarlos y editar sus argumentos. La ejecución respeta dependencias y solo paraleliza comandos contiguos que comparten `parallel_group`, hasta el límite de `--workers`. Omite una etapa si todas sus salidas están presentes y son válidas. Si solo algunas existen o alguna salida es inválida, bloquea la etapa para impedir que se sobrescriban datos accidentalmente; `--overwrite` permite regenerarlas. La hora registrada corresponde a Buenos Aires (UTC−3).

El downloader requiere `ListadoRemoto.json` en la carpeta pública. Primero crea `data/ListadoDescarga.json` desde el inventario visible y luego descarga y valida el manifiesto. Este debe incluir los IDs de Drive, tamaños y SHA-256. Los archivos se descargan a temporales, se verifican y solo entonces se mueven a `data/`; los faltantes/distintos se descargan por defecto y `--overwrite` fuerza todos. Nunca se eliminan archivos locales. La fecha de descarga es local y no se comparte en el manifiesto remoto. Drive debe ser plano: los modelos e informes RNA se pueden agrupar con `drive bundle-rna`, que produce `data/rna_modelos_finales.zip`; la subida y actualización del manifiesto siguen siendo manuales.

El estado refleja el código revisado en esta versión y no una garantía de rendimiento. Las funciones marcadas como implementadas aún dependen de que existan datos válidos, dependencias compatibles y acceso a los modelos/servicios externos. Al cambiar comportamiento, mantener actualizados los comandos, la referencia de funciones y esta página.

## To do:

Lo proximo a realizar:

- Persistir los vectorizadores TF-IDF monograma y bigrama (vocabulario, pesos `idf_` y configuración) para transformar reseñas nuevas en inferencia. Para reutilizar modelos RNA ya entrenados, volver a ajustar y guardar los vectorizadores sobre el mismo `dataset_clean.csv` y configuración; esto no requiere reentrenar la RNA. En inferencia, cargar el vectorizador guardado y usar `transform()`.
