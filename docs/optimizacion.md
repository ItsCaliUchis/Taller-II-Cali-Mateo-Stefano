# Optimización y ejecución programada

## Comando `schedule`

`schedule` ejecuta etapas como procesos independientes, escribe logs separados y guarda el estado en `optimization_results/pipeline_state.json`.

```bash
poetry run schedule plan
poetry run schedule start --workers 2
poetry run schedule run --overwrite
poetry run schedule status
poetry run schedule gui
poetry run schedule stop
```

- `start` lanza el coordinador en segundo plano en Windows y retorna el control a la terminal.
- `run` ejecuta el plan en primer plano; `start` y `run` aceptan `--overwrite` y `--plan RUTA`.
- `optimization_results/plan.json` guarda localmente la selección, el orden y los argumentos; está excluido de Git. Si aún no existe, se crea desde los comandos activos que tienen metadata `schedule` en `src/commands.info.json`.
- `plan` muestra la cola guardada. También puedes editar el JSON directamente; cada elemento usa `{"name": "bow", "args": ["--limit", "500"]}`.
- La GUI permite marcar comandos, reordenarlos arrastrando, editar sus argumentos al hacer clic en la columna correspondiente y guardar el plan. Al iniciar desde la GUI, los cambios se guardan automáticamente.
- `status` muestra estado, PID y progreso por etapa.
- `stop` pide una parada controlada al terminar la operación activa. Una corrida nueva vuelve a validar los artefactos existentes.
- `gui` muestra el plan, la ejecución actual y el log de las etapas activas; incluye controles de overwrite, evaluación SGD completa y cantidad de workers.
- Al terminar o fallar una ejecución en Windows, `schedule` muestra una notificación en el área de notificación. `--no-notify` desactiva el aviso.
- Cada `start` o `run` nuevo borra el estado JSON y los logs creados por `schedule` en `optimization_results/`, y empieza un historial limpio. Los datasets, matrices, checkpoints y otros resultados del pipeline se conservan.
- Si Windows mantiene un log anterior abierto, el nuevo proceso escribe en una carpeta de logs exclusiva para no fallar al iniciar; se vuelve a intentar limpiar los logs anteriores en una corrida posterior. Al detener una etapa, el coordinador termina también los procesos descendientes de Poetry para liberar esos archivos.
- El orden y las etapas se toman del plan, no de una secuencia fija en `schedule.py`. `--skip-contextual` omite BERT y ELMo de esa ejecución.
- Cada subcomando se inicia con su nombre de CLI registrado en Poetry. Antes de ejecutarlo, `schedule` revisa sus salidas: si todas existen y son válidas, omite la etapa; si existe solo una parte o hay archivos inválidos, la bloquea para evitar sobrescribirlos. `--overwrite` autoriza regenerar las salidas y transmite al comando el argumento de overwrite declarado en el catálogo. También valida códigos de retorno, archivos producidos y punteros LFS sin contenido.
- Las marcas de tiempo del estado usan el horario de Buenos Aires, UTC−3.
- Los mensajes de consola usan caracteres ASCII y muestran las rutas de log relativas a la raiz del repositorio.

### Grupos y workers

`--workers N` define el máximo de comandos simultáneos dentro de un grupo consecutivo. Los comandos sin `parallel_group` se ejecutan solos. Cuando un comando tiene grupo, el scheduler puede ejecutar en paralelo los comandos inmediatamente posteriores que tengan el mismo grupo; una etapa de otro grupo corta ese lote. Por eso, los comandos deben aparecer contiguos en el plan para aprovechar workers juntos. Una dependencia entre comandos de un mismo lote crea una barrera antes de ejecutar el dependiente.

Por ejemplo, con `--workers 2` y este plan:

```text
extractor (sin grupo)
clean (prepare)
elmo (embeddings)
bert (embeddings)
explore (prepare)
bow (embeddings)
```

la ejecución sigue `extractor`, `clean`, `elmo + bert`, `explore`, `bow`. Aunque `clean` y `explore` compartan grupo, no son contiguos y no se juntan. El valor predeterminado es `--workers 1`.

La metadata de cada comando vive en `commands.info.json`: `outputs` contiene rutas relativas al proyecto, `depends_on` nombres de comandos, `parallel_group` el grupo opcional, `default_args` los argumentos iniciales y `overwrite_arg` el argumento que permite regenerar salidas. `outputs_by_argument` y `depends_on_by_argument` permiten declarar artefactos y dependencias condicionales según una opción CLI; `allow_partial_outputs` habilita comandos con reanudación interna y `restart_arg` identifica un reinicio explícito. `list --add` y `list --modify` solicitan estos datos.

Ejemplo de ejecución sin los modelos contextuales:

```bash
poetry run schedule start --skip-contextual --workers 2
```

Para regenerar todos los artefactos producidos por las etapas incluidas, solicita sobrescritura explícita:

```bash
poetry run schedule start --overwrite
poetry run schedule run --overwrite
```

## Tiempo y checkpoints de SGD

El trabajo dominante era repetir curvas de aprendizaje para 16 configuraciones de modelo. Antes cada curva usaba 5 folds × 5 tamaños (25 ajustes); el modo estándar usa 3 × 3 (9 ajustes). Esto reduce el número de ajustes de las curvas, pero sus métricas no son directamente comparables con una corrida 5×5. `poetry run sgd --full-evaluation` recupera 5 folds × 5 tamaños.

Las curvas se calculan con un proceso (`--jobs 1`) por defecto, para dejar capacidad de CPU disponible. `--jobs N` ajusta la concurrencia de scikit-learn. El scheduler corre los comandos con prioridad inferior en Windows para que las aplicaciones interactivas tengan preferencia.

Al terminar cada representación, SGD guarda su reporte, matriz de confusión y curva en `optimization_results/sgd_checkpoints/`. Si el proceso se interrumpe, vuelve a usar los checkpoints completados y sigue por el primer modelo pendiente. El identificador del checkpoint incorpora fechas de modificación y tamaños de dataset/embeddings, además de la configuración de evaluación, para evitar reutilizar resultados tras cambiar las entradas.

El comando programado añade `--no-plots`, ya que no hay una sesión gráfica que atender en segundo plano. Los reportes CSV se siguen generando. En una ejecución manual, SGD conserva las gráficas salvo que se indique `--no-plots`.

Los comandos de las etapas devuelven código 0 solo al completar correctamente y código distinto de cero cuando falta una entrada, se cancela una operación o falla la generación. De este modo `schedule` puede detener la secuencia en la primera etapa fallida y mostrar el log correspondiente.

## RNA sobre embeddings dispersos y densos

`rna` entrena un MLP por reseña para BOW, TF-IDF monograma, TF-IDF bigrama y FastText. `--embedding all` procesa esas cuatro representaciones; también se puede seleccionar una individualmente. Por cada una compara el baseline con la mejor configuración encontrada por Optuna, con y sin escalado. Optuna maximiza F1 macro de validación y usa MedianPruner para detener pruebas poco prometedoras; el conjunto de test se reserva para la evaluación final.

```bash
poetry run rna --embedding all --device auto
poetry run rna --embedding bow --device cuda --optimization-trials 10
poetry run rna --overwrite
poetry run rna --overwrite --restart
```

Las matrices de vocabulario se mantienen dispersas en la primera capa y se preparan por minibatch para evitar densificar todo el corpus. Si una matriz dispersa supera las 100.000 columnas, RNA conserva las características con mejor chi-cuadrado respecto a las etiquetas del conjunto train; el mismo selector se aplica a validación y test y queda cacheado. `--max-features` cambia ese límite. El batch por defecto es 2048; `--batch-size` permite ajustarlo. CUDA se usa cuando está disponible, y el log informa dispositivo y VRAM asignada/reservada. Los checkpoints de los entrenamientos finales se eliminan después de guardar sus artefactos completos; los trials de Optuna son efímeros.

Los reportes, matrices, historiales y curvas se guardan en `classification_reports/rna/<representacion>/`. Los checkpoints de los entrenamientos finales quedan bajo `optimization_results/rna_checkpoints/`, los modelos y metadatos bajo `optimization_results/rna_results/` y los estudios Optuna en SQLite bajo `optimization_results/rna_results/optuna/`. Al terminar, RNA borra el checkpoint de esa corrida; los checkpoints que queden de interrupciones anteriores solo reanudan la corrida compatible y pueden ocupar mucho espacio. Los trials de búsqueda no guardan checkpoints por época: SQLite conserva sus resultados y parámetros. Si un trial falla, el siguiente inicio continúa con los trials pendientes; el trial fallido no se repite. `--overwrite` regenera artefactos y continúa checkpoints compatibles de entrenamientos finales. `--restart` descarta el checkpoint y vuelve a entrenar. `--scale` o `--no-scale` ejecutan una sola modalidad de escalado; por defecto se generan ambas.

Las cargas de etiquetas/matrices, entrenamiento y optimización usan `ejecutar_etapa`: la consola muestra tiempo transcurrido y RAM periódicamente, además del tiempo por época, pérdida, F1 y VRAM reservada. Cuando se lanza desde `schedule`, la salida queda en `optimization_results/pipeline_logs/rna.log`.

La selección `--device auto` usa CUDA cuando la build de PyTorch y el driver la exponen; de lo contrario usa CPU. Para instalar soporte CUDA, usa el selector oficial de PyTorch y elige la build compatible con el sistema y driver: [Get Started Locally](https://pytorch.org/get-started/locally/). Esta instalación se administra aparte de Poetry; verifica con `poetry run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`. `--device cuda` falla con un mensaje explícito si CUDA no está disponible. La precisión mixta puede desactivarse con `--no-amp`.

## Estado local

El estado del scheduler, logs y checkpoints son archivos locales ignorados por Git. Los checkpoints sirven para reanudar en la misma copia de trabajo; no se recomiendan para compartir ni incluir en respaldos. Para empaquetar y publicar únicamente los artefactos finales de RNA, usa `poetry run drive bundle-rna`; el ZIP contiene los modelos y reportes de las cuatro representaciones y excluye checkpoints y estudios Optuna. Los modelos TF-IDF aún no incluyen los vectorizadores ajustados necesarios para inferencia de texto nuevo; ese pendiente está registrado en `estado.md`.
