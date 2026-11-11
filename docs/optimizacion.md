# Optimización y ejecución programada

## Comando `schedule`

`schedule` ejecuta etapas como procesos independientes, escribe logs separados y guarda el estado en `optimization_results/pipeline_state.json`.

```bash
poetry run schedule start
poetry run schedule status
poetry run schedule gui
poetry run schedule stop
```

- `start` lanza el coordinador en segundo plano en Windows y retorna el control a la terminal.
- `run` ejecuta el mismo plan en primer plano.
- `status` muestra estado, PID y progreso por etapa.
- `stop` pide una parada controlada al terminar la operación activa. Una corrida nueva vuelve a validar los artefactos existentes.
- `gui` abre una ventana Tkinter con estado actual, tabla en el mismo orden del plan, botones para iniciar/detener y un panel que sigue el log de la etapa activa.
- Al terminar o fallar una ejecución en Windows, `schedule` muestra una notificación en el área de notificación. `--no-notify` desactiva el aviso.
- Cada `start` o `run` nuevo borra el estado JSON y los logs creados por `schedule` en `optimization_results/`, y empieza un historial limpio. Los datasets, matrices, checkpoints y otros resultados del pipeline se conservan.
- Si Windows mantiene un log anterior abierto, el nuevo proceso escribe en una carpeta de logs exclusiva para no fallar al iniciar; se vuelve a intentar limpiar los logs anteriores en una corrida posterior. Al detener una etapa, el coordinador termina también los procesos descendientes de Poetry para liberar esos archivos.
- La secuencia es extracción, exploración, limpieza, recursos NLTK, BoW, TF-IDF, FastText, BERT, ELMo y SGD. `--skip-contextual` omite BERT y ELMo.
- Cada subcomando se inicia con su entrypoint instalado de Poetry (por ejemplo, `poetry run clean`). Antes de ejecutarlo, `schedule` revisa sus salidas: si todas existen y son válidas, omite la etapa; si existe solo una parte o hay archivos inválidos, la bloquea para evitar sobrescribirlos. `--overwrite` autoriza regenerar las salidas existentes. También valida los códigos de retorno, los archivos producidos y los punteros LFS sin contenido.
- Las marcas de tiempo del estado usan el horario de Buenos Aires, UTC−3.
- Los mensajes de consola usan caracteres ASCII y muestran las rutas de log relativas a la raiz del repositorio.
- `--workers 1` serializa los vectorizadores por defecto para reducir presión de memoria y CPU. Se puede subir el número de workers para usar un `ThreadPoolExecutor`, asumiendo mayor uso de recursos.

Ejemplo de ejecución completa sin los modelos contextuales:

```bash
poetry run schedule start --skip-contextual
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

## Estado local

El estado del scheduler, logs y checkpoints son archivos locales ignorados por Git. Los datasets, matrices y modelos descargados compatibles con LFS se pueden compartir en el remoto. Los checkpoints sirven para reanudar en la misma copia de trabajo; no se suben por defecto.
