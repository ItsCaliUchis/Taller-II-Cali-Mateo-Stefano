# Estado del scope RNA

Fecha: 2026-11-12  
Rama: `rna_training`  
Estado: implementación en progreso, cambios sin commit para probar en la laptop.

## Objetivo

Añadir entrenamiento RNA (red neuronal artificial) como comando CLI integrado con el catálogo de comandos, `schedule`, las representaciones densas BERT/ELMo y la infraestructura de métricas/checkpoints existente.

## Decisiones acordadas

- Primera versión: MLP por reseña sobre embeddings precomputados.
- BERT y ELMo se entrenan por defecto; `--embedding bert|elmo|both` permite seleccionar.
- Colab queda para una etapa posterior. La laptop con RTX 3060 debe usar CUDA cuando esté disponible.
- La instalación de PyTorch con CUDA se documenta aparte de Poetry.
- `--overwrite` regenera salidas y conserva/reanuda checkpoints compatibles.
- `--restart` descarta el progreso de la configuración actual y vuelve a entrenar desde cero.
- `--limit` utiliza el contrato común de `utils.py` y requiere que las matrices correspondan al mismo prefijo del dataset.
- El scope queda en la rama `rna_training`, sin commits. Al integrar, se hará rebase y se dividirá en commits más pequeños.

## Implementado

- `src/training/data.py`: carga etiquetas y matrices densas; mapea las etiquetas originales 0–4 a `negative`, `neutral`, `positive`; cachea el mapeo sin modificar `dataset_clean.csv`.
- `src/training/rna.py`: MLP con mini-batches, división estratificada train/valid/test, AdamW, early stopping por F1 macro, opción CUDA y AMP, reportes, matrices de confusión, historial/gráficos y pesos finales.
- Checkpoints atómicos por época con estados de modelo/optimizador, aleatoriedad, historial y mejor modelo, organizados por representación y firma de datos/configuración.
- `src/training/sgd.py` consume la función común de carga de etiquetas para mantener el mapeo alineado.
- `src/training/graphics.py`: permite guardar gráficos sin abrir ventana y crea curvas de entrenamiento RNA.
- `src/schedule.py`/`src/commands.info.json`: metadata de salidas y dependencias condicionales según `--embedding`, soporte de resultados parciales y propagación del límite requerido a BERT/ELMo.
- `src/list_tools.py`: soporta editar y validar esos campos de schedule al registrar/modificar herramientas.
- Entry point `rna` registrado en `pyproject.toml`; documentación de CLI, checkpoints, CUDA y arquitectura actualizada.
- `optimization_results/plan.json` local incorpora RNA después de SGD. El archivo está ignorado por Git.

## Uso

```bash
poetry run rna --help
poetry run rna --embedding both --device auto
poetry run rna --embedding bert --limit 1000 --device cuda
poetry run rna --overwrite
poetry run rna --overwrite --restart
```

Para schedule, el plan local ya contiene RNA. Se puede ejecutar con `poetry run schedule run` o `poetry run schedule start`; schedule registra la salida de RNA en `optimization_results/pipeline_logs/rna.log`.

## Duración y logs

RNA usa `ejecutar_etapa` para cargar etiquetas, cargar cada matriz y entrenar cada representación. El monitor de consola informa tiempo transcurrido y RAM aproximadamente cada segundo, y al terminar informa duración total y RAM final. Cada época informa duración, loss y F1 macro. Al ejecutarse desde schedule, esta salida queda en el log de etapa. Los timestamps de inicio y fin también quedan en `optimization_results/pipeline_state.json`.

## Verificaciones realizadas

- `python -m compileall -q src`
- JSON de catálogo y plan local parsean correctamente.
- `git diff --check`
- `poetry run rna --help`

No se ejecutó entrenamiento ni pruebas automatizadas. En el entorno desktop disponible, PyTorch es CPU-only y no hay CUDA detectada; la validación con RTX 3060 queda para la laptop.

## Siguiente paso al probar en laptop

1. Instalar/seleccionar una build CUDA de PyTorch compatible con el driver NVIDIA, siguiendo el selector oficial: https://pytorch.org/get-started/locally/.
2. Confirmar `torch.cuda.is_available()` desde el entorno Poetry.
3. Verificar/generar BERT y ELMo para las mismas filas del dataset.
4. Ejecutar primero con un límite pequeño y una representación, confirmar duración, artefactos y reanudación de checkpoint; luego correr ambas representaciones.
5. Registrar observaciones y, cuando esté aprobado, rebasear/slice-ar el scope en commits Conventional Commits.
