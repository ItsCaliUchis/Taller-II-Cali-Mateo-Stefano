# Mellow · Sentiment Lab

Frontend Gradio separado por funcionalidad. No importa ni ejecuta código del pipeline de ML.

## Estructura

```text
app.py
content/
  es.json                 # Textos visibles, labels y fragmentos HTML editables
mellow/
  content.py              # Carga y resolución de textos del JSON
  core/
    demo.py               # Inferencia léxica de demostración y resultado
    batch.py              # Procesamiento CSV de demostración
    feedback.py           # Registro de feedback (una respuesta por inferencia)
    history.py            # Renderizado de historial y feedback
  ui/
    layout.py             # Composición general
    styles.py             # CSS, paleta y microinteracciones
    pages/
      home.py
      inference.py
      batch.py
      workers.py
      feedback.py
      history.py
```


Las predicciones siguen siendo una simulación léxica en inglés. No representan los modelos entrenados. El feedback se conserva en el estado de sesión de Gradio; todavía no hay persistencia externa ni conexión con workers.
