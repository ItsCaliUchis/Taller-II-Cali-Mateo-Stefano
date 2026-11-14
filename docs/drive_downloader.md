# Descargas desde Drive

El comando `drive` lee la carpeta pública configurada en `src/config.py`. No usa OAuth ni tokens. La carpeta de Drive debe ser plana y contener un `ListadoRemoto.json` válido. Para publicar resultados RNA, `bundle-rna` reúne los modelos finales y sus reportes en un ZIP plano que esta herramienta puede inventariar.

## Flujo

1. Enumera primero los archivos locales de `data/`, calcula sus SHA-256 y crea `data/ListadoDescarga.json`. Esto preserva el estado local aunque falle la conexión con Drive.
2. Lee los nombres e IDs visibles en la carpeta pública y actualiza `ListadoDescarga.json` con ese inventario.
3. Descarga `ListadoRemoto.json` y lo guarda localmente en `data/`.
4. Compara cada archivo de `ListadoRemoto.json` con `data/` usando SHA-256 local. No descarga los contenidos para comparar.
5. Descarga los seleccionados a un temporal en `data/`, verifica tamaño y SHA-256 y solo entonces reemplaza el archivo final. Actualiza `ListadoDescarga.json` tras cada descarga correcta.

Los archivos locales que no figuren en Drive se informan, pero nunca se borran. Los archivos en Drive que no figuren en el manifiesto se muestran como `sin manifiesto` y no se descargan.

## Formato de `ListadoRemoto.json`

El archivo remoto es la fuente de verdad y debe contener el ID de Drive para localizar cada objeto, además de su checksum para evitar descargar todos los contenidos durante la comparación:

```json
{
  "schema_version": 1,
  "files": [
    {
      "drive_id": "ID_DEL_ARCHIVO_EN_DRIVE",
      "name": "dataset_raw.csv",
      "extension": ".csv",
      "size_bytes": 41417514,
      "created_at": "2026-11-10T18:00:00Z",
      "modified_at": "2026-11-10T18:00:00Z",
      "sha256": "<64 caracteres hexadecimales>"
    }
  ]
}
```

`created_at` y `modified_at` describen el artefacto en la fuente y usan ISO-8601. `downloaded_at` no va en el manifiesto remoto: cada copia local tiene una fecha distinta y se guarda en `ListadoDescarga.json`. El manifiesto remoto no debe incluirse a sí mismo en `files`.

`ListadoDescarga.json` se genera localmente antes de solicitar el manifiesto remoto. Primero registra los archivos presentes en `data/` con tamaño y SHA local; luego agrega el inventario público y se completa con los metadatos del manifiesto, el estado de comparación y la fecha de descarga. Los archivos locales que no figuren en el manifiesto remoto permanecen registrados con estado `solo local`. Ambos JSON y la copia local de `ListadoRemoto.json` quedan en `data/`, ignorados por Git.

## Uso

```bash
poetry run drive list
poetry run drive --download
poetry run drive --overwrite
poetry run drive gui
poetry run drive manifest
poetry run drive bundle-rna
```

- `list` crea el inventario, descarga el manifiesto y muestra `igual`, `faltante`, `diferente`, `solo local` y `sin manifiesto`.
- `--download` transfiere solo archivos faltantes o cuyo tamaño/SHA-256 no coincide.
- `--overwrite` transfiere todos los archivos de `ListadoRemoto.json`, incluso los iguales.
- `gui` muestra los archivos con casillas; `Marcar diferencias` prepara los faltantes y distintos, y `Descargar seleccionados` los transfiere.
- `manifest` genera `data/ListadoRemoto.json` usando los archivos locales que tengan `drive_id` en `ListadoDescarga.json`. Calcula SHA-256, tamaño, extensión y fechas; también actualiza `local_sha256` para los archivos inventariados.
- `bundle-rna` crea `data/rna_modelos_finales.zip` con los 16 modelos RNA completos (cuatro representaciones × dos etapas × dos escalados), sus metadatos y los informes, matrices, historiales y curvas. Excluye checkpoints y bases SQLite de Optuna. El ZIP queda en `data/` y aparece en el inventario normal.

El comando `manifest` prepara el JSON para subirlo manualmente a Drive; no lo sube. Solo incluye archivos que existen localmente y tienen `drive_id` en `ListadoDescarga.json`. Calcula fechas a partir de los metadatos del archivo local y las escribe en UTC. Para publicar el ZIP RNA, créalo, súbelo manualmente a la carpeta pública, registra su ID de Drive en `ListadoDescarga.json` y genera/sube el manifiesto actualizado. El downloader no sincroniza carpetas anidadas ni realiza subidas; el ZIP permite conservar la estructura interna usando el formato plano existente.

Los nombres y mensajes que muestra la CLI se normalizan a ASCII para compatibilidad con consolas de Windows; los nombres originales se conservan en ambos manifiestos.

Al agregar o actualizar los entrypoints del proyecto, ejecuta `poetry install --only-root` para instalar solo el proyecto y refrescar sus comandos sin reinstalar las dependencias.

Si todavía no hay un `ListadoRemoto.json` válido en Drive, el comando conserva el inventario inicial en `ListadoDescarga.json` y termina con un mensaje de error. Ese manifiesto debe generarse y colocarse manualmente en la carpeta compartida hasta que exista un uploader.
