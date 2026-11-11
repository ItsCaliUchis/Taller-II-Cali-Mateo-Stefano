from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import uuid

import config
import requests

from .client import DriveAccessError, DriveFile, PublicDriveClient


REMOTE_MANIFEST_NAME = "ListadoRemoto.json"
LOCAL_MANIFEST_NAME = "ListadoDescarga.json"
LOCAL_REMOTE_MANIFEST = config.DRIVE_REMOTE_MANIFEST
LOCAL_DOWNLOAD_MANIFEST = config.DRIVE_DOWNLOAD_MANIFEST
MANIFEST_SCHEMA_VERSION = 1
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_INVALID_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


class ManifestError(ValueError):
    """El manifiesto remoto no existe, es invalido o no coincide con Drive."""


@dataclass(frozen=True)
class ComparedFile:
    drive_file: DriveFile
    metadata: dict
    status: str
    local_sha256: str | None


@dataclass
class ScanResult:
    files: list[ComparedFile]
    local_only: list[str]
    untracked_remote: list[DriveFile]
    manifest_file: DriveFile


def _now():
    buenos_aires = timezone(timedelta(hours=-3), name="UTC-03:00")
    return datetime.now(buenos_aires).isoformat(timespec="seconds")


def _write_json_atomic(path: Path, payload: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_name(name: str) -> str:
    if (
        not isinstance(name, str)
        or not name
        or name in {".", ".."}
        or _INVALID_FILENAME.search(name)
        or name.rstrip(". ") != name
    ):
        raise ManifestError(f"Nombre de archivo no seguro para data/: {name!r}")
    return name


def _iso_timestamp(value, field_name: str, file_name: str) -> str:
    if not isinstance(value, str):
        raise ManifestError(f"{field_name} faltante para {file_name}.")
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ManifestError(f"{field_name} invalido para {file_name}: {value!r}") from error
    return value


def _load_remote_manifest(path: Path, listed_by_id: dict[str, DriveFile], manifest_id: str):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ManifestError(f"No se pudo leer {REMOTE_MANIFEST_NAME}: {error}") from error

    if not isinstance(payload, dict) or payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ManifestError(
            f"{REMOTE_MANIFEST_NAME} debe tener schema_version={MANIFEST_SCHEMA_VERSION}."
        )
    rows = payload.get("files")
    if not isinstance(rows, list):
        raise ManifestError(f"{REMOTE_MANIFEST_NAME} debe contener una lista 'files'.")

    entries = []
    seen_ids = set()
    seen_names = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ManifestError("Cada elemento de 'files' debe ser un objeto JSON.")
        drive_id = row.get("drive_id")
        name = _safe_name(row.get("name"))
        extension = row.get("extension")
        if not isinstance(drive_id, str) or not drive_id or drive_id == manifest_id:
            raise ManifestError(f"drive_id invalido para {name}.")
        if drive_id in seen_ids:
            raise ManifestError(f"drive_id repetido en el manifiesto: {drive_id}.")
        if name.casefold() in seen_names:
            raise ManifestError(f"Nombre repetido en la carpeta plana: {name}.")
        seen_ids.add(drive_id)
        seen_names.add(name.casefold())

        drive_file = listed_by_id.get(drive_id)
        if drive_file is None:
            raise ManifestError(f"{name} figura en el manifiesto, pero no en la carpeta Drive.")
        if drive_file.name != name:
            raise ManifestError(
                f"El nombre de {drive_id} cambio en Drive: manifiesto={name!r}, "
                f"carpeta={drive_file.name!r}."
            )
        expected_extension = Path(name).suffix
        if extension != expected_extension:
            raise ManifestError(
                f"Extension incorrecta para {name}: se esperaba {expected_extension!r}."
            )

        size_bytes = row.get("size_bytes")
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0:
            raise ManifestError(f"size_bytes invalido para {name}.")
        sha256 = row.get("sha256")
        if not isinstance(sha256, str) or not _SHA256.fullmatch(sha256):
            raise ManifestError(f"sha256 faltante o invalido para {name}.")

        entry = dict(row)
        entry.update({
            "drive_id": drive_id,
            "name": name,
            "extension": extension,
            "size_bytes": size_bytes,
            "created_at": _iso_timestamp(row.get("created_at"), "created_at", name),
            "modified_at": _iso_timestamp(row.get("modified_at"), "modified_at", name),
            "sha256": sha256.lower(),
        })
        entries.append((drive_file, entry))
    return payload, entries


def generate_remote_manifest(
    local_manifest_path=LOCAL_DOWNLOAD_MANIFEST,
    output_path=LOCAL_REMOTE_MANIFEST,
):
    """Genera el manifiesto remoto a partir de archivos locales asociados a Drive."""
    local_manifest_path = Path(local_manifest_path)
    try:
        ledger = json.loads(local_manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ManifestError(f"No se pudo leer {LOCAL_MANIFEST_NAME}: {error}") from error
    if not isinstance(ledger, dict) or not isinstance(ledger.get("files"), list):
        raise ManifestError(f"{LOCAL_MANIFEST_NAME} no contiene una lista 'files' valida.")

    rows = []
    updated_local_rows = []
    seen_ids = set()
    seen_names = set()
    for old in ledger["files"]:
        if not isinstance(old, dict) or not isinstance(old.get("name"), str):
            continue
        name = _safe_name(old["name"])
        path = config.DATA / name
        if not path.is_file():
            updated_local_rows.append(old)
            continue

        stat = path.stat()
        digest = _sha256(path)
        local_row = {
            **old,
            "name": name,
            "extension": path.suffix,
            "size_bytes": stat.st_size,
            "local_sha256": digest,
        }
        updated_local_rows.append(local_row)

        drive_id = old.get("drive_id")
        if not drive_id:
            continue
        local_row["sha256"] = digest
        local_row["created_at"] = datetime.fromtimestamp(stat.st_ctime, timezone.utc) \
            .isoformat(timespec="seconds").replace("+00:00", "Z")
        local_row["modified_at"] = datetime.fromtimestamp(stat.st_mtime, timezone.utc) \
            .isoformat(timespec="seconds").replace("+00:00", "Z")
        updated_local_rows[-1] = local_row
        if drive_id in seen_ids or name.casefold() in seen_names:
            raise ManifestError(f"ID o nombre repetido en {LOCAL_MANIFEST_NAME}: {name}.")
        seen_ids.add(drive_id)
        seen_names.add(name.casefold())
        rows.append({
            "drive_id": drive_id,
            "name": name,
            "extension": path.suffix,
            "size_bytes": stat.st_size,
            "created_at": datetime.fromtimestamp(stat.st_ctime, timezone.utc)
            .isoformat(timespec="seconds").replace("+00:00", "Z"),
            "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc)
            .isoformat(timespec="seconds").replace("+00:00", "Z"),
            "sha256": digest,
        })

    if not rows:
        raise ManifestError(
            f"No hay archivos locales existentes con drive_id en {LOCAL_MANIFEST_NAME}."
        )

    ledger["files"] = updated_local_rows
    ledger["listed_at"] = _now()
    _write_json_atomic(local_manifest_path, ledger)
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "files": rows,
    }
    _write_json_atomic(Path(output_path), payload)
    return payload


class DriveDownloader:
    """Crea un inventario local, lee el manifiesto remoto y sincroniza por SHA-256."""

    def __init__(self, folder_url: str):
        self.client = PublicDriveClient(folder_url)
        self.folder_id = self.client.folder_id
        self._scan_result: ScanResult | None = None
        self._ledger: dict = {}

    def close(self):
        self.client.close()

    def scan(self, *, progress=None) -> ScanResult:
        # Guarda el estado local antes de solicitar informacion a Drive.
        # Asi se conserva el inventario aunque falle la conexion remota.
        previous = self._read_local_manifest()
        self._write_local_inventory(previous)

        drive_files = self.client.list_files()
        by_id = {item.drive_id: item for item in drive_files}
        by_name = {}
        for item in drive_files:
            safe_name = _safe_name(item.name)
            key = safe_name.casefold()
            if key in by_name and by_name[key].drive_id != item.drive_id:
                raise ManifestError(f"Drive contiene nombres ambiguos en carpeta plana: {item.name}.")
            by_name[key] = item

        manifest_matches = [item for item in drive_files if item.name == REMOTE_MANIFEST_NAME]
        if len(manifest_matches) != 1:
            self._write_inventory(drive_files, None)
            raise ManifestError(
                f"Se esperaba exactamente un {REMOTE_MANIFEST_NAME} en la carpeta; "
                f"se encontraron {len(manifest_matches)}. Se guardo el listado inicial local."
            )
        manifest_file = manifest_matches[0]
        # Persiste el inventario antes de descargar el manifiesto de referencia.
        self._write_inventory(drive_files, previous, manifest_file=manifest_file)
        with tempfile.NamedTemporaryFile(
            prefix=".ListadoRemoto-", suffix=".part", dir=config.DATA, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
        try:
            self.client.download_file(manifest_file, temporary_path, progress=progress)
            payload, manifest_entries = _load_remote_manifest(
                temporary_path, by_id, manifest_file.drive_id
            )
            os.replace(temporary_path, LOCAL_REMOTE_MANIFEST)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise

        listed_data_ids = {
            item.drive_id for item in drive_files if item.drive_id != manifest_file.drive_id
        }
        manifest_data_ids = {drive_file.drive_id for drive_file, _ in manifest_entries}
        untracked_remote = [
            item for item in drive_files
            if item.drive_id != manifest_file.drive_id and item.drive_id not in manifest_data_ids
        ]
        stale_ids = manifest_data_ids - listed_data_ids
        if stale_ids:
            raise ManifestError(
                "ListadoRemoto.json contiene archivos que ya no estan en la carpeta: "
                + ", ".join(sorted(stale_ids))
            )

        ledger = self._read_local_manifest()
        old_rows = {
            row.get("drive_id"): row
            for row in ledger.get("files", [])
            if isinstance(row, dict) and row.get("drive_id")
        }
        old_rows_by_name = {
            row["name"].casefold(): row
            for row in ledger.get("files", [])
            if isinstance(row, dict) and isinstance(row.get("name"), str)
        }
        compared = []
        current_names = set()
        for drive_file, entry in manifest_entries:
            current_names.add(entry["name"].casefold())
            local_path = config.DATA / entry["name"]
            old_local = old_rows_by_name.get(entry["name"].casefold(), {})
            if local_path.is_file():
                local_sha256 = old_local.get("local_sha256")
                if (
                    not isinstance(local_sha256, str)
                    or local_path.stat().st_size != old_local.get("size_bytes")
                ):
                    local_sha256 = _sha256(local_path)
            else:
                local_sha256 = None
            if local_sha256 is None:
                status = "faltante"
            elif local_sha256 != entry["sha256"] or local_path.stat().st_size != entry["size_bytes"]:
                status = "diferente"
            else:
                status = "igual"

            old = old_rows.get(drive_file.drive_id, {})
            entry.update({
                "downloaded_at": old.get("downloaded_at"),
                "local_sha256": local_sha256,
                "status": status,
                "checked_at": _now(),
            })
            compared.append(ComparedFile(drive_file, entry, status, local_sha256))

        local_only = []
        excluded = {
            LOCAL_DOWNLOAD_MANIFEST.name.casefold(),
            LOCAL_REMOTE_MANIFEST.name.casefold(),
            Path(config.negaciones_path).name.casefold(),
        }
        for path in config.DATA.iterdir():
            if path.is_file() and path.name.casefold() not in excluded and path.name.casefold() not in current_names:
                if not path.name.startswith(".ListadoRemoto-"):
                    local_only.append(path.name)

        self._ledger = ledger
        local_only_names = {name.casefold() for name in local_only}
        local_only_rows = []
        for row in ledger.get("files", []):
            if (
                not isinstance(row, dict)
                or not isinstance(row.get("name"), str)
                or row["name"].casefold() not in local_only_names
            ):
                continue
            path = config.DATA / row["name"]
            if path.is_file():
                row = {
                    **row,
                    "size_bytes": path.stat().st_size,
                    "status": "solo local",
                }
                local_only_rows.append(row)
        self._ledger.update({
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "folder_id": self.folder_id,
            "listed_at": _now(),
            "remote_manifest": {
                "drive_id": manifest_file.drive_id,
                "name": REMOTE_MANIFEST_NAME,
                "sha256": _sha256(LOCAL_REMOTE_MANIFEST),
                "downloaded_at": _now(),
            },
            "files": [item.metadata for item in compared] + local_only_rows,
        })
        self._write_local_manifest()
        self._scan_result = ScanResult(compared, sorted(local_only), untracked_remote, manifest_file)
        return self._scan_result

    def download(self, *, selected_ids=None, overwrite=False, progress=None):
        if self._scan_result is None:
            raise RuntimeError("Primero debes ejecutar el listado y la comparacion.")

        selected_ids = None if selected_ids is None else set(selected_ids)
        downloaded = []
        failures = []
        for item in self._scan_result.files:
            should_download = (
                overwrite
                or item.drive_file.drive_id in (selected_ids or set())
                or (selected_ids is None and item.status != "igual")
            )
            if not should_download:
                continue

            destination = config.DATA / item.metadata["name"]
            temporary = config.DATA / f".drive-{uuid.uuid4().hex}.part"
            try:
                size, digest = self.client.download_file(
                    item.drive_file,
                    temporary,
                    expected_size=item.metadata["size_bytes"],
                    expected_sha256=item.metadata["sha256"],
                    progress=progress,
                )
                os.replace(temporary, destination)
                item.metadata.update({
                    "downloaded_at": _now(),
                    "local_sha256": digest,
                    "status": "descargado",
                    "size_bytes": size,
                })
                downloaded.append(item.metadata["name"])
                self._write_local_manifest()
            except (OSError, requests.RequestException, DriveAccessError) as error:
                temporary.unlink(missing_ok=True)
                failures.append((item.metadata["name"], str(error)))
                item.metadata["status"] = "error"
                self._write_local_manifest()
        return downloaded, failures

    def _read_local_manifest(self):
        if not LOCAL_DOWNLOAD_MANIFEST.is_file():
            return {}
        try:
            payload = json.loads(LOCAL_DOWNLOAD_MANIFEST.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return {}

    def _write_inventory(self, drive_files, previous, manifest_file=None):
        previous = previous or self._read_local_manifest()
        old_rows = {
            row.get("drive_id"): row
            for row in previous.get("files", [])
            if isinstance(row, dict) and row.get("drive_id")
        }
        old_rows_by_name = {
            row.get("name", "").casefold(): row
            for row in previous.get("files", [])
            if isinstance(row, dict) and isinstance(row.get("name"), str)
        }
        manifest_file = manifest_file or next(
            (item for item in drive_files if item.name == REMOTE_MANIFEST_NAME), None
        )
        rows = []
        for item in drive_files:
            if manifest_file and item.drive_id == manifest_file.drive_id:
                continue
            old = old_rows.get(item.drive_id) or old_rows_by_name.get(item.name.casefold(), {})
            rows.append({
                "drive_id": item.drive_id,
                "name": item.name,
                "extension": Path(item.name).suffix,
                "size_bytes": old.get("size_bytes"),
                "created_at": old.get("created_at"),
                "modified_at": old.get("modified_at"),
                "sha256": old.get("sha256"),
                "downloaded_at": old.get("downloaded_at"),
                "local_sha256": old.get("local_sha256"),
                "status": "listado",
            })
        remote_names = {row["name"].casefold() for row in rows}
        for name, old in old_rows_by_name.items():
            if name in remote_names:
                continue
            rows.append({
                **old,
                "drive_id": None,
                "status": "solo local",
            })
        _write_json_atomic(LOCAL_DOWNLOAD_MANIFEST, {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "folder_id": self.folder_id,
            "listed_at": _now(),
            "remote_manifest": {
                "drive_id": manifest_file.drive_id if manifest_file else None,
                "name": REMOTE_MANIFEST_NAME,
                "downloaded_at": None,
            },
            "files": rows,
        })

    def _write_local_inventory(self, previous):
        """Guarda los archivos existentes en data/ antes de contactar con Drive."""
        previous_rows = {
            row.get("name", "").casefold(): row
            for row in previous.get("files", [])
            if isinstance(row, dict) and isinstance(row.get("name"), str)
        }
        excluded = {
            LOCAL_DOWNLOAD_MANIFEST.name.casefold(),
            LOCAL_REMOTE_MANIFEST.name.casefold(),
            Path(config.negaciones_path).name.casefold(),
        }
        rows = []
        for path in sorted(config.DATA.iterdir(), key=lambda item: item.name.casefold()):
            if not path.is_file() or path.name.casefold() in excluded:
                continue
            if path.name.startswith((".ListadoRemoto-", ".drive-")):
                continue
            old = previous_rows.get(path.name.casefold(), {})
            rows.append({
                "drive_id": old.get("drive_id"),
                "name": path.name,
                "extension": path.suffix,
                "size_bytes": path.stat().st_size,
                "created_at": old.get("created_at"),
                "modified_at": old.get("modified_at"),
                "sha256": old.get("sha256"),
                "downloaded_at": old.get("downloaded_at"),
                "local_sha256": _sha256(path),
                "status": "local",
            })
        _write_json_atomic(LOCAL_DOWNLOAD_MANIFEST, {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "folder_id": self.folder_id,
            "listed_at": _now(),
            "remote_manifest": {
                "drive_id": None,
                "name": REMOTE_MANIFEST_NAME,
                "downloaded_at": None,
            },
            "files": rows,
        })

    def _write_local_manifest(self):
        _write_json_atomic(LOCAL_DOWNLOAD_MANIFEST, self._ledger)
