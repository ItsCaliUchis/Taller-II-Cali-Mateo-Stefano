from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from urllib.parse import parse_qs, urlencode, urljoin, urlsplit, urlunsplit

from bs4 import BeautifulSoup
import requests


class DriveAccessError(RuntimeError):
    """Error cuando falla una descarga o la carpeta contiene subcarpetas."""


@dataclass(frozen=True)
class DriveFile:
    drive_id: str
    name: str
    resource_key: str | None = None


class PublicDriveClient:
    _FILE_ID = re.compile(r"/file/d/([\w-]+)(?:/|$)")
    _FOLDER_ID = re.compile(r"/drive/folders/([\w-]+)(?:/|$)")

    def __init__(self, folder_url: str, *, timeout=(15, 120)):
        self.folder_url = folder_url
        self.timeout = timeout
        match = self._FOLDER_ID.search(urlsplit(folder_url).path)
        if not match:
            raise ValueError("La URL de Drive no contiene un ID de carpeta valido.")
        self.folder_id = match.group(1)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        })

    def close(self):
        self.session.close()

    def list_files(self) -> list[DriveFile]:
        response = self.session.get(
            "https://drive.google.com/embeddedfolderview",
            params={"id": self.folder_id},
            timeout=self.timeout,
        )
        if response.status_code != 200:
            raise DriveAccessError(
                f"Drive devolvio HTTP {response.status_code} al listar la carpeta."
            )

        soup = BeautifulSoup(response.text, "html.parser")
        if soup.title is None or soup.title.string is None:
            raise DriveAccessError(
                "No se pudo leer el listado publico de Drive; verifica que el enlace "
                "permita el acceso a cualquier persona con el vinculo."
            )

        files = []
        seen_ids = set()
        nested_folders = []
        for anchor in soup.find_all("a", href=True):
            href = urljoin(response.url, anchor["href"])
            parsed = urlsplit(href)
            drive_id = None
            is_folder = False

            match = self._FILE_ID.search(parsed.path)
            if match:
                drive_id = match.group(1)
            else:
                match = self._FOLDER_ID.search(parsed.path)
                if match:
                    drive_id = match.group(1)
                    is_folder = True
                elif parsed.hostname in {"drive.google.com", "docs.google.com"}:
                    query_id = parse_qs(parsed.query).get("id", [None])[0]
                    if query_id and parsed.path in {"/open", "/uc"}:
                        drive_id = query_id

            if not drive_id or drive_id == self.folder_id:
                continue
            if is_folder:
                nested_folders.append(anchor.get_text(" ", strip=True) or drive_id)
                continue
            if drive_id in seen_ids:
                continue

            name = (
                anchor.get_text(" ", strip=True)
                or anchor.get("aria-label")
                or anchor.get("title")
            )
            if not name:
                continue
            resource_key = parse_qs(parsed.query).get("resourcekey", [None])[0]
            files.append(DriveFile(drive_id, name, resource_key))
            seen_ids.add(drive_id)

        if nested_folders:
            names = ", ".join(nested_folders)
            raise DriveAccessError(
                f"La carpeta debe ser plana; se encontraron subcarpetas: {names}."
            )
        return files

    def download_file(
        self,
        drive_file: DriveFile,
        destination: Path,
        *,
        expected_size: int | None = None,
        expected_sha256: str | None = None,
        progress=None,
    ) -> tuple[int, str]:
        params = {"export": "download", "id": drive_file.drive_id}
        if drive_file.resource_key:
            params["resourcekey"] = drive_file.resource_key

        response = self.session.get(
            "https://drive.google.com/uc",
            params=params,
            stream=True,
            timeout=self.timeout,
        )
        response.raise_for_status()
        first_chunk = next(response.iter_content(chunk_size=1024 * 1024), b"")
        if self._is_confirmation_page(response, first_chunk):
            body = first_chunk + b"".join(response.iter_content(chunk_size=1024 * 1024))
            download_url, confirmation_params = self._confirmation_request(
                response, body, params
            )
            response.close()
            response = self.session.get(
                download_url,
                params=confirmation_params,
                stream=True,
                timeout=self.timeout,
            )
            response.raise_for_status()
            first_chunk = next(response.iter_content(chunk_size=1024 * 1024), b"")
            if self._is_confirmation_page(response, first_chunk):
                response.close()
                raise DriveAccessError(
                    f"Drive no entrego el contenido binario de {drive_file.name}."
                )

        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        total = 0
        try:
            with destination.open("wb") as output:
                if first_chunk:
                    output.write(first_chunk)
                    digest.update(first_chunk)
                    total += len(first_chunk)
                    if progress:
                        progress(drive_file.name, total, expected_size)
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    output.write(chunk)
                    digest.update(chunk)
                    total += len(chunk)
                    if progress:
                        progress(drive_file.name, total, expected_size)
        except BaseException:
            destination.unlink(missing_ok=True)
            raise
        finally:
            response.close()

        actual_sha256 = digest.hexdigest()
        if expected_size is not None and total != expected_size:
            destination.unlink(missing_ok=True)
            raise DriveAccessError(
                f"Tamanio incorrecto para {drive_file.name}: se esperaban "
                f"{expected_size} bytes y llegaron {total}."
            )
        if expected_sha256 and actual_sha256.lower() != expected_sha256.lower():
            destination.unlink(missing_ok=True)
            raise DriveAccessError(
                f"El SHA-256 de {drive_file.name} no coincide con ListadoRemoto.json."
            )
        return total, actual_sha256

    @staticmethod
    def _is_confirmation_page(response, first_chunk: bytes) -> bool:
        content_type = response.headers.get("Content-Type", "").lower()
        preview = first_chunk.lstrip().lower()[:64]
        return "text/html" in content_type or preview.startswith((b"<!doctype html", b"<html"))

    def _confirmation_request(self, response, body: bytes, original_params: dict):
        html = body.decode(response.encoding or "utf-8", errors="replace")
        soup = BeautifulSoup(html, "html.parser")
        form = soup.select_one("#download-form")
        if form is not None:
            action = urljoin(response.url, form.get("action", response.url))
            form_params = {
                item.get("name"): item.get("value", "")
                for item in form.find_all("input", attrs={"type": "hidden"})
                if item.get("name")
            }
            form_params.setdefault("id", original_params["id"])
            if "resourcekey" in original_params:
                form_params.setdefault("resourcekey", original_params["resourcekey"])
            return action, form_params

        warning_token = next(
            (cookie.value for cookie in self.session.cookies if cookie.name.startswith("download_warning")),
            None,
        )
        if warning_token:
            return "https://drive.google.com/uc", {**original_params, "confirm": warning_token}

        for anchor in soup.find_all("a", href=True):
            href = urljoin(response.url, anchor["href"].replace("&amp;", "&"))
            if "export=download" in href:
                parsed = urlsplit(href)
                return urlunsplit(parsed._replace(query="")), parse_qs(parsed.query, keep_blank_values=True)
        raise DriveAccessError(
            "Drive solicito una confirmacion que no se pudo interpretar; "
            "el formato publico de descarga pudo haber cambiado."
        )
