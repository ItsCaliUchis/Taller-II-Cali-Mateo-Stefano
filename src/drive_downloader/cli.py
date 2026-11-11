import argparse
import sys
import unicodedata

import config

from .client import DriveAccessError
from .gui import open_gui
from .sync import DriveDownloader, ManifestError, generate_remote_manifest


def _console_text(value):
    value = str(value).replace("ñ", "n").replace("Ñ", "N")
    return unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")


def _print_comparison(result):
    print("Estado\tArchivo\tTamano\tSHA-256")
    for item in result.files:
        print(
            f"{item.status}\t{_console_text(item.metadata['name'])}\t"
            f"{item.metadata['size_bytes']}\t{item.metadata['sha256']}"
        )
    for name in result.local_only:
        print(f"solo local\t{_console_text(name)}\t-\t-")
    for item in result.untracked_remote:
        print(f"sin manifiesto\t{_console_text(item.name)}\t-\t-")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Lista y descarga artefactos desde la carpeta publica de Drive."
    )
    parser.add_argument(
        "action", nargs="?", choices=("list", "gui", "manifest"), default="list",
        help="Lista/compara, abre la GUI o genera ListadoRemoto.json desde data/.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--download", action="store_true",
        help="Descarga unicamente archivos faltantes o distintos.",
    )
    mode.add_argument(
        "--overwrite", action="store_true",
        help="Descarga y reemplaza todos los archivos enumerados por ListadoRemoto.json.",
    )
    parser.add_argument("--folder-url", default=config.DRIVE_FOLDER_URL)
    args = parser.parse_args(argv)

    if args.action == "manifest":
        if args.download or args.overwrite:
            parser.error("manifest genera el JSON y no admite --download ni --overwrite.")
        try:
            payload = generate_remote_manifest()
        except (ManifestError, OSError, ValueError) as error:
            print(f"Error: {_console_text(error)}", file=sys.stderr)
            return 1
        print(f"Generado {config.DRIVE_REMOTE_MANIFEST.name} con {len(payload['files'])} archivo(s).")
        print("ListadoDescarga.json tambien fue actualizado con los SHA-256 locales.")
        return 0

    if args.action == "gui":
        if args.download or args.overwrite:
            parser.error("--download y --overwrite se usan desde la CLI, no con gui.")
        return open_gui(args.folder_url)

    downloader = DriveDownloader(args.folder_url)
    try:
        result = downloader.scan()
        _print_comparison(result)
        if args.download or args.overwrite:
            downloaded, failures = downloader.download(overwrite=args.overwrite)
            for name in downloaded:
                print(f"Descargado y verificado: {_console_text(name)}")
            for name, error in failures:
                print(f"Error al descargar {_console_text(name)}: {_console_text(error)}", file=sys.stderr)
            if failures:
                return 1
        return 0
    except (DriveAccessError, ManifestError, OSError, ValueError) as error:
        print(f"Error: {_console_text(error)}", file=sys.stderr)
        return 1
    finally:
        downloader.close()


if __name__ == "__main__":
    raise SystemExit(main())
