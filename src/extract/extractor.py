import argparse

import requests
import pandas as pd
import config
import sys
import time
from pathlib import Path

URL = "https://amazon-reviews-api-g5ae.onrender.com/reviews"
PAGE_SIZE = 1000


def mostrar_progreso(actual, total, inicio):
    porcentaje = min(actual / total, 1) if total else 1
    ancho = 40
    completado = int(ancho * porcentaje)

    barra = "█" * completado + "░" * (ancho - completado)

    velocidad = actual / (time.time() - inicio)
    restante = (total - actual) / velocidad if velocidad > 0 else 0

    sys.stdout.write(
        f"\r[{barra}] {porcentaje:6.2%} | "
        f"{actual:,}/{total:,} | "
        f"{velocidad:.1f} reseñas/s | "
        f"ETA: {restante:.0f}s"
    )
    sys.stdout.flush()


def obtener_resenias(tamanio_miles=None, sobreescribir=False):

    if tamanio_miles is not None and tamanio_miles <= 0:
        raise ValueError("--size debe ser un entero positivo")

    if Path(config.dataset_raw).exists():
        if not sobreescribir:
            respuesta = input(
                f'El dataset ya existe en "{config.dataset_raw}". '
                "¿Querés sobreescribirlo? [s/N]: "
            )

            if respuesta.lower() != "s":
                print("\nDescarga cancelada. El Programa ha terminado")
                return
            
    lista_resenias = []
    offset = 0
    inicio = time.time()
    total = None
    objetivo = None

    while objetivo is None or len(lista_resenias) < objetivo:
        parametros = {"limit": PAGE_SIZE, "offset": offset}

        print(f"\nSolicitando reseñas {offset:,} → {offset + PAGE_SIZE:,}...")

        response = requests.get(URL, params=parametros)

        if response.status_code == 200:
            respuesta = response.json()

            resenias = respuesta.get("data", [])
            total = respuesta.get("total_matching", 0)
            objetivo = total if tamanio_miles is None else min(
                tamanio_miles * PAGE_SIZE,
                total,
            )

            restantes = objetivo - len(lista_resenias)
            lista_resenias.extend(resenias[:restantes])
            offset += len(resenias)

            mostrar_progreso(len(lista_resenias), objetivo, inicio)

            if not resenias or len(lista_resenias) >= objetivo:
                break

        else:
            print(f"\nError HTTP: {response.status_code}")
            break

    print("\n\nDescarga finalizada.")
    print(f"Reseñas descargadas: {len(lista_resenias):,}")
    print(f"Tiempo total: {time.time() - inicio:.1f} segundos")

    dataset = pd.DataFrame(lista_resenias)

    dataset.to_csv(config.dataset_raw, index=False)

    print("\nDataset:")
    print(dataset.describe())


def main():
    parser = argparse.ArgumentParser(
        description="Descarga las reseñas de Amazon y genera el dataset raw."
    )

    parser.add_argument(
        "--size",
        type=int,
        default=None,
        help=(
            "Cantidad de miles de reseñas a descargar; por ejemplo, "
            "--size 5 descarga hasta 5000. Cada petición solicita 1000. "
            "Si se omite, descarga todas las disponibles."
        ),
    )

    parser.add_argument(
        "--overwrite",
        "--no-confirm",
        action="store_true",
        help="Sobreescribe el dataset existente sin pedir confirmación."
    )

    args = parser.parse_args()

    if args.size is not None and args.size <= 0:
        parser.error("--size debe ser un entero positivo")

    obtener_resenias(tamanio_miles=args.size, sobreescribir=args.overwrite)


if __name__ == "__main__":
    main()
