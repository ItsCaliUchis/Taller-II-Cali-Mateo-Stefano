import argparse

import requests
import pandas as pd
import config
import sys
import time
from pathlib import Path

URL = "https://amazon-reviews-api-g5ae.onrender.com/reviews"


def mostrar_progreso(actual, total, inicio):
    porcentaje = actual / total
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


def obtener_resenias(tamanio, sobreescribir=False):

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

    while True:
        parametros = {"limit": tamanio, "offset": offset}

        print(f"\nSolicitando reseñas {offset:,} → {offset + tamanio:,}...")

        response = requests.get(URL, params=parametros)

        if response.status_code == 200:
            respuesta = response.json()

            resenias = respuesta.get("data", [])
            total = respuesta.get("total_matching", 0)
            devueltas = respuesta.get("returned", 0)

            lista_resenias.extend(resenias)
            offset += devueltas

            mostrar_progreso(offset, total, inicio)

            if offset >= total:
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
        default=1000,
        help="Cantidad de reseñas solicitadas por petición (default: 1000)."
    )

    parser.add_argument(
        "--overwrite",
        "--no-confirm",
        action="store_true",
        help="Sobreescribe el dataset existente sin pedir confirmación."
    )

    args = parser.parse_args()

    obtener_resenias(
        tamanio=args.size,
        sobreescribir=args.overwrite
    )


if __name__ == "__main__":
    main()