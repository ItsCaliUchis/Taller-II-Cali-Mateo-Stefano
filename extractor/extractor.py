import requests
import pandas as pd
import config
import sys
import time

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


def obtener_resenias(tamanio):
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


obtener_resenias(tamanio=1000)