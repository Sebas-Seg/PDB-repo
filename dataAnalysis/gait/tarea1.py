"""
Tarea 1 - Procesamiento de Datos Biomecánicos

Autor: Luis Morel
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import pandas as pd


# ------------------------------------------------------------
COLUMNAS_INTERES = [
    "Angle_X",
    "Linear_Acceleration_Z",
    "Segmentation_output",
    "Sync",
]


# ------------------------------------------------------------
@dataclass
class RegistroCSV:
    nombre_fichero: str = ""

    metadatos: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=["campo", "valor"])
    )

    datos: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=COLUMNAS_INTERES)
    )

    @property
    def total_metadatos(self) -> int:
        return len(self.metadatos)


# ------------------------------------------------------------
def listar_archivos_csv(ruta_carpeta: str) -> list[str]:
    archivos = []

    for f in os.listdir(ruta_carpeta):
        if f.lower().endswith(".csv"):
            archivos.append(f)

    archivos.sort()
    return archivos


# ------------------------------------------------------------
def cargar_csv(path: str) -> list[str]:
    path = os.path.abspath(os.path.expanduser(path))

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    return listar_archivos_csv(path)


# ------------------------------------------------------------
def cargar_metadatos(ruta_carpeta: str, data_base: list[RegistroCSV]) -> None:
    """
    Lee metadatos de cada CSV
    """
    for registro in data_base:

        ruta = os.path.join(ruta_carpeta, registro.nombre_fichero)

        with open(ruta, "r", encoding="utf-8-sig") as f:
            lineas = f.read().splitlines()

        filas = []

        for linea in lineas:
            if not linea.strip():
                break

            partes = linea.split(",", 1)

            if len(partes) == 2:
                filas.append({
                    "campo": partes[0].strip(),
                    "valor": partes[1].strip()
                })

        registro.metadatos = pd.DataFrame(filas)


# ------------------------------------------------------------
def cargar_senales(ruta_carpeta: str, data_base: list[RegistroCSV]) -> None:
    """
    Lee señales del CSV
    """
    for registro in data_base:

        ruta = os.path.join(ruta_carpeta, registro.nombre_fichero)

        with open(ruta, "r", encoding="utf-8-sig") as f:
            lineas = f.read().splitlines()

        # buscar separación metadatos / datos
        sep = 0
        for i, l in enumerate(lineas):
            if not l.strip():
                sep = i
                break

        df = pd.read_csv(ruta, skiprows=sep + 1)

        df.columns = [c.strip() for c in df.columns]

        df = df[COLUMNAS_INTERES].copy()

        for c in COLUMNAS_INTERES:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.dropna().reset_index(drop=True)

        registro.datos = df


# ------------------------------------------------------------
def sombrear_intervalos_sync(ax, tiempo, sync) -> None:
    """
    Sombrea intervalos donde Sync = 1
    """

    inicio = None

    for i in range(len(sync)):

        if sync.iloc[i] == 1 and inicio is None:
            inicio = tiempo[i]

        elif sync.iloc[i] == 0 and inicio is not None:
            ax.axvspan(inicio, tiempo[i], color="gray", alpha=0.3)
            inicio = None


# ------------------------------------------------------------
def graficar_registro(
    nombre_fichero: str,
    frecuencia_muestreo: float,
    angle_x,
    acc_z,
    sync,
) -> None:

    angle_x = pd.Series(angle_x)
    acc_z = pd.Series(acc_z)
    sync = pd.Series(sync)

    n = len(angle_x)
    tiempo = [i / frecuencia_muestreo for i in range(n)]

    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # ---------------- ANGLE X
    sombrear_intervalos_sync(ax[0], tiempo, sync)
    ax[0].plot(tiempo, angle_x)
    ax[0].set_ylabel("Angle X")

    # ---------------- ACC Z
    sombrear_intervalos_sync(ax[1], tiempo, sync)
    ax[1].plot(tiempo, acc_z)
    ax[1].set_ylabel("Acc Z")
    ax[1].set_xlabel("Tiempo (s)")

    fig.suptitle(nombre_fichero)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
def obtener_frecuencia_muestreo(registro: RegistroCSV) -> float:

    fila = registro.metadatos[
        registro.metadatos["campo"] == "Sampling Frequency"
    ]

    if fila.empty:
        return 0.0

    return float(fila.iloc[0]["valor"])


# ------------------------------------------------------------
def imprimir_resumen(ruta_csv: str, num_ficheros: int) -> None:
    print("Carpeta:", ruta_csv)
    print("CSV encontrados:", num_ficheros)


# ------------------------------------------------------------
def main() -> None:

    data_base: list[RegistroCSV] = []

    base_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "gait")
    )

    ficheros = cargar_csv(base_path)

    for f in ficheros:
        data_base.append(RegistroCSV(nombre_fichero=f))

    cargar_metadatos(base_path, data_base)
    cargar_senales(base_path, data_base)

    imprimir_resumen(base_path, len(ficheros))

    # mostrar ejemplo
    for r in data_base[:3]:
        print(r.nombre_fichero)

    
        indice = 0
        registro = data_base[indice]
        fs = obtener_frecuencia_muestreo(registro)
     
        graficar_registro(
          registro.nombre_fichero,
          fs,
          registro.datos["Angle_X"],
          registro.datos["Linear_Acceleration_Z"],
          registro.datos["Sync"],
      )


# ------------------------------------------------------------
if __name__ == "__main__":
    main()