"""
    PRUEBA del branch
    tarea1_template.py

    Plantilla base para la tarea de analisis de archivos CSV de gait.

    La idea de esta plantilla es que el estudiante complete las funciones
    necesarias para:
    1. cargar una "base de datos" simple con los nombres de los ficheros,
    2. leer los metadatos de cada CSV,
    3. leer las senales de interes de cada CSV,
    4. obtener la frecuencia de muestreo desde los metadatos,
    5. graficar Angle_X y Linear_Acceleration_Z usando un eje temporal.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Estas son las columnas de senales que nos interesa conservar del CSV.
COLUMNAS_INTERES = [
    "Angle_X",
    "Linear_Acceleration_Z",
    "Segmentation_output",
    "Sync",
]


# ---------------------------------------------------------------------------
@dataclass
class RegistroCSV:
    # Estructura principal para representar un archivo CSV del dataset.
    # - nombre_fichero: nombre del CSV.
    # - metadatos: tabla con columnas "campo" y "valor".
    # - datos: tabla con las senales seleccionadas.

    nombre_fichero: str = ""
    metadatos: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=["campo", "valor"])
    )
    datos: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=COLUMNAS_INTERES)
    )

    # @property para usar un metodo como atributo
    @property
    def total_metadatos(self) -> int:
        # Cantidad de filas cargadas en la tabla de metadatos.
        return len(self.metadatos)
# ---------------------------------------------------------------------------
def listar_archivos_csv(ruta_carpeta: str) -> list[str]:
    # Recorre una carpeta y devuelve solo los nombres de archivos .csv.
    # La salida queda ordenada alfabeticamente para mantener un orden estable.
    archivos_csv = []

    for nombre in os.listdir(ruta_carpeta):
        nombre_en_minusculas = nombre.lower()

        if nombre_en_minusculas.endswith(".csv"):
            archivos_csv.append(nombre)

    archivos_csv.sort()
    return archivos_csv


# ---------------------------------------------------------------------------
def cargar_csv(path: str) -> list[str]:
    # Valida la carpeta de entrada y devuelve la lista de archivos CSV.
    path_data_base = os.path.abspath(os.path.expanduser(path))

    if not os.path.exists(path_data_base):
        raise FileNotFoundError(f"No existe la ruta: {path_data_base}")

    archivos_csv = listar_archivos_csv(path_data_base)

    if not archivos_csv:
        raise FileNotFoundError(
            f"No se encontraron archivos CSV en: {path_data_base}"
        )

    return archivos_csv


# ---------------------------------------------------------------------------
def imprimir_resumen(ruta_csv: str, num_ficheros: int) -> None:
    # Imprime un resumen basico de la carpeta cargada.
    print(f"Carpeta de ficheros: {ruta_csv}")
    print(f"# de ficheros .csv: {num_ficheros}")


# ---------------------------------------------------------------------------
def cargar_metadatos(ruta_carpeta: str, data_base: list[RegistroCSV]) -> None:
    # Carga el bloque inicial de metadatos en cada registro.
    # El CSV se lee linea por linea hasta encontrar la primera linea vacia.
    # Cada linea de metadato se separa en:
    # - campo
    # - valor
    #
    # Tarea del estudiante:
    # 1. recorrer data_base,
    # 2. armar archivo_csv con os.path.join,
    # 3. abrir cada fichero,
    # 4. detenerse en la primera linea vacia,
    # 5. separar cada linea en campo y valor,
    # 6. guardar el resultado en registro.metadatos.
    #
    # Esta version deja una tabla vacia para que el script siga funcionando.
    for registro in data_base:
        archivo_csv = os.path.join(
            ruta_carpeta,
            registro.nombre_fichero
        )
        filas = []
        with open(archivo_csv, "r", encoding="utf-8-sig") as archivo:
            for linea in archivo:
                if linea.strip() == "":
                    break
                campo, valor = linea.split(",", 1)
                filas.append({
                    "campo": campo,
                    "valor": valor.strip()
                })
        registro.metadatos = pd.DataFrame(filas)


# ---------------------------------------------------------------------------
def cargar_senales(ruta_carpeta: str, data_base: list[RegistroCSV]) -> None:
    # Carga solo las senales de interes del bloque numerico de cada CSV.
    #
    # Tarea del estudiante:
    # 1. recorrer data_base,
    # 2. abrir el fichero y contar cuantas lineas hay antes del bloque
    #    numerico,
    # 3. leer la tabla numerica con pandas.read_csv(..., skiprows=...),
    # 4. limpiar los nombres de columnas si hace falta,
    # 5. quedarse solo con COLUMNAS_INTERES,
    # 6. convertir las columnas a numericas,
    # 7. guardar el resultado en registro.datos.
    #
    # Esta version deja una tabla vacia con las columnas esperadas para que
    # el script siga funcionando.
    for registro in data_base:
        archivo_csv = os.path.join(
            ruta_carpeta,
            registro.nombre_fichero
        )
        lineas = 0
        with open(archivo_csv, "r", encoding="utf-8-sig") as archivo:
            for linea in archivo:
                lineas += 1
                if linea.strip() == "":
                    break
        datos = pd.read_csv(
            archivo_csv,
            skiprows=lineas
        )
        datos = datos[COLUMNAS_INTERES]
        registro.datos = datos


# ---------------------------------------------------------------------------
def sombrear_intervalos_sync(ax, tiempo, sync) -> None:
    # Sombrea en gris claro los intervalos donde Sync toma valor 1.
    # La idea es:
    # - cuando Sync pasa de 0 a 1, se abre un intervalo sombreado;
    # - cuando Sync vuelve a 0, se cierra ese intervalo.
    #
    # Tarea del estudiante:
    # 1. recorrer la senal sync,
    # 2. detectar los cambios de 0 a 1 y de 1 a 0,
    # 3. usar ax.axvspan(inicio, fin, ...) para sombrear.
    #
    # Esta version no hace nada para no interrumpir la ejecucion.
    return


# ---------------------------------------------------------------------------
def graficar_registro(
    nombre_fichero: str,
    frecuencia_muestreo: float,
    angle_x,
    acc_z,
    sync,
) -> None:
    # Grafica Angle_X y Acc_Z usando tiempo en el eje X.
    # Parametros:
    # - nombre_fichero: se usa como titulo general de la figura.
    # - frecuencia_muestreo: valor en Hz para construir el vector tiempo.
    # - angle_x: senal de angulo en X.
    # - acc_z: senal de aceleracion lineal en Z.
    # - sync: senal binaria usada para sombrear el fondo.
    #
    # Tarea del estudiante:
    # 1. convertir las entradas a arreglos o series numericas,
    # 2. construir el vector tiempo como muestra / frecuencia,
    # 3. crear la figura con dos subplots,
    # 4. llamar a sombrear_intervalos_sync en ambos ejes,
    # 5. graficar Angle_X y Linear_Acceleration_Z,
    # 6. poner como titulo general el nombre del fichero.
    #
    # Esta version solo genera una figura vacia muy simple para que se vea
    # la estructura del resultado sin exigir la implementacion completa.
    figura, ejes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ejes[0].set_title("Angle X")
    ejes[0].set_ylabel("Angulo [deg]")
    ejes[1].set_title("Acceleration Z")
    ejes[1].set_ylabel("Aceleracion [m/s2]")
    ejes[1].set_xlabel("Tiempo [s]")
    figura.suptitle(nombre_fichero)
    figura.tight_layout(rect=(0, 0, 1, 0.97))
    plt.show()


# ---------------------------------------------------------------------------
def obtener_frecuencia_muestreo(registro: RegistroCSV) -> float | None:
    # Extrae la frecuencia de muestreo desde la tabla de metadatos.
    #
    # Tarea del estudiante:
    # 1. buscar en registro.metadatos la fila donde campo sea
    #    "Sampling Frequency",
    # 2. tomar el valor asociado,
    # 3. convertirlo a float,
    # 4. devolver ese numero.
    #
    # Mientras no este implementada, devuelve None.
    for _, fila in registro.metadatos.iterrows():

        if fila["campo"] == "Sampling Frequency":
            return float(fila["valor"])

    return None


# ---------------------------------------------------------------------------
def main() -> None:
    # Flujo principal del programa.
    data_base: list[RegistroCSV] = []

    # Calculamos la carpeta del dataset usando la ubicacion del script.
    directorio_script = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.abspath(
        os.path.join(directorio_script, "..", "..", "data", "raw", "gait")
    )

    # Paso 1: listamos archivos CSV.
    ficheros = cargar_csv(db_path)

    # Paso 2: creamos un registro por cada archivo encontrado.
    for fichero in ficheros:
        data_base.append(RegistroCSV(nombre_fichero=fichero))

    # Paso 3: completamos metadatos y senales en cada registro.
    cargar_metadatos(db_path, data_base)
    cargar_senales(db_path, data_base)

    imprimir_resumen(db_path, len(ficheros))

    # Mostramos un resumen corto de cada archivo cargado.
    for registro in data_base[:5]:
        print(registro.nombre_fichero)
        print(registro.metadatos.head())
        print(registro.datos.head())

    # Elegimos un indice de ejemplo para graficar.
    #
    # Una vez implementadas las funciones anteriores, descomentar lo
    # siguiente para generar las curvas de un registro:
    #
    # indice = 5
    # registro = data_base[indice]
    # frecuencia_muestreo = obtener_frecuencia_muestreo(registro)
    # graficar_registro(
    #     registro.nombre_fichero,
    #     frecuencia_muestreo,
    #     registro.datos["Angle_X"],
    #     registro.datos["Linear_Acceleration_Z"],
    #     registro.datos["Sync"],
    # )


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()