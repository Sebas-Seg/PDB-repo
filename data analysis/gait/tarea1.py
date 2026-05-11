"""
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

import csv
import os
import re
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
def _encontrar_skiprows_tabla_numerica(path_csv: str) -> int:
    # Devuelve la cantidad de lineas a saltar para que pandas lea la cabecera
    # de la tabla numerica (despues del bloque de metadatos y lineas vacias).
    with open(path_csv, "r", newline="", encoding="utf-8") as archivo:
        lineas = archivo.readlines()

    indice = 0
    while indice < len(lineas) and lineas[indice].strip() != "":
        indice += 1

    # Saltamos la primera linea vacia y cualquier linea vacia adicional.
    while indice < len(lineas) and lineas[indice].strip() == "":
        indice += 1

    return indice


# ---------------------------------------------------------------------------
def _parsear_float_desde_texto(texto: str) -> float | None:
    # Extrae el primer numero (int o float) de un texto.
    if texto is None:
        return None

    coincidencia = re.search(r"[-+]?\d+(?:\.\d+)?", str(texto))
    if not coincidencia:
        return None

    try:
        return float(coincidencia.group(0))
    except ValueError:
        return None
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
    for registro in data_base:
        archivo_csv = os.path.join(ruta_carpeta, registro.nombre_fichero)
        filas_metadatos: list[dict[str, str]] = []

        with open(archivo_csv, "r", newline="", encoding="utf-8") as archivo:
            lector = csv.reader(archivo)
            for fila in lector:
                # La linea vacia separa metadatos del bloque numerico.
                if not fila:
                    break

                campo = (fila[0] or "").strip()
                # Algunos valores contienen comas: los reconstruimos.
                valor = ",".join(fila[1:]).strip() if len(fila) > 1 else ""

                if campo == "":
                    continue

                filas_metadatos.append({"campo": campo, "valor": valor})

        registro.metadatos = pd.DataFrame(filas_metadatos, columns=["campo", "valor"])


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
    for registro in data_base:
        archivo_csv = os.path.join(ruta_carpeta, registro.nombre_fichero)
        skiprows = _encontrar_skiprows_tabla_numerica(archivo_csv)

        tabla = pd.read_csv(archivo_csv, skiprows=skiprows)
        tabla.columns = [str(c).strip() for c in tabla.columns]

        faltantes = [c for c in COLUMNAS_INTERES if c not in tabla.columns]
        if faltantes:
            raise KeyError(
                f"Faltan columnas requeridas en {registro.nombre_fichero}: {faltantes}"
            )

        tabla_interes = tabla[COLUMNAS_INTERES].copy()
        for col in COLUMNAS_INTERES:
            tabla_interes[col] = pd.to_numeric(tabla_interes[col], errors="coerce")

        registro.datos = tabla_interes


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
    tiempo = np.asarray(tiempo, dtype=float)
    sync_arr = np.asarray(sync, dtype=float)
    sync_arr = np.nan_to_num(sync_arr, nan=0.0)
    sync_on = sync_arr >= 0.5

    if len(tiempo) == 0 or len(sync_on) == 0:
        return

    n = min(len(tiempo), len(sync_on))
    tiempo = tiempo[:n]
    sync_on = sync_on[:n]

    # Detectamos flancos usando padding en ambos extremos.
    cambios = np.diff(sync_on.astype(int), prepend=0, append=0)
    inicios = np.where(cambios == 1)[0]
    finales = np.where(cambios == -1)[0]

    for inicio_idx, fin_idx in zip(inicios, finales, strict=False):
        if inicio_idx >= len(tiempo):
            continue
        if fin_idx <= 0:
            continue

        inicio_t = float(tiempo[inicio_idx])
        fin_t = float(tiempo[min(fin_idx - 1, len(tiempo) - 1)])
        if fin_t <= inicio_t:
            continue

        ax.axvspan(inicio_t, fin_t, color="0.9", zorder=0)


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
    if frecuencia_muestreo is None or float(frecuencia_muestreo) <= 0:
        raise ValueError("frecuencia_muestreo debe ser un numero positivo (Hz)")

    angle_x = pd.to_numeric(pd.Series(angle_x), errors="coerce").to_numpy(dtype=float)
    acc_z = pd.to_numeric(pd.Series(acc_z), errors="coerce").to_numpy(dtype=float)
    sync = pd.to_numeric(pd.Series(sync), errors="coerce").to_numpy(dtype=float)

    n = min(len(angle_x), len(acc_z), len(sync))
    angle_x = angle_x[:n]
    acc_z = acc_z[:n]
    sync = sync[:n]

    tiempo = np.arange(n, dtype=float) / float(frecuencia_muestreo)

    figura, ejes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    sombrear_intervalos_sync(ejes[0], tiempo, sync)
    sombrear_intervalos_sync(ejes[1], tiempo, sync)

    ejes[0].plot(tiempo, angle_x, linewidth=1)
    ejes[1].plot(tiempo, acc_z, linewidth=1)

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
    if registro.metadatos is None or registro.metadatos.empty:
        return None

    tabla = registro.metadatos.copy()
    tabla["campo"] = tabla["campo"].astype(str)

    mascara = tabla["campo"].str.strip().str.casefold() == "sampling frequency".casefold()
    if not mascara.any():
        return None

    valor = tabla.loc[mascara, "valor"].iloc[0]
    return _parsear_float_desde_texto(str(valor))


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