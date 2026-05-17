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
 423 changes: 423 additions & 0 deletions423  
dataAnalysis/gait/tarea2_template.py
Original file line number	Diff line number	Diff line change
@@ -0,0 +1,423 @@
"""Plantilla base para la tarea 2 de procesamiento de datos de marcha.
La idea de esta plantilla es que el estudiante complete las funciones
necesarias para:
1. listar los archivos CSV del dataset gait y seleccionar uno,
2. extraer metadatos y frecuencia de muestreo,
3. cargar las senales Angle_X, Linear_Acceleration_Z,
   Segmentation_output y Sync,
4. corregir el signo de la aceleracion en Z,
5. calcular metricas temporales y espaciales simples del registro.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
DISTANCIA_UTIL_10MWT_M = 6.0
CAMPO_FRECUENCIA = "Sampling Frequency"
COLUMNAS_INTERES = [
    "Angle_X",
    "Linear_Acceleration_Z",
    "Segmentation_output",
    "Sync",
]


# ---------------------------------------------------------------------------
@dataclass
class RegistroCSV:
    # Estructura principal para representar el archivo de trabajo.
    # - nombre_fichero: nombre del CSV analizado.
    # - metadatos: tabla con columnas "campo" y "valor".
    # - datos: tabla con las senales necesarias para la tarea.

    nombre_fichero: str = ""
    metadatos: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=["campo", "valor"])
    )
    datos: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=COLUMNAS_INTERES)
    )

    @property
    def total_muestras(self) -> int:
        # Cantidad de muestras validas cargadas en el bloque numerico.
        return len(self.datos)


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
def seleccionar_archivo_csv(
    ruta_carpeta: str,
    ficheros: list[str],
    indice: int,
) -> str:
    # Selecciona un solo CSV de la carpeta a partir de un indice.
    if indice < 0 or indice >= len(ficheros):
        raise IndexError(
            f"Indice fuera de rango: {indice}. "
            f"Debe estar entre 0 y {len(ficheros) - 1}."
        )

    return os.path.join(ruta_carpeta, ficheros[indice])


# ---------------------------------------------------------------------------
def leer_lineas_csv(ruta_csv: str) -> list[str]:
    # Lee el archivo completo y devuelve sus lineas sin saltos finales.
    with open(ruta_csv, "r", encoding="utf-8-sig") as archivo:
        return archivo.read().splitlines()


# ---------------------------------------------------------------------------
def encontrar_linea_separadora(lineas: list[str]) -> int:
    # Busca la primera linea vacia que separa metadatos y senales.
    for indice, linea in enumerate(lineas):
        if not linea.strip():
            return indice

    raise ValueError("El archivo no contiene una linea vacia de separacion.")


# ---------------------------------------------------------------------------
def cargar_metadatos(ruta_csv: str) -> pd.DataFrame:
    # Carga el bloque de metadatos como una tabla de dos columnas:
    # - campo
    # - valor
    lineas = leer_lineas_csv(ruta_csv)
    indice_separador = encontrar_linea_separadora(lineas)
    filas_metadatos = []

    for linea in lineas[:indice_separador]:
        partes = linea.split(",", 1)

        if len(partes) == 2:
            campo = partes[0].strip()
            valor = partes[1].strip()
            filas_metadatos.append({"campo": campo, "valor": valor})

    return pd.DataFrame(filas_metadatos, columns=["campo", "valor"])


# ---------------------------------------------------------------------------
def cargar_datos(ruta_csv: str) -> pd.DataFrame:
    # Carga solo las senales necesarias para la tarea.
    # Luego convierte las columnas a formato numerico y elimina filas
    # incompletas.
    lineas = leer_lineas_csv(ruta_csv)
    indice_separador = encontrar_linea_separadora(lineas)
    datos = pd.read_csv(ruta_csv, skiprows=indice_separador + 1)
    datos.columns = [columna.strip() for columna in datos.columns]

    columnas_faltantes = [
        columna for columna in COLUMNAS_INTERES if columna not in datos.columns
    ]
    if columnas_faltantes:
        raise ValueError(
            f"Faltan columnas requeridas en el CSV: {columnas_faltantes}"
        )

    datos_filtrados = datos[COLUMNAS_INTERES].copy()

    for columna in COLUMNAS_INTERES:
        datos_filtrados[columna] = pd.to_numeric(
            datos_filtrados[columna],
            errors="coerce",
        )

    datos_filtrados = datos_filtrados.dropna(
        subset=COLUMNAS_INTERES
    ).reset_index(drop=True)

    return datos_filtrados


# ---------------------------------------------------------------------------
def construir_registro_desde_csv(ruta_csv: str) -> RegistroCSV:
    # Arma la estructura principal del ejercicio a partir de un solo archivo.
    return RegistroCSV(
        nombre_fichero=os.path.basename(ruta_csv),
        metadatos=cargar_metadatos(ruta_csv),
        datos=cargar_datos(ruta_csv),
    )


# ---------------------------------------------------------------------------
def obtener_frecuencia_muestreo(registro: RegistroCSV) -> float:
    # Extrae la frecuencia de muestreo desde la tabla de metadatos.
    #
    # Tarea del estudiante:
    # 1. buscar la fila donde campo sea "Sampling Frequency",
    # 2. recuperar el valor asociado,
    # 3. convertirlo a float,
    # 4. devolver ese numero.
    #
    # Mientras no se implemente, devuelve 0.0.
    return 0.0


# ---------------------------------------------------------------------------
def corregir_aceleracion(registro: RegistroCSV) -> None:
    # Cambia el signo de Linear_Acceleration_Z en todas las muestras.
    #
    # Tarea del estudiante:
    # 1. acceder a la columna Linear_Acceleration_Z,
    # 2. multiplicarla por -1,
    # 3. guardar el resultado en la misma tabla.
    #
    return


# ---------------------------------------------------------------------------
def calcular_longitud_temporal(total_muestras: int, frecuencia_muestreo: float) -> float:
    # Convierte cantidad de muestras en tiempo total del registro.
    if total_muestras <= 0 or frecuencia_muestreo <= 0:
        return 0.0

    return total_muestras / frecuencia_muestreo


# ---------------------------------------------------------------------------
def buscar_indice_primera_sync(sync) -> int:
    # Devuelve el indice de la primera muestra donde Sync es distinto de 0.
    #
    # Tarea del estudiante:
    # 1. recorrer la senal Sync,
    # 2. detectar la primera muestra distinta de cero,
    # 3. devolver ese indice.
    #
    # Mientras no se implemente, devuelve -1.
    return -1


# ---------------------------------------------------------------------------
def buscar_indice_ultima_sync(sync) -> int:
    # Devuelve el indice de la ultima muestra donde Sync es distinto de 0.
    #
    # Tarea del estudiante:
    # 1. recorrer la senal desde el final,
    # 2. detectar la ultima muestra distinta de cero,
    # 3. devolver ese indice.
    #
    # Mientras no se implemente, devuelve -1.
    return -1


# ---------------------------------------------------------------------------
def contar_transiciones_s3_s0(segmentation_output, inicio: int, fin: int) -> int:
    # Cuenta cuantas veces la senal pasa de 3 a 0 dentro de la ventana Sync.
    #
    # Tarea del estudiante:
    # 1. recorrer Segmentation_output entre inicio y fin,
    # 2. comparar cada muestra con la siguiente,
    # 3. contar las transiciones donde aparece 3 seguido de 0.
    #
    # Mientras no se implemente, devuelve 0.
    return 0


# ---------------------------------------------------------------------------
def calcular_velocidad_marcha(
    muestras_sync: int,
    frecuencia_muestreo: float,
    distancia_m: float = DISTANCIA_UTIL_10MWT_M,
) -> float:
    # Calcula la velocidad media de marcha en la ventana util del 10MWT.
    #
    # Tarea del estudiante:
    # 1. convertir la cantidad de muestras Sync en tiempo,
    # 2. usar la distancia util de 6 metros,
    # 3. devolver distancia / tiempo.
    #
    # Mientras no se implemente, devuelve 0.0.
    return 0.0


# ---------------------------------------------------------------------------
def calcular_velocidad_pasos(
    pasos: int,
    muestras_pasos: int,
    frecuencia_muestreo: float,
) -> float:
    # Calcula la cantidad de pasos por segundo dentro de la ventana Sync.
    #
    # Tarea del estudiante:
    # 1. convertir muestras_pasos en tiempo,
    # 2. dividir la cantidad de pasos por ese tiempo,
    # 3. devolver el resultado en pasos/s.
    #
    # Mientras no se implemente, devuelve 0.0.
    return 0.0


# ---------------------------------------------------------------------------
def calcular_longitud_zancada(
    velocidad_marcha: float,
    velocidad_pasos: float,
) -> float:
    # Estima la distancia media recorrida por cada paso detectado.
    #
    # Tarea del estudiante:
    # 1. tomar la velocidad de marcha en m/s,
    # 2. dividirla por la velocidad de pasos en pasos/s,
    # 3. devolver el resultado en metros por paso.
    #
    # Mientras no se implemente, devuelve 0.0.
    return 0.0


# ---------------------------------------------------------------------------
def calcular_metricas(
    registro: RegistroCSV,
    frecuencia_muestreo: float,
) -> dict[str, float | int]:
    # Reune todas las metricas pedidas en la tarea.
    inicio_sync = buscar_indice_primera_sync(registro.datos["Sync"])
    fin_sync = buscar_indice_ultima_sync(registro.datos["Sync"])
    muestras_sync = 0

    if inicio_sync >= 0 and fin_sync > inicio_sync:
        muestras_sync = fin_sync - inicio_sync

    pasos = contar_transiciones_s3_s0(
        registro.datos["Segmentation_output"],
        inicio_sync,
        fin_sync,
    )
    tiempo_total = calcular_longitud_temporal(
        registro.total_muestras,
        frecuencia_muestreo,
    )
    velocidad_marcha = calcular_velocidad_marcha(
        muestras_sync,
        frecuencia_muestreo,
    )
    velocidad_pasos = calcular_velocidad_pasos(
        pasos,
        muestras_sync,
        frecuencia_muestreo,
    )
    longitud_zancada = calcular_longitud_zancada(
        velocidad_marcha,
        velocidad_pasos,
    )

    return {
        "muestras_sync": muestras_sync,
        "pasos": pasos,
        "tiempo_total": tiempo_total,
        "velocidad_marcha": velocidad_marcha,
        "velocidad_pasos": velocidad_pasos,
        "longitud_zancada": longitud_zancada,
    }


# ---------------------------------------------------------------------------
def imprimir_resultados(
    registro: RegistroCSV,
    frecuencia_muestreo: float,
    metricas: dict[str, float | int],
) -> None:
    # Imprime en pantalla el resumen numerico del analisis.
    print(f"Fichero analizado: {registro.nombre_fichero}")
    print(f"Muestras leidas: {registro.total_muestras}")
    print(f"Frecuencia de muestreo: {frecuencia_muestreo:.3f} Hz")
    print(f"Muestras entre Sync: {metricas['muestras_sync']}")
    print(f"Velocidad de marcha: {metricas['velocidad_marcha']:.3f} m/s")
    print(f"Pasos detectados (S3->S0): {metricas['pasos']}")
    print(
        "Velocidad media de pasos: "
        f"{metricas['velocidad_pasos']:.3f} pasos/s"
    )
    print(
        "Longitud media estimada por paso: "
        f"{metricas['longitud_zancada']:.3f} m"
    )
    print(
        "Correccion aplicada: cambio de signo en "
        "Linear_Acceleration_Z de cada muestra."
    )
    print(f"Tiempo total del registro: {metricas['tiempo_total']:.3f} s")


# ---------------------------------------------------------------------------
def main() -> None:
    # Flujo principal del programa.
    directorio_script = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.abspath(
        os.path.join(directorio_script, "..", "..", "data", "raw", "gait")
    )

    # Paso 1: listamos todos los archivos CSV disponibles.
    ficheros = cargar_csv(db_path)
    imprimir_resumen(db_path, len(ficheros))

    # Paso 2: elegimos un solo archivo para esta tarea.
    #
    # Tarea del estudiante:
    # 1. decidir que indice de la lista quiere analizar,
    # 2. usar seleccionar_archivo_csv(...) para obtener la ruta completa.
    indice = 0
    ruta_csv = seleccionar_archivo_csv(db_path, ficheros, indice)

    # Paso 3: cargamos el archivo seleccionado en un registro.
    registro = construir_registro_desde_csv(ruta_csv)
    frecuencia_muestreo = obtener_frecuencia_muestreo(registro)

    # Cuando las funciones principales esten completas, este flujo deberia
    # producir resultados reales a partir del CSV.
    corregir_aceleracion(registro)

    metricas = calcular_metricas(registro, frecuencia_muestreo)
    imprimir_resultados(registro, frecuencia_muestreo, metricas)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()