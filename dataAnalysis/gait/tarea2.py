"""
------------------------------------------------------------
Tarea 2 - Procesamiento de Datos Biomecánicos
Programa COIL España - Paraguay

Archivo: Tarea2.py

Autores:
- Luis Morel

Descripción:
Programa para analizar registros de marcha del protocolo
10MWT a partir de archivos CSV.

El programa:
1. Carga la carpeta de archivos CSV.
2. Permite seleccionar un archivo.
3. Lee metadatos y señales.
4. Corrige el signo de Linear_Acceleration_Z.
5. Detecta la ventana Sync.
6. Cuenta pasos mediante transiciones S3 -> S0.
7. Calcula métricas temporales y espaciales.

------------------------------------------------------------
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import pandas as pd


# ------------------------------------------------------------
# CONSTANTES
# ------------------------------------------------------------

DISTANCIA_UTIL_10MWT_M = 6.0

CAMPO_FRECUENCIA = "Sampling Frequency"

COLUMNAS_INTERES = [
    "Angle_X",
    "Linear_Acceleration_Z",
    "Segmentation_output",
    "Sync",
]


# ------------------------------------------------------------
# ESTRUCTURA PRINCIPAL
# ------------------------------------------------------------

@dataclass
class RegistroCSV:
    """
    Representa un archivo CSV del dataset de marcha.
    """

    nombre_fichero: str = ""

    metadatos: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(
            columns=["campo", "valor"]
        )
    )

    datos: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(
            columns=COLUMNAS_INTERES
        )
    )

    @property
    def total_muestras(self) -> int:
        """
        Cantidad total de muestras válidas.
        """
        return len(self.datos)


# ------------------------------------------------------------
# FUNCIONES AUXILIARES
# ------------------------------------------------------------

def listar_archivos_csv(ruta_carpeta: str) -> list[str]:
    """
    Devuelve una lista ordenada de archivos CSV.
    """

    archivos_csv = []

    for nombre in os.listdir(ruta_carpeta):
        if nombre.lower().endswith(".csv"):
            archivos_csv.append(nombre)

    archivos_csv.sort()

    return archivos_csv


# ------------------------------------------------------------

def cargar_csv(path: str) -> list[str]:
    """
    Verifica la carpeta y devuelve los archivos CSV.
    """

    path_data_base = os.path.abspath(
        os.path.expanduser(path)
    )

    if not os.path.exists(path_data_base):
        raise FileNotFoundError(
            f"No existe la ruta: {path_data_base}"
        )

    archivos_csv = listar_archivos_csv(path_data_base)

    if not archivos_csv:
        raise FileNotFoundError(
            f"No se encontraron archivos CSV en: {path_data_base}"
        )

    return archivos_csv


# ------------------------------------------------------------

def imprimir_resumen(
    ruta_csv: str,
    num_ficheros: int,
) -> None:
    """
    Imprime información básica de la carpeta.
    """

    print("\n----------------------------------------")
    print("RESUMEN DEL DATASET")
    print("----------------------------------------")
    print(f"Carpeta: {ruta_csv}")
    print(f"Cantidad de archivos CSV: {num_ficheros}")
    print("----------------------------------------\n")


# ------------------------------------------------------------

def seleccionar_archivo_csv(
    ruta_carpeta: str,
    ficheros: list[str],
    indice: int,
) -> str:
    """
    Devuelve la ruta completa del archivo seleccionado.
    """

    if indice < 0 or indice >= len(ficheros):
        raise IndexError(
            f"Índice fuera de rango: {indice}"
        )

    return os.path.join(
        ruta_carpeta,
        ficheros[indice],
    )


# ------------------------------------------------------------

def leer_lineas_csv(ruta_csv: str) -> list[str]:
    """
    Lee el archivo CSV completo.
    """

    with open(
        ruta_csv,
        "r",
        encoding="utf-8-sig"
    ) as archivo:

        return archivo.read().splitlines()


# ------------------------------------------------------------

def encontrar_linea_separadora(
    lineas: list[str]
) -> int:
    """
    Busca la línea vacía que separa
    metadatos y señales.
    """

    for indice, linea in enumerate(lineas):

        if not linea.strip():
            return indice

    raise ValueError(
        "No se encontró línea separadora."
    )


# ------------------------------------------------------------
# CARGA DE METADATOS
# ------------------------------------------------------------

def cargar_metadatos(
    ruta_csv: str
) -> pd.DataFrame:
    """
    Carga los metadatos del CSV.
    """

    lineas = leer_lineas_csv(ruta_csv)

    indice_separador = encontrar_linea_separadora(
        lineas
    )

    filas_metadatos = []

    for linea in lineas[:indice_separador]:

        partes = linea.split(",", 1)

        if len(partes) == 2:

            campo = partes[0].strip()
            valor = partes[1].strip()

            filas_metadatos.append({
                "campo": campo,
                "valor": valor,
            })

    return pd.DataFrame(
        filas_metadatos,
        columns=["campo", "valor"],
    )


# ------------------------------------------------------------
# CARGA DE SEÑALES
# ------------------------------------------------------------

def cargar_datos(
    ruta_csv: str
) -> pd.DataFrame:
    """
    Carga únicamente las señales necesarias.
    """

    lineas = leer_lineas_csv(ruta_csv)

    indice_separador = encontrar_linea_separadora(
        lineas
    )

    datos = pd.read_csv(
        ruta_csv,
        skiprows=indice_separador + 1
    )

    datos.columns = [
        columna.strip()
        for columna in datos.columns
    ]

    columnas_faltantes = [
        columna
        for columna in COLUMNAS_INTERES
        if columna not in datos.columns
    ]

    if columnas_faltantes:
        raise ValueError(
            f"Faltan columnas: {columnas_faltantes}"
        )

    datos_filtrados = datos[
        COLUMNAS_INTERES
    ].copy()

    for columna in COLUMNAS_INTERES:

        datos_filtrados[columna] = pd.to_numeric(
            datos_filtrados[columna],
            errors="coerce",
        )

    datos_filtrados = datos_filtrados.dropna()

    datos_filtrados = datos_filtrados.reset_index(
        drop=True
    )

    return datos_filtrados


# ------------------------------------------------------------

def construir_registro_desde_csv(
    ruta_csv: str
) -> RegistroCSV:
    """
    Construye la estructura RegistroCSV.
    """

    return RegistroCSV(
        nombre_fichero=os.path.basename(ruta_csv),
        metadatos=cargar_metadatos(ruta_csv),
        datos=cargar_datos(ruta_csv),
    )


# ------------------------------------------------------------
# FRECUENCIA DE MUESTREO
# ------------------------------------------------------------

def obtener_frecuencia_muestreo(
    registro: RegistroCSV
) -> float:
    """
    Obtiene la frecuencia de muestreo.
    """

    fila = registro.metadatos[
        registro.metadatos["campo"] == CAMPO_FRECUENCIA
    ]

    if fila.empty:
        raise ValueError(
            "No se encontró Sampling Frequency."
        )

    valor = fila.iloc[0]["valor"]

    return float(valor)


# ------------------------------------------------------------
# CORRECCIÓN DE ACELERACIÓN
# ------------------------------------------------------------

def corregir_aceleracion(
    registro: RegistroCSV
) -> None:
    """
    Cambia el signo de la aceleración Z.
    """

    registro.datos["Linear_Acceleration_Z"] *= -1


# ------------------------------------------------------------
# MÉTRICAS TEMPORALES
# ------------------------------------------------------------

def calcular_longitud_temporal(
    total_muestras: int,
    frecuencia_muestreo: float,
) -> float:
    """
    Calcula el tiempo total del registro.
    """

    if frecuencia_muestreo <= 0:
        return 0.0

    return total_muestras / frecuencia_muestreo


# ------------------------------------------------------------
# VENTANA SYNC
# ------------------------------------------------------------

def buscar_indice_primera_sync(sync) -> int:
    """
    Busca la primera muestra Sync != 0.
    """

    for i, valor in enumerate(sync):

        if valor != 0:
            return i

    return -1


# ------------------------------------------------------------

def buscar_indice_ultima_sync(sync) -> int:
    """
    Busca la última muestra Sync != 0.
    """

    for i in range(len(sync) - 1, -1, -1):

        if sync.iloc[i] != 0:
            return i

    return -1


# ------------------------------------------------------------
# CONTEO DE PASOS
# ------------------------------------------------------------

def contar_transiciones_s3_s0(
    segmentation_output,
    inicio: int,
    fin: int,
) -> int:
    """
    Cuenta transiciones 3 -> 0.
    """

    if inicio < 0 or fin <= inicio:
        return 0

    pasos = 0

    for i in range(inicio, fin):

        actual = segmentation_output.iloc[i]
        siguiente = segmentation_output.iloc[i + 1]

        if actual == 3 and siguiente == 0:
            pasos += 1

    return pasos


# ------------------------------------------------------------
# VELOCIDAD DE MARCHA
# ------------------------------------------------------------

def calcular_velocidad_marcha(
    muestras_sync: int,
    frecuencia_muestreo: float,
    distancia_m: float = DISTANCIA_UTIL_10MWT_M,
) -> float:
    """
    Calcula la velocidad media de marcha.
    """

    if muestras_sync <= 0:
        return 0.0

    tiempo_sync = (
        muestras_sync / frecuencia_muestreo
    )

    return distancia_m / tiempo_sync


# ------------------------------------------------------------
# VELOCIDAD DE PASOS
# ------------------------------------------------------------

def calcular_velocidad_pasos(
    pasos: int,
    muestras_pasos: int,
    frecuencia_muestreo: float,
) -> float:
    """
    Calcula pasos por segundo.
    """

    if muestras_pasos <= 0:
        return 0.0

    tiempo = (
        muestras_pasos / frecuencia_muestreo
    )

    return pasos / tiempo


# ------------------------------------------------------------
# LONGITUD DE PASO
# ------------------------------------------------------------

def calcular_longitud_zancada(
    velocidad_marcha: float,
    velocidad_pasos: float,
) -> float:
    """
    Estima longitud media por paso.
    """

    if velocidad_pasos <= 0:
        return 0.0

    return (
        velocidad_marcha /
        velocidad_pasos
    )


# ------------------------------------------------------------
# MÉTRICAS COMPLETAS
# ------------------------------------------------------------

def calcular_metricas(
    registro: RegistroCSV,
    frecuencia_muestreo: float,
) -> dict[str, float | int]:

    inicio_sync = buscar_indice_primera_sync(
        registro.datos["Sync"]
    )

    fin_sync = buscar_indice_ultima_sync(
        registro.datos["Sync"]
    )

    muestras_sync = 0

    if inicio_sync >= 0 and fin_sync > inicio_sync:
        muestras_sync = (
            fin_sync - inicio_sync
        )

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


# ------------------------------------------------------------
# IMPRESIÓN DE RESULTADOS
# ------------------------------------------------------------

def imprimir_resultados(
    registro: RegistroCSV,
    frecuencia_muestreo: float,
    metricas: dict[str, float | int],
) -> None:

    print("\n========================================")
    print("RESULTADOS DEL ANÁLISIS")
    print("========================================")

    print(f"Archivo analizado: {registro.nombre_fichero}")

    print(f"Muestras leídas: {registro.total_muestras}")

    print(
        f"Frecuencia de muestreo: "
        f"{frecuencia_muestreo:.3f} Hz"
    )

    print(
        f"Tiempo total: "
        f"{metricas['tiempo_total']:.3f} s"
    )

    print(
        f"Muestras en Sync: "
        f"{metricas['muestras_sync']}"
    )

    print(
        f"Pasos detectados: "
        f"{metricas['pasos']}"
    )

    print(
        f"Velocidad de marcha: "
        f"{metricas['velocidad_marcha']:.3f} m/s"
    )

    print(
        f"Velocidad de pasos: "
        f"{metricas['velocidad_pasos']:.3f} pasos/s"
    )

    print(
        f"Longitud media por paso: "
        f"{metricas['longitud_zancada']:.3f} m"
    )

    print(
        "\nCorrección aplicada:"
        " cambio de signo en "
        "Linear_Acceleration_Z"
    )

    print("========================================\n")


# ------------------------------------------------------------
# FUNCIÓN PRINCIPAL
# ------------------------------------------------------------

def main() -> None:

    directorio_script = os.path.dirname(
        os.path.abspath(__file__)
    )

    db_path = os.path.abspath(
        os.path.join(
            directorio_script,
            "..",
            "..",
            "data",
            "raw",
            "gait",
        )
    )

    # --------------------------------------------------------
    # CARGA DE CSV
    # --------------------------------------------------------

    ficheros = cargar_csv(db_path)

    imprimir_resumen(
        db_path,
        len(ficheros),
    )

    print("Archivos disponibles:\n")

    for i, fichero in enumerate(ficheros):
        print(f"[{i}] {fichero}")

    # --------------------------------------------------------
    # SELECCIÓN DEL ARCHIVO
    # --------------------------------------------------------

    indice = int(
        input(
            "\nIngrese el índice del archivo: "
        )
    )

    ruta_csv = seleccionar_archivo_csv(
        db_path,
        ficheros,
        indice,
    )

    # --------------------------------------------------------
    # CONSTRUCCIÓN DEL REGISTRO
    # --------------------------------------------------------

    registro = construir_registro_desde_csv(
        ruta_csv
    )

    frecuencia_muestreo = (
        obtener_frecuencia_muestreo(
            registro
        )
    )

    # --------------------------------------------------------
    # CORRECCIÓN DE ACELERACIÓN
    # --------------------------------------------------------

    corregir_aceleracion(registro)

    # --------------------------------------------------------
    # CÁLCULO DE MÉTRICAS
    # --------------------------------------------------------

    metricas = calcular_metricas(
        registro,
        frecuencia_muestreo,
    )

    # --------------------------------------------------------
    # RESULTADOS
    # --------------------------------------------------------

    imprimir_resultados(
        registro,
        frecuencia_muestreo,
        metricas,
    )


# ------------------------------------------------------------

if __name__ == "__main__":
    main()