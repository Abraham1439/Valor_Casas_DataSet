"""
Módulo de Optimización de Memoria.
Reduce el uso de memoria del DataFrame mediante downcasting de tipos numéricos
y procesamiento de archivos Excel grandes en chunks.
"""

import pandas as pd
import numpy as np


def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce el uso de memoria del DataFrame haciendo downcasting de columnas
    numéricas al tipo más pequeño posible sin pérdida de información.

    Estrategia:
    - Enteros: int64 → int32/int16/int8 según el rango de valores.
    - Decimales: float64 → float32 si el rango lo permite.
    - Strings: se convierten a 'category' si tienen baja cardinalidad (< 50% únicos).

    Args:
        df (pd.DataFrame): DataFrame original a optimizar.

    Returns:
        pd.DataFrame: DataFrame optimizado con menor uso de memoria.
    """
    try:
        original_mem = df.memory_usage(deep=True).sum() / 1024**2
        print(f"💾 Memoria original: {original_mem:.2f} MB")

        df_opt = df.copy()

        # --- Downcasting numérico ---
        for col in df_opt.select_dtypes(include=["int", "float"]).columns:
            try:
                c_min = df_opt[col].min()
                c_max = df_opt[col].max()
                orig_type = str(df_opt[col].dtype)

                if orig_type.startswith("int"):
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df_opt[col] = df_opt[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df_opt[col] = df_opt[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df_opt[col] = df_opt[col].astype(np.int32)

                elif orig_type.startswith("float"):
                    # Se verifica que no haya NaN antes de comparar rangos
                    if pd.notna(c_min) and pd.notna(c_max):
                        if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                            df_opt[col] = df_opt[col].astype(np.float32)

            except Exception as e:
                print(f"⚠️ ADVERTENCIA: No se pudo optimizar la columna '{col}': {e}")
                continue

        # --- Conversión a 'category' para strings de baja cardinalidad ---
        for col in df_opt.select_dtypes(include=["object", "string"]).columns:
            try:
                n_unique = df_opt[col].nunique()
                n_total = len(df_opt[col])
                if n_unique / n_total < 0.5:  # Menos del 50% de valores únicos
                    df_opt[col] = df_opt[col].astype("category")
            except Exception as e:
                print(f"⚠️ ADVERTENCIA: No se pudo categorizar la columna '{col}': {e}")
                continue

        final_mem = df_opt.memory_usage(deep=True).sum() / 1024**2
        savings = 100 * (original_mem - final_mem) / original_mem

        print(f"🚀 Memoria optimizada: {final_mem:.2f} MB")
        print(f"📉 Ahorro total: {savings:.1f}%")
        return df_opt

    except Exception as e:
        print(f"❌ ERROR CRÍTICO en optimización de memoria: {e}")
        return df


def load_excel_in_chunks(file_path, chunk_size: int = 300) -> pd.DataFrame:
    """
    Carga un archivo Excel en chunks para demostrar el manejo de
    archivos de gran escala (chunking). Combina los fragmentos al final.

    Justificación técnica: En datasets reales de miles o millones de filas,
    cargar todo en memoria puede agotar los recursos del sistema. El chunking
    permite procesar los datos de forma incremental.

    Args:
        file_path (Path): Ruta al archivo Excel.
        chunk_size (int): Número de filas por chunk. Default: 300.

    Returns:
        pd.DataFrame: DataFrame completo ensamblado desde los chunks.
    """
    try:
        # Leer para saber el total de filas
        df_full = pd.read_excel(file_path)
        total_rows = len(df_full)
        chunks = []

        print(f"📦 Cargando {total_rows} filas en chunks de {chunk_size}...")

        for start in range(0, total_rows, chunk_size):
            end = min(start + chunk_size, total_rows)
            chunk = df_full.iloc[start:end].copy()
            chunks.append(chunk)
            print(f"   Chunk {len(chunks)}: filas {start}–{end} ✓")

        df_combined = pd.concat(chunks, ignore_index=True)
        print(f"✅ Dataset ensamblado: {df_combined.shape[0]} filas × {df_combined.shape[1]} columnas")
        return df_combined

    except FileNotFoundError:
        print(f"❌ ERROR: Archivo no encontrado en {file_path}")
        raise
    except Exception as e:
        print(f"❌ ERROR al cargar en chunks: {e}")
        raise
