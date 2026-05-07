"""
Módulo de construcción del Pipeline de preprocesamiento.
Ensambla todos los transformadores custom en un Pipeline de scikit-learn
compatible con futuros modelos de ML.

Arquitectura del pipeline:
    1. drop_leaks        → Elimina columnas irrelevantes/data leakage semántico
    2. parking_fix       → Convierte 'N_Estacionamientos' de texto a numérico
    3. surface_fix       → Convierte 'Superficie_Construida_M2' de texto a numérico
    4. target_creator    → Crea variable objetivo + elimina Valor_UF y Valor_CLP
    5. drop_high_nan     → Elimina columnas con >80% nulos
    6. outlier_capper    → Winsorización IQR en columnas numéricas
    7. smart_imputer     → Imputación adaptativa de nulos restantes
    8. preprocessing     → ColumnTransformer: StandardScaler + OneHotEncoder
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold

from src.transformers import (
    DropColumnsTransformer,
    ParkingToNumericTransformer,
    SurfaceToNumericTransformer,
    DropHighMissingTransformer,
    OutlierCapper,
    SmartImputerTransformer,
    TargetCreatorTransformer,
)

# Columnas a eliminar por ser irrelevantes o causar data leakage semántico
COLUMNS_TO_DROP = [
    "Link",           # URL del aviso: no es característica de la propiedad
    "Tipo_Vivienda",  # Solo contiene 'Casa' → varianza cero semántica
    "Dirección",      # Texto libre granular, no generalizable
    "Quién_Vende",    # 303 valores únicos → ruido para el modelo
    "Corredor",       # Ídem a Quién_Vende
]


def build_preprocessing_pipeline() -> Pipeline:
    """
    Construye y retorna el pipeline completo de preprocesamiento para el
    dataset de casas usadas RM.

    El pipeline es genérico y reproducible: puede aplicarse a cualquier
    subconjunto del dataset (train/test) sin riesgo de data leakage,
    ya que los parámetros se aprenden solo en fit().

    Returns:
        Pipeline: Pipeline de scikit-learn listo para fit_transform().
    """

    # 1. Definimos el nombre exacto de la variable objetivo que crea el transformer
    TARGET_COL = "precio_sobre_mediana"

    # 2. Funciones dinámicas para proteger el Target del escalado
    def select_numeric(X):
        return [col for col in X.select_dtypes(include=["number"]).columns if col != TARGET_COL]

    def select_categorical(X):
        return [col for col in X.select_dtypes(exclude=["number"]).columns if col != TARGET_COL]

    # 3. Rutas de procesamiento
    num_pipe = Pipeline([
        ("capper", OutlierCapper(apply_capping=True)),
        ("zero_variance", VarianceThreshold(threshold=0.0)),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    # 4. Enrutador con passthrough para salvar el target
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, select_numeric),
            ("cat", cat_pipe, select_categorical),
        ],
        remainder="passthrough",
    )

    # 5. Pipeline maestro (Eliminamos el OutlierCapper redundante global)
    full_pipeline = Pipeline([
        ("drop_leaks",     DropColumnsTransformer(columns_to_drop=COLUMNS_TO_DROP)),
        ("parking_fix",    ParkingToNumericTransformer()),
        ("surface_fix",    SurfaceToNumericTransformer()),
        ("target_creator", TargetCreatorTransformer()),
        ("drop_high_nan",  DropHighMissingTransformer(threshold=0.8)),
        ("smart_imputer",  SmartImputerTransformer(low_threshold=0.10)),
        ("preprocessing",  preprocessor),
    ])

    return full_pipeline
