"""
Pipeline construction for real estate classification.
Protects the binary target from being scaled and handles high-cardinality features.
"""

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from src.transformers import (
    DropColumnsTransformer, ParkingToNumericTransformer,
    SurfaceToNumericTransformer, DropHighMissingTransformer,
    OutlierCapper, SmartImputerTransformer, TargetCreatorTransformer
)

def build_preprocessing_pipeline(df, target_col='precio_sobre_mediana'):
    """Builds a scikit-learn pipeline for classification."""
    
    # Columnas que no aportan valor predictivo o son ruidosas
    COLUMNS_TO_DROP = ["Link", "Tipo_Vivienda", "Dirección", "Quién_Vende"]

    # PROTECCIÓN DEL TARGET: Funciones para aplicar escala solo a predictores reales
    def select_numeric(X):
        return [c for c in X.select_dtypes(include=['number']).columns if c != target_col]

    def select_categorical(X):
        return [c for c in X.select_dtypes(exclude=['number']).columns if c != target_col]

    # Procesamiento numérico con eliminación de varianza cero nativa
    num_pipe = Pipeline([
        ("capper", OutlierCapper(apply_capping=True)),
        ("zero_variance", VarianceThreshold(threshold=0.0)),
        ("scaler", StandardScaler()),
    ])

    # Procesamiento de categorías (Comunas, etc.)
    cat_pipe = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    # Ensamblaje: El target 'precio_sobre_mediana' pasa intacto por el remainder
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, select_numeric),
            ("cat", cat_pipe, select_categorical),
        ],
        remainder="passthrough"
    )

    return Pipeline([
        ("drop_leaks",     DropColumnsTransformer(columns_to_drop=COLUMNS_TO_DROP)),
        ("parking_fix",    ParkingToNumericTransformer()),
        ("surface_fix",    SurfaceToNumericTransformer()),
        ("target_creator", TargetCreatorTransformer()),
        ("drop_high_nan",  DropHighMissingTransformer(threshold=0.8)),
        ("smart_imputer",  SmartImputerTransformer()),
        ("preprocessing",  preprocessor)
    ])