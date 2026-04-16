"""
Transformadores Custom de Scikit-Learn para el proyecto Casas RM.
Cada clase hereda de BaseEstimator y TransformerMixin para ser compatible
con Pipeline y GridSearchCV de scikit-learn.

Transformadores implementados:
- DropColumnsTransformer     : Elimina columnas que causan data leakage o no aportan.
- ParkingToNumericTransformer: Convierte 'N_Estacionamientos' de texto a número.
- SurfaceToNumericTransformer: Convierte 'Superficie_Construida_M2' de texto a número.
- DropHighMissingTransformer : Elimina columnas con más de X% de nulos.
- OutlierCapper              : Limita outliers usando el método IQR (Winsorización).
- DropZeroVarianceTransformer: Elimina columnas con varianza cero (constantes).
- SmartImputerTransformer    : Imputa nulos con estrategia adaptativa según % de nulos.
- TargetCreatorTransformer   : Crea la variable objetivo binaria basada en la mediana de UF.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    """
    Elimina columnas del DataFrame para evitar Data Leakage u obsolescencia.

    Justificación de columnas eliminadas en este proyecto:
    - 'Link'         : URL del aviso, no es una característica de la propiedad.
    - 'Tipo_Vivienda': Contiene solo 'Casa' en todos los registros (varianza cero semántica).
    - 'Dirección'    : Texto libre, demasiado granular para modelar directamente.
    - 'Quién_Vende'  : 303 valores únicos, no generaliza; es ruido para el modelo.
    - 'Corredor'     : Ídem a Quién_Vende.
    - 'Valor_CLP'    : Es una transformación directa de Valor_UF → data leakage puro.
    """

    def __init__(self, columns_to_drop: list):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        cols_existentes = [c for c in self.columns_to_drop if c in X_copy.columns]
        dropped = X_copy.drop(columns=cols_existentes)
        print(f"   🗑️  DropColumns: eliminadas {cols_existentes}")
        return dropped


class ParkingToNumericTransformer(BaseEstimator, TransformerMixin):
    """
    Convierte la columna 'N_Estacionamientos' a tipo numérico.

    Problema detectado en EDA: la columna mezcla strings 'No' (indica 0 estacionamientos)
    con valores numéricos como '1', '2', '3'. Esto impide operaciones matemáticas.

    Estrategia:
    - 'No' → 0
    - Valores numéricos → int
    - Cualquier otro string → NaN (para ser imputado después)
    """

    def __init__(self, col: str = "N_Estacionamientos"):
        self.col = col

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        if self.col in X_copy.columns:
            X_copy[self.col] = (
                X_copy[self.col]
                .astype(str)
                .str.strip()
                .replace("No", "0")
                .replace("nan", np.nan)
            )
            X_copy[self.col] = pd.to_numeric(X_copy[self.col], errors="coerce")
            print(f"   🔢 ParkingToNumeric: '{self.col}' convertida a numérico.")
        return X_copy


class SurfaceToNumericTransformer(BaseEstimator, TransformerMixin):
    """
    Convierte la columna 'Superficie_Construida_M2' de tipo object a numérico.

    Problema detectado en EDA: la columna fue importada como 'object' (texto)
    en lugar de float64, posiblemente por caracteres no numéricos en algunos registros.

    Estrategia: pd.to_numeric con errors='coerce' → convierte lo que puede,
    el resto lo deja como NaN para imputación posterior.
    """

    def __init__(self, col: str = "Superficie_Construida_M2"):
        self.col = col

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        if self.col in X_copy.columns:
            X_copy[self.col] = pd.to_numeric(X_copy[self.col], errors="coerce")
            print(f"   🔢 SurfaceToNumeric: '{self.col}' convertida a numérico.")
        return X_copy


class DropHighMissingTransformer(BaseEstimator, TransformerMixin):
    """
    Elimina columnas que superan un umbral de valores nulos.

    Justificación: columnas con más del 80% de nulos no pueden imputarse
    de forma confiable sin introducir sesgos significativos en el modelo.
    """

    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
        self.cols_to_drop_ = []

    def fit(self, X: pd.DataFrame, y=None):
        pct_nulos = X.isnull().mean()
        self.cols_to_drop_ = pct_nulos[pct_nulos > self.threshold].index.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        cols = [c for c in self.cols_to_drop_ if c in X_copy.columns]
        if cols:
            print(f"   🗑️  DropHighMissing (>{self.threshold*100:.0f}% nulos): {cols}")
        return X_copy.drop(columns=cols)


class OutlierCapper(BaseEstimator, TransformerMixin):
    """
    Limita outliers numéricos usando el método IQR (Winsorización).

    Justificación técnica:
    - El dataset de casas RM tiene outliers extremos en Valor_UF (max 70.828 vs mediana 6.700).
    - Eliminar esas filas implicaría perder clientes reales del mercado de lujo.
    - La Winsorización recorta los valores extremos al límite del rango aceptable
      sin eliminar la observación, preservando la cantidad de datos.

    Límites IQR:
        Inferior = Q1 - 1.5 * IQR
        Superior = Q3 + 1.5 * IQR
    """

    def __init__(self, apply_capping: bool = True):
        self.apply_capping = apply_capping
        self.bounds_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        if not self.apply_capping:
            return self
        for col in X.select_dtypes(include=["number"]).columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            self.bounds_[col] = (Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        if not self.apply_capping:
            return X_copy
        for col, (lower, upper) in self.bounds_.items():
            if col in X_copy.columns:
                before = X_copy[col].copy()
                X_copy[col] = np.clip(X_copy[col], lower, upper)
                n_capped = (before != X_copy[col]).sum()
                if n_capped > 0:
                    print(f"   ✂️  OutlierCapper: '{col}' → {n_capped} valores recortados "
                          f"[{lower:.1f}, {upper:.1f}]")
        return X_copy

    def get_feature_names_out(self, input_features=None):
        return input_features


class DropZeroVarianceTransformer(BaseEstimator, TransformerMixin):
    """
    Elimina columnas numéricas con varianza cero (valores constantes).

    Justificación: una columna constante no aporta información discriminatoria
    al modelo, y su desviación estándar de 0 puede causar errores en StandardScaler.
    """

    def __init__(self):
        self.cols_to_drop_ = []

    def fit(self, X: pd.DataFrame, y=None):
        num_cols = X.select_dtypes(include=["number"]).columns
        self.cols_to_drop_ = [col for col in num_cols if X[col].std() == 0]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        cols = [c for c in self.cols_to_drop_ if c in X_copy.columns]
        if cols:
            print(f"   🗑️  DropZeroVariance: {cols}")
        return X_copy.drop(columns=cols)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return None
        return np.array([f for f in input_features if f not in self.cols_to_drop_])


class SmartImputerTransformer(BaseEstimator, TransformerMixin):
    """
    Imputa valores nulos con estrategia adaptativa según el porcentaje de nulos:

    - < 10% de nulos  → Imputación simple: mediana (numérico) o moda (categórico).
    - 10% - 80% nulos → Imputación compleja: mediana/moda como fallback
                        (en producción se reemplazaría por KNN Imputer).
    - > 80% de nulos  → Ignorado (ya fueron eliminados por DropHighMissingTransformer).

    Justificación de mediana vs media:
    - Se prefiere la mediana porque el dataset tiene distribuciones sesgadas
      (outliers en superficies y precios). La media sería arrastrada por extremos.
    """

    def __init__(self, low_threshold: float = 0.10):
        self.low_threshold = low_threshold
        self.cols_simples_ = []
        self.cols_complejas_ = []
        self.imputation_values_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        pct_nulos = X.isnull().mean()
        self.cols_simples_ = []
        self.cols_complejas_ = []

        for col in X.columns:
            pct = pct_nulos[col]
            if 0 < pct <= self.low_threshold:
                self.cols_simples_.append(col)
            elif self.low_threshold < pct <= 0.8:
                self.cols_complejas_.append(col)

        # Aprender valores de imputación durante fit
        for col in self.cols_simples_ + self.cols_complejas_:
            if pd.api.types.is_numeric_dtype(X[col]):
                self.imputation_values_[col] = X[col].median()
            else:
                mode_vals = X[col].mode()
                self.imputation_values_[col] = mode_vals[0] if len(mode_vals) > 0 else "desconocido"

        print(f"   🧠 SmartImputer - Simples (<10%): {self.cols_simples_}")
        print(f"   🚧 SmartImputer - Complejas (10-80%): {self.cols_complejas_}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        for col in self.cols_simples_ + self.cols_complejas_:
            if col in X_copy.columns and col in self.imputation_values_:
                n_nulos = X_copy[col].isnull().sum()
                if n_nulos > 0:
                    X_copy[col] = X_copy[col].fillna(self.imputation_values_[col])
                    print(f"   💉 Imputados {n_nulos} nulos en '{col}' "
                          f"con {self.imputation_values_[col]:.2f}"
                          if pd.api.types.is_numeric_dtype(X_copy[col])
                          else f"   💉 Imputados {n_nulos} nulos en '{col}'")
        return X_copy

    def get_feature_names_out(self, input_features=None):
        return input_features


class TargetCreatorTransformer(BaseEstimator, TransformerMixin):
    """
    Crea la variable objetivo binaria 'precio_sobre_mediana' basada en Valor_UF.

    Variable objetivo:
        1 → La casa tiene un precio SOBRE la mediana del mercado (Valor_UF > mediana).
        0 → La casa tiene un precio BAJO o IGUAL a la mediana del mercado.

    Justificación:
    - Transforma un problema de regresión en clasificación binaria, replicando
      la estructura del proyecto de referencia (Bank Marketing: yes/no).
    - La mediana se aprende en fit() para evitar data leakage: en producción,
      la mediana se calcularía solo con datos de entrenamiento.

    Columnas eliminadas tras crear el target:
    - 'Valor_UF' : Es la fuente directa del target → data leakage total.
    - 'Valor_CLP': Es conversión directa de Valor_UF → data leakage total.
    """

    def __init__(self, uf_col: str = "Valor_UF", clp_col: str = "Valor_CLP"):
        self.uf_col = uf_col
        self.clp_col = clp_col
        self.median_uf_ = None

    def fit(self, X: pd.DataFrame, y=None):
        if self.uf_col in X.columns:
            self.median_uf_ = X[self.uf_col].median()
            print(f"   🎯 TargetCreator: mediana UF aprendida = {self.median_uf_:.2f} UF")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        if self.uf_col in X_copy.columns and self.median_uf_ is not None:
            X_copy["precio_sobre_mediana"] = (X_copy[self.uf_col] > self.median_uf_).astype(int)
            n_sobre = X_copy["precio_sobre_mediana"].sum()
            n_bajo = len(X_copy) - n_sobre
            print(f"   🎯 TargetCreator: {n_sobre} casas SOBRE mediana | {n_bajo} casas BAJO mediana")

        # Eliminar columnas con data leakage
        leakage_cols = [c for c in [self.uf_col, self.clp_col] if c in X_copy.columns]
        X_copy = X_copy.drop(columns=leakage_cols)
        print(f"   🚫 Leakage eliminado: {leakage_cols}")
        return X_copy

    def get_feature_names_out(self, input_features=None):
        return input_features
