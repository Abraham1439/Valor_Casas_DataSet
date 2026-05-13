"""
Custom Scikit-Learn transformers for the Casas RM project.
Includes structural cleaning and binary target generation.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    """Drops columns that cause data leakage or are irrelevant for modeling."""
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Elimina columnas como 'Link' o 'Dirección' que son ruido para el modelo
        return X.drop(columns=self.columns_to_drop, errors='ignore')

class ParkingToNumericTransformer(BaseEstimator, TransformerMixin):
    """Converts 'N_Estacionamientos' from text/mixed to numeric format."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        if 'N_Estacionamientos' in X_copy.columns:
            # Extraemos los números de cadenas como '2 estacionamientos'
            X_copy['N_Estacionamientos'] = pd.to_numeric(
                X_copy['N_Estacionamientos'].str.extract(r'(\d+)')[0], errors='coerce'
            ).fillna(0)
        return X_copy

class SurfaceToNumericTransformer(BaseEstimator, TransformerMixin):
    """Converts 'Superficie_Construida_M2' from string to float."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        if 'Superficie_Construida_M2' in X_copy.columns:
            # Limpiamos unidades y convertimos a decimal para el escalador
            X_copy['Superficie_Construida_M2'] = pd.to_numeric(
                X_copy['Superficie_Construida_M2'], errors='coerce'
            )
        return X_copy

class TargetCreatorTransformer(BaseEstimator, TransformerMixin):
    """
    Creates a binary target based on the median price (UF).
    Removes price columns to prevent target leakage.
    """
    def __init__(self, uf_col="Valor_UF"):
        self.uf_col = uf_col
        self.median_uf_ = None

    def fit(self, X, y=None):
        # Aprendemos la mediana del mercado SOLO del set de entrenamiento
        if self.uf_col in X.columns:
            self.median_uf_ = X[self.uf_col].median()
        return self

    def transform(self, X):
        X_copy = X.copy()
        if self.uf_col in X_copy.columns and self.median_uf_ is not None:
            # Creamos la clase: 1 (Premium/Sobre Mediana) | 0 (Estándar/Bajo Mediana)
            X_copy["precio_sobre_mediana"] = (X_copy[self.uf_col] > self.median_uf_).astype(int)
            
        # ELIMINACIÓN DE FUGAS: Borramos los precios originales para que el modelo no haga trampa
        return X_copy.drop(columns=[self.uf_col, "Valor_CLP"], errors='ignore')

class SmartImputerTransformer(BaseEstimator, TransformerMixin):
    """Adaptive imputer that uses Median for numeric and Mode for categorical data."""
    def __init__(self):
        self.impute_values_ = {}

    def fit(self, X, y=None):
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                self.impute_values_[col] = X[col].median()
            else:
                self.impute_values_[col] = X[col].mode()[0] if not X[col].mode().empty else 'Unknown'
        return self

    def transform(self, X):
        # Rellenamos huecos usando el conocimiento adquirido en el fit
        return X.fillna(self.impute_values_)