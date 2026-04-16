# Informe Técnico: Preparación de Datos
# Casas Usadas — Región Metropolitana, Mayo 2020 🏠

**Asignatura:** SCY1101 — Programación para la Ciencia de Datos  
**Evaluación:** Parcial N°1  
**Institución:** DuocUC  
**Fecha:** Mayo 2025  

---

## 1. Resumen Ejecutivo

Este proyecto implementa un pipeline profesional y reproducible de preparación de datos sobre el dataset de **casas usadas de la Región Metropolitana (Mayo 2020)**, compuesto por 1.139 registros y 13 variables originales.

El objetivo principal es transformar los datos crudos en un conjunto limpio, validado y normalizado que sirva como insumo para un futuro modelo de **clasificación binaria**: predecir si una propiedad tiene un precio **sobre o bajo la mediana del mercado** (expresada en UF), siguiendo una arquitectura equivalente a la del proyecto de referencia Bank Marketing.

Los resultados clave del proyecto son:
- Dataset procesado sin valores nulos, con outliers controlados y variables codificadas.
- Reducción de memoria del dataset mediante downcasting numérico.
- Verificación de integridad mediante hash SHA-256 del archivo original.
- Pipeline modular de scikit-learn con 8 pasos encadenados, completamente reproducible.

---

## 2. Análisis Exploratorio Inicial (EDA)

### 2.1 Estructura del Dataset

| Métrica | Valor |
|---|---|
| Total de registros | 1.139 |
| Total de columnas originales | 13 |
| Columnas numéricas | 5 |
| Columnas de texto | 8 |
| Comunas representadas | 41 |
| Período de datos | Mayo 2020 |

### 2.2 Estadísticas Descriptivas de Variables Clave

| Variable | Media | Mediana | Desv. Estándar | Mín | Máx |
|---|---|---|---|---|---|
| `Valor_UF` | 10.218 | 6.700 | 9.653 | 1.215 | 70.828 |
| `N_Habitaciones` | 4,11 | 4 | 1,60 | 1 | 19 |
| `N_Baños` | 2,65 | 2 | 1,39 | 1 | 12 |
| `Total_Superficie_M2` | variable | — | — | — | — |

### 2.3 Análisis de Valores Nulos

| Columna | Nulos | % |
|---|---|---|
| `N_Estacionamientos` | 72 | 6,3% |
| `Total_Superficie_M2` | 37 | 3,2% |
| `Superficie_Construida_M2` | 36 | 3,2% |
| `Dirección` | 37 | 3,2% |
| `N_Baños` | 21 | 1,8% |
| `N_Habitaciones` | 8 | 0,7% |

Ninguna columna supera el umbral del 80% de nulos, por lo que no se eliminó ninguna columna por este criterio. Las columnas `Dirección`, `Quién_Vende` y `Corredor` fueron eliminadas por razones de utilidad modelística, no por nulos.

### 2.4 Hallazgos Clave del EDA

**Desbalance de la variable objetivo:** La distribución resultó aproximadamente balanceada (~50/50) al usar la mediana como umbral, lo que es ideal para modelos de clasificación.

**Disparidad geográfica de precios:** Las comunas de Lo Barnechea (mediana: 27.690 UF), Las Condes (24.750 UF) y Vitacura (18.995 UF) están muy por encima de la mediana de la RM (6.700 UF), evidenciando una fuerte segmentación territorial.

**Correlaciones:** `Valor_UF` presenta correlación positiva con `N_Habitaciones`, `N_Baños`, `Total_Superficie_M2` y `Superficie_Construida_M2`, lo que valida su uso como base para la variable objetivo.

**Outliers extremos:** `Valor_UF` tiene un máximo de 70.828 UF contra una mediana de 6.700 UF, lo que confirma la necesidad de Winsorización.

---

## 3. Metodología de Transformación

### 3.1 Arquitectura del Pipeline

Se implementó un Pipeline de 8 pasos en scikit-learn, diseñado para ser secuencial, sin data leakage y reproducible:

```
df_raw
  ↓ [1] DropColumnsTransformer      → Elimina columnas irrelevantes
  ↓ [2] ParkingToNumericTransformer → Convierte 'No' → 0 en estacionamientos
  ↓ [3] SurfaceToNumericTransformer → Convierte superficie de object a float
  ↓ [4] TargetCreatorTransformer    → Crea precio_sobre_mediana, elimina UF/CLP
  ↓ [5] DropHighMissingTransformer  → Elimina columnas >80% nulos
  ↓ [6] OutlierCapper               → Winsorización IQR
  ↓ [7] SmartImputerTransformer     → Imputación adaptativa
  ↓ [8] ColumnTransformer           → StandardScaler + OneHotEncoder
  ↓
df_processed
```

### 3.2 Decisiones Técnicas y Justificaciones

**Eliminación de columnas (`Link`, `Tipo_Vivienda`, `Dirección`, `Quién_Vende`, `Corredor`):**
- `Link`: URL del aviso web. No es una característica intrínseca de la propiedad.
- `Tipo_Vivienda`: Contiene únicamente el valor 'Casa' en los 1.139 registros. Una columna constante no aporta información discriminatoria al modelo.
- `Dirección`: Texto libre con 1.102 valores únicos. No es generalizable.
- `Quién_Vende` y `Corredor`: 303 y ~200 valores únicos respectivamente. Alta cardinalidad que generaría ruido en el modelo.

**Eliminación de `Valor_CLP` (Data Leakage):**
`Valor_CLP` es una conversión directa de `Valor_UF` al tipo de cambio del período. Incluirla en el modelo sería data leakage puro: el modelo "vería" el precio en una unidad mientras predice si está sobre o bajo la mediana expresada en otra. Se elimina junto a `Valor_UF` tras crear el target.

**Winsorización IQR en lugar de eliminación de outliers:**
El dataset contiene casas del mercado de lujo en Lo Barnechea y Las Condes con precios de hasta 70.828 UF. Eliminar estos registros implicaría sesgar el dataset hacia el mercado medio-bajo. La Winsorización recorta los valores al límite del rango IQR (Q3 + 1.5×IQR) sin eliminar la observación, preservando la información demográfica del comprador.

**Imputación por mediana (no media):**
El dataset presenta distribuciones sesgadas a la derecha en variables de precio y superficie. La mediana es más robusta ante este tipo de sesgo, ya que no es arrastrada por los valores extremos como lo sería la media aritmética.

**Variable objetivo basada en mediana de mercado:**
Se usa la mediana (6.700 UF) y no la media (10.218 UF) como umbral de clasificación, porque la mediana divide el dataset en dos grupos de tamaño aproximadamente igual, produciendo un balance de clases óptimo para el entrenamiento de modelos.

### 3.3 Conversión de Tipos de Datos

**`N_Estacionamientos`:** La columna mezcla strings `'No'` (representando 0 estacionamientos) con valores numéricos (`'1'`, `'2'`, etc.), lo que lleva a pandas a inferir el tipo `object`. Se aplica un reemplazo explícito `'No' → '0'` seguido de `pd.to_numeric(..., errors='coerce')`, convirtiendo cualquier valor irreconocible a NaN para ser imputado después.

**`Superficie_Construida_M2`:** Inferida como `object` por pandas debido a valores no numéricos en algunos registros. Se aplica `pd.to_numeric(..., errors='coerce')` de la misma manera.

---

## 4. Resultados y Validación Técnica

### 4.1 Verificación de Integridad (SHA-256)

El archivo original fue auditado generando su firma SHA-256 mediante el módulo `src/audit.py`. El hash se almacena en `data/raw/metadata.json`. En cada ejecución posterior del pipeline, el hash se recalcula y compara: si difieren, el pipeline se detiene para proteger la reproducibilidad del análisis.

```json
{
  "file": "Casas_usadas_-_RM_Mayo_2020.xlsx",
  "hash_sha256": "<hash generado en primera ejecución>",
  "descripcion": "Dataset de casas usadas en la Región Metropolitana, Mayo 2020.",
  "fuente": "Portal inmobiliario chileno",
  "registros": 1139,
  "columnas": 13
}
```

### 4.2 Optimización de Memoria

El módulo `src/optimization.py` aplica downcasting numérico:
- Enteros: `int64` → `int8`/`int16`/`int32` según rango.
- Decimales: `float64` → `float32` si el rango lo permite.
- Strings de baja cardinalidad: `object` → `category` (< 50% valores únicos).

Adicionalmente, `load_excel_in_chunks()` demuestra el procesamiento a gran escala cargando el dataset en fragmentos de 300 filas, simulando el comportamiento necesario para datasets de millones de registros.

### 4.3 Validación del Dataset Procesado

Tras la ejecución completa del pipeline:

| Métrica | Valor |
|---|---|
| Valores nulos restantes | 0 |
| Filas conservadas | 1.139 (100%) |
| Columnas finales | Variables numéricas estandarizadas + dummies de comunas |
| Distribución de clases | ~50% clase 0 / ~50% clase 1 |

### 4.4 Evidencia de Calidad

- **Sin data leakage**: `Valor_UF` y `Valor_CLP` eliminadas antes del ColumnTransformer.
- **Sin nulos**: SmartImputer cubre todos los casos detectados.
- **Sin outliers distorsionadores**: Winsorización IQR aplicada a variables numéricas.
- **Variables estandarizadas**: Media ≈ 0 y desviación estándar ≈ 1 en columnas numéricas (verificable vía `df_final.describe()`).

---

## 5. Conclusiones y Recomendaciones

### 5.1 Conclusiones

El proyecto logró transformar un dataset con múltiples problemas de calidad (tipos incorrectos, outliers, nulos, columnas irrelevantes) en un dataset limpio, validado y normalizado, listo para ser consumido por modelos de clasificación. La arquitectura en Pipeline garantiza reproducibilidad total: cualquier integrante del equipo puede clonar el repositorio, ejecutar `python main.py` y obtener exactamente el mismo resultado.

La variable objetivo `precio_sobre_mediana` captura efectivamente la segmentación del mercado inmobiliario de la RM, con una distribución balanceada que favorece el entrenamiento de modelos robustos.

### 5.2 Lecciones Aprendidas

- La detección temprana de tipos de datos incorrectos (`N_Estacionamientos` como object) es fundamental antes de aplicar cualquier transformación matemática.
- El uso de entornos virtuales y `requirements.txt` evitó conflictos de versiones durante el desarrollo colaborativo.
- La Winsorización demostró ser superior a la eliminación de outliers cuando se trabaja con mercados segmentados (lujo vs. básico).

### 5.3 Mejoras Futuras

- **Imputación por KNN**: Reemplazar la imputación por mediana en `SmartImputerTransformer` por `KNNImputer` de scikit-learn para columnas con >10% de nulos.
- **Feature Engineering**: Crear variables derivadas como `ratio_construida_total` (`Superficie_Construida_M2 / Total_Superficie_M2`) que puede ser un predictor más informativo que las superficies por separado.
- **Georreferenciación**: Enriquecer el dataset con datos externos de las comunas (ingreso per cápita, distancia al centro) para mejorar el poder predictivo.
- **Modelo baseline**: Entrenar un modelo de Regresión Logística como punto de partida para validar que el pipeline produce features útiles.

---

*Informe generado como parte de la Evaluación Parcial N°1 — SCY1101, DuocUC 2025.*
