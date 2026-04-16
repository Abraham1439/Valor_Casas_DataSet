# Casas Usadas RM — Pipeline de Preparación de Datos 🏠

## Descripción del Proyecto

Pipeline profesional de Data Engineering para el dataset de **Casas Usadas de la Región Metropolitana (Mayo 2020)**. El proyecto limpia, transforma y prepara los datos para un modelo de **clasificación binaria** que predice si una propiedad tiene un precio sobre o bajo la mediana del mercado (en UF).

**Asignatura:** SCY1101 — Programación para la Ciencia de Datos  
**Evaluación:** Parcial N°1 | DuocUC 2025

---

## Estructura del Proyecto

```text
casas_rm_project/
├── data/
│   ├── raw/                          # Dataset original (inmutable)
│   │   ├── Casas_usadas_-_RM_Mayo_2020.xlsx
│   │   └── metadata.json             # Hash SHA-256 para auditoría
│   └── processed/
│       └── casas_rm_processed.csv    # Dataset limpio listo para ML
├── docs/
│   └── technical_report.md           # Informe técnico (8-12 páginas)
├── notebooks/
│   ├── 01_EDA.ipynb                  # Análisis exploratorio completo
│   └── 02_Pipelines.ipynb            # Construcción y validación del pipeline
├── src/
│   ├── __init__.py
│   ├── audit.py                      # Verificación SHA-256 de integridad
│   ├── optimization.py               # Downcasting y carga en chunks
│   ├── transformers.py               # Transformadores custom de scikit-learn
│   └── pipeline.py                   # Ensamble del pipeline completo
├── outputs/                          # Visualizaciones y gráficos generados
├── main.py                           # Script orquestador principal
└── requirements.txt                  # Dependencias del proyecto
```

---

## Variable Objetivo

| Clase | Descripción | Criterio |
|---|---|---|
| `1` | Precio **sobre** la mediana | `Valor_UF > 6.700 UF` |
| `0` | Precio **bajo** la mediana | `Valor_UF ≤ 6.700 UF` |

La mediana del mercado (6.700 UF) divide el dataset en dos grupos aproximadamente iguales, produciendo un balance de clases óptimo.

---

## Configuración del Entorno

### Requisitos previos
- Python 3.10 o superior
- pip

### Pasos para replicar el entorno

**1. Clonar el repositorio:**
```bash
git clone <url-del-repositorio>
cd casas_rm_project
```

**2. Crear y activar el entorno virtual:**
```bash
# En Linux / macOS:
python3 -m venv .venv
source .venv/bin/activate

# En Windows:
python -m venv .venv
.venv\Scripts\activate
```

**3. Instalar dependencias:**
```bash
pip install -r requirements.txt
```

**4. Verificar instalación:**
```bash
python -c "import pandas, numpy, sklearn, seaborn; print('✅ Todo instalado correctamente')"
```

---

## Ejecución

### Pipeline completo (recomendado)
```bash
python main.py
```

El script ejecuta automáticamente:
1. Auditoría SHA-256 del dataset original
2. Carga en chunks (simulación de gran escala)
3. Optimización de memoria
4. Pipeline de 8 pasos de transformación
5. Guardado del dataset procesado

### Notebooks interactivos
```bash
jupyter notebook
```
- `notebooks/01_EDA.ipynb` → Análisis exploratorio
- `notebooks/02_Pipelines.ipynb` → Pipeline y validación

---

## Arquitectura del Pipeline

```
df_raw (1.139 filas × 13 columnas)
  │
  ├─[1] DropColumnsTransformer      → Elimina Link, Tipo_Vivienda, Dirección, etc.
  ├─[2] ParkingToNumericTransformer → Convierte 'No' → 0 en N_Estacionamientos
  ├─[3] SurfaceToNumericTransformer → Convierte Superficie_Construida_M2 a float
  ├─[4] TargetCreatorTransformer    → Crea precio_sobre_mediana, elimina Valor_UF/CLP
  ├─[5] DropHighMissingTransformer  → Elimina columnas con >80% de nulos
  ├─[6] OutlierCapper               → Winsorización IQR en variables numéricas
  ├─[7] SmartImputerTransformer     → Imputación por mediana/moda adaptativa
  └─[8] ColumnTransformer
            ├── StandardScaler     → Variables numéricas estandarizadas
            └── OneHotEncoder      → Comunas codificadas como dummies
  │
df_processed (listo para ML)
```

---

## Dependencias

```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
openpyxl>=3.1.0
jupyter>=1.0.0
```
