"""
Script principal de orquestación del pipeline de datos.
Casas Usadas - Región Metropolitana, Mayo 2020.

Flujo de ejecución:
    1. Auditoría de integridad del dataset (SHA-256)
    2. Carga del dataset en chunks (demostración de gran escala)
    3. Optimización de memoria (downcasting)
    4. Construcción y ejecución del pipeline de preprocesamiento
    5. Guardado del dataset procesado en data/processed/
"""

import pandas as pd
from pathlib import Path

from src.audit import audit_data
from src.optimization import optimize_memory, load_excel_in_chunks
from src.pipeline import build_preprocessing_pipeline


def main():
    """Función principal de orquestación del pipeline de ciencia de datos."""
    print("=" * 60)
    print("🏠 PIPELINE: CASAS USADAS REGIÓN METROPOLITANA 2020")
    print("=" * 60)

    try:
        # ── 1. AUDITORÍA DE INTEGRIDAD ────────────────────────────
        print("\n📋 PASO 1: Auditoría de integridad del dataset")
        print("-" * 40)
        if not audit_data():
            print("\n🛑 Pipeline detenido por fallo en auditoría.")
            return

        # ── 2. CARGA EN CHUNKS ────────────────────────────────────
        print("\n📦 PASO 2: Carga del dataset en chunks")
        print("-" * 40)
        raw_dir = Path("data/raw")
        excel_files = list(raw_dir.glob("*.xlsx"))

        if not excel_files:
            raise FileNotFoundError("No se encontró ningún archivo .xlsx en data/raw/")

        excel_path = excel_files[0]
        df_raw = load_excel_in_chunks(excel_path, chunk_size=300)

        # ── 3. OPTIMIZACIÓN DE MEMORIA ───────────────────────────
        print("\n⚙️  PASO 3: Optimización de memoria")
        print("-" * 40)
        df_opt = optimize_memory(df_raw)

        # ── 4. PIPELINE DE PREPROCESAMIENTO ──────────────────────
        print("\n🏗️  PASO 4: Construcción y ejecución del pipeline")
        print("-" * 40)
        pipeline = build_preprocessing_pipeline()

        print("\n🔄 Aplicando transformaciones...")
        processed_matrix = pipeline.fit_transform(df_opt)

        # ── 5. GUARDADO DEL RESULTADO ─────────────────────────────
        print("\n💾 PASO 5: Guardando dataset procesado")
        print("-" * 40)

        # Recuperar nombres de columnas del ColumnTransformer
        try:
            feature_names = pipeline.named_steps["preprocessing"].get_feature_names_out()
            feature_names = [
                name.replace("num__", "").replace("cat__", "")
                for name in feature_names
            ]
        except Exception:
            feature_names = [f"col_{i}" for i in range(processed_matrix.shape[1])]

        df_processed = pd.DataFrame(processed_matrix, columns=feature_names)

        processed_dir = Path("data/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)
        output_path = processed_dir / "casas_rm_processed.csv"

        df_processed.to_csv(output_path, index=False)

        # ── RESUMEN FINAL ─────────────────────────────────────────
        print("\n" + "=" * 60)
        print("✅ PIPELINE COMPLETADO EXITOSAMENTE")
        print("=" * 60)
        print(f"📂 Dataset guardado en: {output_path}")
        print(f"📊 Dimensiones finales: {df_processed.shape[0]} filas × {df_processed.shape[1]} columnas")
        print(f"📁 Columnas del dataset procesado:")
        for col in df_processed.columns[:10]:
            print(f"   • {col}")
        if len(df_processed.columns) > 10:
            print(f"   ... y {len(df_processed.columns) - 10} columnas más (OneHot de comunas)")

    # ── MANEJO DE EXCEPCIONES ESPECÍFICAS ────────────────────────
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: Archivo o directorio no encontrado → {e}")
        print("   Asegúrate de ejecutar este script desde la raíz del proyecto.")

    except pd.errors.EmptyDataError:
        print("\n❌ ERROR: El archivo de datos está completamente vacío.")

    except pd.errors.ParserError:
        print("\n❌ ERROR: No se pudo parsear el archivo. Revisa el formato.")

    except MemoryError:
        print("\n❌ ERROR: Memoria insuficiente. Reduce el tamaño del chunk en load_excel_in_chunks().")

    except Exception as e:
        print(f"\n❌ ERROR FATAL INESPERADO: El pipeline falló → {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
