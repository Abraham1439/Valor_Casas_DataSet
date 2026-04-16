"""
Módulo de Auditoría de Datos.
Verifica la integridad y procedencia del dataset usando hash SHA-256.
Garantiza que el archivo original no haya sido modificado o corrompido
entre distintas ejecuciones del pipeline.
"""

import hashlib
import json
from pathlib import Path


def generate_file_hash(file_path: Path) -> str | None:
    """
    Genera una firma SHA-256 única para el archivo, leyéndolo en bloques
    para no cargar archivos grandes en memoria de una sola vez.

    Args:
        file_path (Path): Ruta al archivo a auditar.

    Returns:
        str | None: Hash SHA-256 en hexadecimal, o None si ocurrió un error.
    """
    try:
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except IOError as e:
        print(f"❌ ERROR: No se pudo leer el archivo para auditar: {e}")
        return None


def audit_data() -> bool:
    """
    Busca el archivo de datos en data/raw/, genera su hash SHA-256 y lo
    compara contra el metadata.json guardado previamente.

    - Si metadata.json NO existe: lo crea con el hash actual (primera ejecución).
    - Si metadata.json SÍ existe: compara el hash guardado con el calculado.
      Si difieren, el archivo fue modificado o corrompido → detiene el pipeline.

    Returns:
        bool: True si la auditoría es exitosa, False si falla.
    """
    try:
        raw_dir = Path("data/raw")
        # Acepta tanto .xlsx como .csv
        data_files = list(raw_dir.glob("*.xlsx")) + list(raw_dir.glob("*.csv"))

        if not data_files:
            print("❌ ERROR: No se encontró ningún archivo de datos en data/raw/")
            return False

        target_file = data_files[0]
        metadata_path = raw_dir / "metadata.json"

        print(f"🔍 Auditando archivo: {target_file.name}")
        calculated_hash = generate_file_hash(target_file)

        if not calculated_hash:
            return False

        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                saved_metadata = json.load(f)

            saved_hash = saved_metadata.get("hash_sha256", "")
            if saved_hash == calculated_hash:
                print("✅ ÉXITO: Integridad verificada. El archivo no ha sido alterado.")
                print(f"   SHA-256: {calculated_hash[:32]}...")
                return True
            else:
                print("🚨 ERROR CRÍTICO: El hash no coincide. El dataset fue modificado o corrompido.")
                print(f"   Hash esperado : {saved_hash[:32]}...")
                print(f"   Hash calculado: {calculated_hash[:32]}...")
                return False
        else:
            # Primera ejecución: creamos el metadata
            new_metadata = {
                "file": target_file.name,
                "hash_sha256": calculated_hash,
                "descripcion": "Dataset de casas usadas en la Región Metropolitana, Mayo 2020.",
                "fuente": "Portal inmobiliario chileno",
                "registros": 1139,
                "columnas": 13
            }
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(new_metadata, f, indent=4, ensure_ascii=False)
            print("📝 Metadata inicial creado exitosamente.")
            print(f"   SHA-256: {calculated_hash[:32]}...")
            return True

    except json.JSONDecodeError:
        print("❌ ERROR: El archivo metadata.json está corrupto y no puede leerse.")
        return False
    except Exception as e:
        print(f"❌ ERROR INESPERADO durante la auditoría: {e}")
        return False
