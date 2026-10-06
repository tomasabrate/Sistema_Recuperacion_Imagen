import os
import subprocess
from pathlib import Path

def download_kaggle_dataset():
    """
    Downloads the required Kaggle dataset for the CBIR project.
    Requires kaggle CLI to be installed and authenticated.
    """
    repo_root = Path(__file__).resolve().parent.parent
    raw_data_dir = repo_root / "data" / "raw"
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    
    print("[INFO] Comprobando dependencias de Kaggle...")
    try:
        import kaggle
    except ImportError:
        print("[ERROR] La librería 'kaggle' no está instalada. Ejecuta 'pip install kaggle'.")
        return
        
    print("[INFO] Descargando dataset tpi-frc-utn-2025-primavera...")
    
    # Suponiendo que el dataset en Kaggle se llama "tpi-frc-utn-2025-primavera" 
    # (El nombre real del competidor/usuario puede variar, requeriría la ruta completa tipo user/dataset)
    # Por simplicidad, agregamos el comando estándar.
    command = [
        "kaggle", "datasets", "download", 
        "-d", "tpi-frc-utn-2025-primavera", # Reemplazar con el ID real del dataset si es diferente
        "-p", str(raw_data_dir),
        "--unzip"
    ]
    
    try:
        subprocess.run(command, check=True)
        print("[SUCCESS] Dataset descargado y extraído exitosamente.")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Falló la descarga del dataset: {e}")
        print("[INFO] Asegúrate de tener tu archivo kaggle.json en la ubicación correcta (~/.kaggle/kaggle.json).")

if __name__ == "__main__":
    download_kaggle_dataset()
