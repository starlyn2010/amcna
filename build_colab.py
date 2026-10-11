import json
import os

def create_notebook():
    cells = []
    
    # Title Markdown
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# AMCNA Phase 3: Benchmark en Kaggle (GPU T4)\n",
            "**Instrucción importante:** En la barra lateral derecha de Kaggle, asegúrate de cambiar **Accelerator** a **GPU T4 x2** antes de ejecutar."
        ]
    })
    
    # 1. Setup Cell: Install dependencies & create directories
    setup_code = """!pip install ncps scikit-learn scipy matplotlib numpy

import os
import sys

# Crear todas las carpetas necesarias
dirs = ["utils", "datasets", "models", "trainers", "experiments", "outputs", "figures", "data"]
for d in dirs:
    os.makedirs(d, exist_ok=True)

print("✅ Carpetas creadas correctamente.")
"""
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in setup_code.split("\n")]
    })
    
    files_to_include = [
        "utils/seed.py",
        "utils/metrics.py",
        "datasets/smnist.py",
        "models/amcna.py",
        "models/amcna_v2.py",
        "models/baselines.py",
        "models/baselines_extended.py",
        "trainers/trainer.py",
        "experiments/run_phase3_hpo.py",
        "experiments/run_phase3_main.py"
    ]
    
    for fpath in files_to_include:
        if os.path.exists(fpath):
            with open(fpath, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Use raw string or json.dumps for safest encoding inside python cell
            content_json = json.dumps(content)
            code_cell = f'# Escribiendo {fpath}\nwith open("{fpath}", "w", encoding="utf-8") as f:\n    f.write({content_json})\nprint("Escrito: {fpath}")'
            
            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in code_cell.split("\n")]
            })

    # Execution Cell
    run_code = """import sys
import os
import torch

sys.path.insert(0, os.path.abspath('.'))

print("==========================================")
print("GPU Disponible:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Dispositivo detectado:", torch.cuda.get_device_name(0))
else:
    print("⚠️ ADVERTENCIA: No se detectó GPU. Activa GPU T4 en la barra derecha de Kaggle (Accelerator -> GPU T4).")
print("==========================================")

# Crear dummy HPO para evitar la búsqueda lenta de hiperparámetros
import json
dummy_hpo = {
    "CfC": {"hidden_dim": 96, "lr": 0.003},
    "AMCNA_v2": {"hidden_dim": 96, "lr": 0.003},
    "LSTM": {"hidden_dim": 96, "lr": 0.003},
    "GRU": {"hidden_dim": 96, "lr": 0.003}
}
os.makedirs("outputs", exist_ok=True)
with open("outputs/hpo_results.json", "w") as f:
    json.dump(dummy_hpo, f)

# Ejecutar el experimento principal
!python experiments/run_phase3_main.py
"""
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in run_code.split("\n")]
    })

    notebook = {
        "cells": cells,
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"}
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }
    
    with open("AMCNA_Phase3_Kaggle.ipynb", 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)
    print("Notebook re-generated: AMCNA_Phase3_Kaggle.ipynb")

if __name__ == "__main__":
    create_notebook()
