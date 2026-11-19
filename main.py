import os
import sys
from pathlib import Path
import webbrowser
from threading import Timer

# Registra as DLLs CUDA do PyTorch/NVIDIA no Windows antes de carregar o llama-cpp-python
venv_site = Path(sys.prefix) / "Lib" / "site-packages"
dll_dirs = [
    venv_site / "torch" / "lib",
    venv_site / "nvidia" / "cublas" / "bin",
    venv_site / "nvidia" / "cuda_runtime" / "bin",
]
for d in dll_dirs:
    if d.exists():
        os.add_dll_directory(str(d))
        os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")

from src.orchestration.engine import LucyEngine
from src.desktop.ui import build_ui

MODEL_PATH = "Data/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
URL = "http://127.0.0.1:7860"

def open_browser():
    webbrowser.open_new(URL)

if __name__ == "__main__":
    print("Iniciando Lucy IA na RTX 3070 Ti...")
    engine = LucyEngine(model_path=MODEL_PATH)
    app, custom_css = build_ui(engine)
    
    Timer(1.5, open_browser).start()
    
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        css=custom_css,
        inbrowser=False
    )
