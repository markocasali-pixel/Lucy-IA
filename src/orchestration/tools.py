import os
import subprocess
import webbrowser
from pathlib import Path
import mss
import mss.tools
import psutil
import torch

WORKSPACE_DIR = Path("Data/workspace")
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = Path("Data/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_ocr_engine = None

def _get_ocr():
    global _ocr_engine
    if _ocr_engine is None:
        from rapidocr_onnxruntime import RapidOCR
        _ocr_engine = RapidOCR()
    return _ocr_engine

def _get_desktop_dir() -> Path:
    user_profile = Path(os.environ.get("USERPROFILE", Path.home()))
    onedrive_desktop = user_profile / "OneDrive" / "Desktop"
    if onedrive_desktop.exists():
        return onedrive_desktop
    local_desktop = user_profile / "Desktop"
    local_desktop.mkdir(parents=True, exist_ok=True)
    return local_desktop

def _resolve_target_path(filename: str, location: str = "workspace") -> Path:
    loc = (location or "workspace").lower()
    p = Path(filename)
    if p.is_absolute():
        return p
    if any(k in loc for k in ("desktop", "area de trabalho", "área de trabalho")) or filename.lower().startswith(("desktop/", "desktop\\")):
        clean_name = p.name
        return _get_desktop_dir() / clean_name
    return WORKSPACE_DIR / p.name

def web_search(query: str) -> str:
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, region="br-pt", max_results=4))
        if not results:
            return "Nenhum resultado encontrado na busca."
        snippets = [f"- {r.get('title')}: {r.get('body')}" for r in results]
        return "\n".join(snippets)
    except Exception as e:
        return f"Erro na pesquisa web: {e}"

def list_files(location: str = "workspace") -> str:
    target_dir = _get_desktop_dir() if "desktop" in (location or "").lower() else WORKSPACE_DIR
    files = [f.name for f in target_dir.iterdir() if f.is_file() and f.name != ".gitkeep"]
    if not files:
        return f"A pasta {target_dir} está vazia."
    return f"Arquivos em {target_dir}: " + ", ".join(files)

def write_or_edit_file(filename: str, content: str, mode: str = "w", location: str = "workspace") -> str:
    try:
        target = _resolve_target_path(filename, location)
        target.parent.mkdir(parents=True, exist_ok=True)
        write_mode = "a" if mode == "a" else "w"
        with open(target, write_mode, encoding="utf-8") as f:
            if write_mode == "a" and target.exists() and target.stat().st_size > 0:
                f.write("\n")
            f.write(content)
        return f"SUCESSO REAL: Arquivo '{target.name}' gravado em '{target}'."
    except Exception as e:
        return f"ERRO ao gravar arquivo: {e}"

def read_file(filename: str, location: str = "workspace") -> str:
    target = _resolve_target_path(filename, location)
    if not target.exists():
        alt = _get_desktop_dir() / Path(filename).name
        if alt.exists():
            target = alt
        else:
            return f"Arquivo '{filename}' não encontrado."
    try:
        if target.suffix.lower() == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(str(target))
            return "\n".join(page.extract_text() or "" for page in reader.pages)[:4000]
        if target.suffix.lower() == ".docx":
            import docx
            doc = docx.Document(str(target))
            return "\n".join(p.text for p in doc.paragraphs)[:4000]
        return target.read_text(encoding="utf-8")[:4000]
    except Exception as e:
        return f"Erro ao ler '{target.name}': {e}"

def open_file_in_os(filename: str, location: str = "workspace") -> str:
    target = _resolve_target_path(filename, location)
    if not target.exists():
        alt = _get_desktop_dir() / Path(filename).name
        if alt.exists():
            target = alt
        else:
            return f"Arquivo '{filename}' não encontrado."
    try:
        os.startfile(str(target))
        return f"SUCESSO REAL: Arquivo '{target}' aberto no Windows."
    except Exception as e:
        return f"Erro ao abrir arquivo: {e}"

def open_app(app_name: str) -> str:
    """Abre aplicativos do Windows ou URLs pelo nome."""
    name = (app_name or "").strip().lower()
    app_map = {
        "calculadora": "calc.exe",
        "calc": "calc.exe",
        "bloco de notas": "notepad.exe",
        "notepad": "notepad.exe",
        "paint": "mspaint.exe",
        "explorador": "explorer.exe",
        "explorer": "explorer.exe",
        "gerenciador de tarefas": "taskmgr.exe",
        "cmd": "cmd.exe",
        "powershell": "powershell.exe",
        "configuracoes": "ms-settings:",
        "configurações": "ms-settings:",
        "spotify": "spotify:",
        "discord": "discord:",
        "steam": "steam://open/main",
        "chrome": "chrome.exe",
        "edge": "msedge.exe",
        "youtube": "https://www.youtube.com",
        "google": "https://www.google.com",
    }
    try:
        if name.startswith(("http://", "https://")):
            webbrowser.open(name)
            return f"SUCESSO REAL: Site '{name}' aberto no navegador."
        for key, cmd in app_map.items():
            if key in name:
                if cmd.startswith(("http", "ms-", "spotify:", "discord:", "steam:")):
                    os.startfile(cmd)
                else:
                    subprocess.Popen(cmd, shell=True)
                return f"SUCESSO REAL: Aplicativo '{app_name}' ({cmd}) iniciado no Windows."
        subprocess.Popen(f'start "" "{app_name}"', shell=True)
        return f"SUCESSO REAL: Comando de abertura enviado para '{app_name}'."
    except Exception as e:
        return f"ERRO ao abrir '{app_name}': {e}"

def system_control(action: str) -> str:
    """Controla volume, mídia ou retorna status do PC/GPU."""
    act = (action or "").lower()
    try:
        if any(k in act for k in ("status", "gpu", "cpu", "ram", "sistema", "temperatura")):
            cpu_pct = psutil.cpu_percent(interval=0.3)
            ram = psutil.virtual_memory()
            gpu_info = "CUDA indisponível"
            if torch.cuda.is_available():
                vram_used = torch.cuda.memory_allocated(0) / (1024**3)
                vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                gpu_info = f"{torch.cuda.get_device_name(0)} ({vram_used:.1f}GB / {vram_total:.1f}GB VRAM)"
            return f"Status do Sistema -> CPU: {cpu_pct}% | RAM: {ram.percent}% ({ram.used/(1024**3):.1f}/{ram.total/(1024**3):.1f}GB) | GPU: {gpu_info}"

        import ctypes
        VK_VOLUME_MUTE = 0xAD
        VK_VOLUME_DOWN = 0xAE
        VK_VOLUME_UP = 0xAF
        VK_MEDIA_NEXT = 0xB0
        VK_MEDIA_PREV = 0xB1
        VK_MEDIA_PLAY_PAUSE = 0xB3

        def _press_key(vk, times=1):
            for _ in range(times):
                ctypes.windll.user32.keybd_event(vk, 0, 0, 0)
                ctypes.windll.user32.keybd_event(vk, 0, 2, 0)

        if "aumentar" in act or "subir" in act:
            _press_key(VK_VOLUME_UP, 5)
            return "SUCESSO REAL: Volume aumentado."
        if "diminuir" in act or "abaixar" in act or "baixar" in act:
            _press_key(VK_VOLUME_DOWN, 5)
            return "SUCESSO REAL: Volume reduzido."
        if "mutar" in act or "mudo" in act or "silenciar" in act:
            _press_key(VK_VOLUME_MUTE, 1)
            return "SUCESSO REAL: Mudo alternado."
        if "pausar" in act or "play" in act or "tocar" in act:
            _press_key(VK_MEDIA_PLAY_PAUSE, 1)
            return "SUCESSO REAL: Play/Pause de mídia acionado."
        if "proxima" in act or "próxima" in act or "avancar" in act:
            _press_key(VK_MEDIA_NEXT, 1)
            return "SUCESSO REAL: Próxima faixa acionada."
        return "Ação de sistema não reconhecida."
    except Exception as e:
        return f"ERRO no controle do sistema: {e}"

def read_screen_or_image(image_path: str | None = None) -> str:
    """Captura a tela atual (incluindo jogos) ou lê uma imagem enviada e extrai todo o texto via OCR."""
    try:
        target_img = image_path
        if not target_img or not Path(target_img).exists():
            snap_path = OUTPUT_DIR / "last_screen_capture.png"
            with mss.mss() as sct:
                monitor = sct.monitors[1]
                sct_img = sct.grab(monitor)
                mss.tools.to_png(sct_img.rgb, sct_img.size, output=str(snap_path))
            target_img = str(snap_path)

        ocr = _get_ocr()
        result, _ = ocr(target_img)
        if not result:
            return "Nenhum texto legível foi encontrado na tela/imagem."
        lines = [item[1] for item in result if len(item) >= 2 and item[1].strip()]
        extracted = "\n".join(lines)
        return f"TEXTO DETECTADO NA TELA/IMAGEM:\n{extracted[:3500]}"
    except Exception as e:
        return f"ERRO ao ler tela/imagem: {e}"
