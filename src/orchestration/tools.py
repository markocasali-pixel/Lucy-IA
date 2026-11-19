import json
import os
import re
import subprocess
import threading
import time
import urllib.parse
import urllib.request
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
_SEARCH_CACHE: dict[str, tuple[float, str]] = {}
_CACHE_TTL_SEC = 120.0

_STOPWORDS_PT = {
    "qual", "quais", "que", "o", "a", "os", "as", "um", "uma", "de", "do", "da",
    "dos", "das", "em", "no", "na", "nos", "nas", "por", "para", "com", "sem",
    "sobre", "hoje", "atual", "atuais", "mais", "recentes", "novidades", "noticias",
    "notícias", "valor", "preco", "preço", "cotacao", "cotação", "quanto", "esta", "está"
}

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

def _is_clean_pt_text(text: str) -> bool:
    if not text or len(text.strip()) < 8:
        return False
    if re.search(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\u0400-\u04ff\u0600-\u06ff]', text):
        return False
    return True

def _is_relevant_to_query(line: str, query: str) -> bool:
    q_low = (query or "").lower()
    l_low = (line or "").lower()
    words = [w for w in re.findall(r"[a-zà-ú0-9]{2,}", q_low) if w not in _STOPWORDS_PT]
    if not words:
        return True
    # Expansão natural de termos chave (ex.: "ia" <-> "inteligência artificial")
    expanded = set(words)
    if "ia" in expanded or "inteligencia" in expanded or "inteligência" in expanded or "artificial" in expanded:
        expanded.update({"ia", "inteligência", "inteligencia", "artificial", "modelo", "modelos", "llm"})
    return any(w in l_low for w in expanded)

def _http_get_json(url: str, timeout_sec: float = 2.5) -> dict | list | None:
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) LucyIA/1.0",
                "Accept": "application/json"
            }
        )
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
            return json.loads(raw)
    except Exception:
        return None

def _search_currency_if_applicable(q_low: str) -> str | None:
    pairs = [
        (("dolar", "dólar", "usd"), "USD-BRL", "Dólar (USD/BRL)"),
        (("euro", "eur"), "EUR-BRL", "Euro (EUR/BRL)"),
        (("bitcoin", "btc"), "BTC-BRL", "Bitcoin (BTC/BRL)"),
    ]
    for keys, pair, label in pairs:
        if any(k in q_low for k in keys) and any(w in q_low for w in ("preco", "preço", "valor", "cotacao", "cotação", "quanto esta", "quanto está", "hoje", "atual")):
            data = _http_get_json(f"https://economia.awesomeapi.com.br/json/last/{pair}", timeout_sec=2.5)
            key = pair.replace("-", "")
            if isinstance(data, dict) and key in data and isinstance(data[key], dict):
                bid = data[key].get("bid")
                if bid:
                    return f"{label} agora: R$ {float(bid):.2f}"
    return None

def _search_ddg_instant(query: str, timeout_sec: float = 2.5) -> list[str]:
    q_enc = urllib.parse.quote_plus(query)
    data = _http_get_json(f"https://api.duckduckgo.com/?q={q_enc}&format=json&no_html=1&skip_disambig=1", timeout_sec=timeout_sec)
    out = []
    if isinstance(data, dict):
        if data.get("AbstractText") and _is_clean_pt_text(str(data["AbstractText"])):
            out.append(str(data["AbstractText"]).strip())
        for topic in (data.get("RelatedTopics") or [])[:4]:
            if isinstance(topic, dict) and topic.get("Text") and _is_clean_pt_text(str(topic["Text"])):
                out.append(str(topic["Text"]).strip())
    return out

def _search_wikipedia_pt(query: str, timeout_sec: float = 2.5) -> list[str]:
    q_enc = urllib.parse.quote_plus(query)
    data = _http_get_json(
        f"https://pt.wikipedia.org/w/api.php?action=query&list=search&srsearch={q_enc}&utf8=&format=json&srlimit=2",
        timeout_sec=timeout_sec
    )
    out = []
    if isinstance(data, dict):
        items = ((data.get("query") or {}).get("search")) or []
        for it in items[:2]:
            title = it.get("title", "")
            snippet = it.get("snippet", "")
            clean_snip = re.sub(r'<[^>]+>', '', snippet)
            clean_snip = clean_snip.replace("&quot;", '"').replace("&#039;", "'")
            combined = f"{title}: {clean_snip}".strip(": ")
            if _is_clean_pt_text(combined):
                out.append(combined)
    return out

def _search_ddgs_library(query: str, timeout_sec: float = 3.0) -> list[str]:
    box = {"items": []}

    def _run():
        try:
            from ddgs import DDGS
            with DDGS(timeout=max(2, int(timeout_sec))) as ddgs:
                results = list(ddgs.text(query, region="br-pt", max_results=4))
            for r in results:
                title = (r.get("title") or "").strip()
                body = (r.get("body") or "").strip()
                combined = f"{title}: {body}".strip(": ")
                if _is_clean_pt_text(combined):
                    box["items"].append(combined)
        except Exception:
            pass

    th = threading.Thread(target=_run, daemon=True)
    th.start()
    th.join(timeout=timeout_sec)
    return box["items"]

def web_search(query: str, timeout_sec: float = 3.5) -> str:
    clean_q = (query or "").strip()
    if not clean_q:
        return "Consulta vazia."

    now = time.time()
    cache_key = clean_q.lower()
    if cache_key in _SEARCH_CACHE:
        ts, cached_val = _SEARCH_CACHE[cache_key]
        if now - ts < _CACHE_TTL_SEC:
            return cached_val

    q_low = clean_q.lower()
    currency = _search_currency_if_applicable(q_low)
    if currency:
        _SEARCH_CACHE[cache_key] = (now, currency)
        return currency

    effective_q = clean_q
    if any(k in q_low for k in ("valor", "preco", "preço", "fipe", "carro", "veiculo", "veículo")) and "brasil" not in q_low and "fipe" not in q_low:
        effective_q = f"{clean_q} tabela fipe valor reais"

    collected: list[str] = []

    def _run_instant():
        collected.extend(_search_ddg_instant(effective_q, timeout_sec=2.2))

    def _run_wiki():
        collected.extend(_search_wikipedia_pt(clean_q, timeout_sec=2.2))

    def _run_ddgs():
        collected.extend(_search_ddgs_library(effective_q, timeout_sec=min(3.0, timeout_sec)))

    threads = [
        threading.Thread(target=_run_instant, daemon=True),
        threading.Thread(target=_run_wiki, daemon=True),
        threading.Thread(target=_run_ddgs, daemon=True),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=timeout_sec)

    unique_lines = []
    seen = set()
    for line in collected:
        if not _is_relevant_to_query(line, clean_q):
            continue
        norm = line.lower()
        if norm not in seen:
            seen.add(norm)
            unique_lines.append(f"- {line[:260]}")

    result_text = "\n".join(unique_lines[:4]) if unique_lines else "Não foi possível obter resultados confiáveis e relevantes agora."
    _SEARCH_CACHE[cache_key] = (now, result_text)
    return result_text

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
    act = (action or "").lower()
    try:
        if any(k in act for k in ("status", "gpu", "cpu", "ram", "sistema", "temperatura")):
            cpu_pct = psutil.cpu_percent(interval=0.2)
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
