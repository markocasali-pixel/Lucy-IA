import os
from pathlib import Path
import requests
from ddgs import DDGS
from pypdf import PdfReader
from docx import Document

WORKSPACE_DIR = Path("Data/workspace").resolve()
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

def _resolve_safe_path(filename: str) -> Path:
    target = (WORKSPACE_DIR / filename).resolve()
    if not str(target).startswith(str(WORKSPACE_DIR)):
        raise PermissionError("Acesso negado fora da pasta Data/workspace.")
    return target

def list_files() -> str:
    files = [f.name for f in WORKSPACE_DIR.iterdir() if f.is_file()]
    return f"Arquivos na pasta workspace: {', '.join(files)}" if files else "A pasta workspace está vazia."

def read_file(filename: str) -> str:
    try:
        path = _resolve_safe_path(filename)
        if not path.exists():
            return f"Erro: Arquivo '{filename}' não encontrado."
        ext = path.suffix.lower()
        if ext == ".pdf":
            reader = PdfReader(str(path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        elif ext == ".docx":
            doc = Document(str(path))
            return "\n".join(p.text for p in doc.paragraphs)
        return path.read_text(encoding="utf-8")
    except Exception as e:
        return f"Erro ao ler arquivo: {str(e)}"

def write_or_edit_file(filename: str, content: str, mode: str = "w") -> str:
    try:
        path = _resolve_safe_path(filename)
        if path.suffix.lower() == ".docx":
            doc = Document(str(path)) if (mode == "a" and path.exists()) else Document()
            for line in content.split("\n"):
                doc.add_paragraph(line)
            doc.save(str(path))
        else:
            with open(path, mode, encoding="utf-8") as f:
                f.write(content)
        return f"Arquivo '{filename}' salvo com sucesso em Data/workspace."
    except Exception as e:
        return f"Erro ao salvar arquivo: {str(e)}"

def open_file_in_os(filename: str) -> str:
    try:
        path = _resolve_safe_path(filename)
        if not path.exists():
            return f"Erro: Arquivo '{filename}' não encontrado."
        os.startfile(str(path))
        return f"Arquivo '{filename}' aberto no Windows."
    except Exception as e:
        return f"Erro ao abrir arquivo: {str(e)}"

def web_search(query: str) -> str:
    try:
        q_lower = query.lower()
        # Consulta direta de cotação se envolver dólar, euro ou bitcoin
        if any(k in q_lower for k in ["dolar", "dólar", "euro", "bitcoin", "cotação", "cotacao", "usd"]):
            res = requests.get("https://economia.awesomeapi.com.br/last/USD-BRL,EUR-BRL,BTC-BRL", timeout=8).json()
            usd = res.get("USDBRL", {})
            eur = res.get("EURBRL", {})
            if usd:
                return (
                    f"Cotação atualizada ({usd.get('create_date')}): "
                    f"Dólar Comercial (USD/BRL) = R$ {float(usd['bid']):.2f} | "
                    f"Euro (EUR/BRL) = R$ {float(eur['bid']):.2f}."
                )

        with DDGS() as ddgs:
            results = list(ddgs.text(query, region="br-pt", max_results=4))
        if not results:
            return "Nenhum resultado encontrado na web para essa busca."
        summary = "\n".join(f"- {r['title']}: {r['body']}" for r in results)
        return f"Resultados da pesquisa na internet:\n{summary}"
    except Exception as e:
        return f"Erro ao consultar a internet: {str(e)}"
