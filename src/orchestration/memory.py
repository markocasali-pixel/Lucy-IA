import json
from datetime import datetime
from pathlib import Path

MEMORY_FILE = Path("Data/memory.json")
MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)

class MemoryManager:
    def __init__(self):
        self.data = self._load()

    def _load(self) -> dict:
        if MEMORY_FILE.exists():
            try:
                return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {
            "user_facts": [],
            "recent_conversations": []
        }

    def save(self):
        MEMORY_FILE.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    def add_fact(self, fact: str) -> str:
        fact = fact.strip()
        if not fact:
            return "Fato vazio ignorado."
        if fact.lower() not in [f.lower() for f in self.data["user_facts"]]:
            self.data["user_facts"].append(fact)
            self.data["user_facts"] = self.data["user_facts"][-40:]
            self.save()
        return f"Memória salva: {fact}"

    def log_turn(self, user_msg: str, assistant_msg: str):
        self.data["recent_conversations"].append({
            "timestamp": datetime.now().strftime("%d/%m %H:%M"),
            "user": user_msg[:300],
            "lucy": assistant_msg[:300]
        })
        self.data["recent_conversations"] = self.data["recent_conversations"][-12:]
        self.save()

    def get_context_block(self) -> str:
        lines = []
        if self.data["user_facts"]:
            lines.append("FATOS LEMBRADOS SOBRE O USUÁRIO:")
            for f in self.data["user_facts"][-15:]:
                lines.append(f"- {f}")
        if self.data["recent_conversations"]:
            lines.append("\nÚLTIMAS INTERAÇÕES ANTERIORES:")
            for c in self.data["recent_conversations"][-4:]:
                lines.append(f"[{c['timestamp']}] Usuário: {c['user']} | Lucy: {c['lucy']}")
        return "\n".join(lines)
