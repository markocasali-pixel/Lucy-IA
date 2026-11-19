import gc
import json
import random
import re
import threading
import traceback
import torch
from llama_cpp import Llama
from src.orchestration import tools
from src.orchestration.memory import MemoryManager
from src.orchestration.vision_gen import ImageGenerator
from src.orchestration.voice import VoiceEngine, DEFAULT_REF_PHRASE

IMAGE_DONE_REPLIES = [
    "Prontinho! Acabei de finalizar sua arte em alta definição. O que você achou do resultado?",
    "Olha só! Sua imagem ficou pronta e já está na tela. Quer ajustar algum detalhe ou testar outra seed?",
    "Missão cumprida! Renderizei a imagem exatamente como você pediu. Ficou demais, né?"
]

STOP_TOKENS = ["Usuário:", "User:", "Lucy:", "Assistente:", "\n\n\n"]

_PT_TO_EN_IMAGE_MAP = [
    (r"\bguerreira\b", "female warrior"),
    (r"\bguerreiro\b", "male warrior"),
    (r"\bvampira\b", "female vampire"),
    (r"\bvampiro\b", "male vampire"),
    (r"\bmenina\b|\bgarota\b", "anime girl"),
    (r"\bmenino\b|\bgaroto\b", "anime boy"),
    (r"\bmulher\b", "woman"),
    (r"\bhomem\b", "man"),
    (r"\bgato\b", "cat"),
    (r"\bcachorro\b", "dog"),
    (r"\bdragao\b|\bdragão\b", "dragon"),
    (r"\bcidade chuvosa\b", "rainy city"),
    (r"\bchuva\b|\bchuvosa\b", "rain, wet reflections"),
    (r"\bà noite\b|\ba noite\b|\bnoite\b", "at night"),
    (r"\biluminação neon\b|\biluminacao neon\b|\bneon\b", "neon lighting, glowing neon"),
    (r"\bciberpunk\b|\bcyberpunk\b", "cyberpunk"),
    (r"\bfuturista\b", "futuristic"),
    (r"\barmadura\b", "armor"),
    (r"\bespada\b", "sword"),
    (r"\basas\b", "wings"),
    (r"\bfloresta\b", "forest"),
    (r"\bcastelo\b", "castle"),
    (r"\bcyber\b", "cyber"),
]

def _sanitize_output(text: str) -> str:
    cleaned = re.sub(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\u0400-\u04ff\u0600-\u06ff]+', '', text)
    cleaned = re.sub(r'\[[^\]]{3,80}\]', '', cleaned)
    for marker in ("Usuário:", "User:", "Assistente:"):
        if marker in cleaned:
            cleaned = cleaned.split(marker)[0]
    cleaned = re.sub(r'^\s*Lucy\s*:\s*', '', cleaned.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r'\s{2,}', ' ', cleaned)
    return re.sub(r'\n{3,}', '\n\n', cleaned).strip()

def _has_placeholder_or_unverified_claim(text: str, tool_result: str) -> bool:
    low_t = (text or "").lower()
    low_r = (tool_result or "").lower()
    if "[" in text or "]" in text or "consulte a cotação" in low_t or "consulte a cotacao" in low_t:
        return True
    no_data_markers = (
        "não foi possível obter resultados",
        "nao foi possivel obter resultados",
        "nenhum resultado encontrado",
        "busca indisponível",
        "busca indisponivel",
        "tempo limite",
    )
    if any(m in low_r for m in no_data_markers):
        return True
    return False

class LucyEngine:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.llm = None
        self._llm_lock = threading.Lock()
        self._load_llm()
        self.memory = MemoryManager()
        self.image_gen = ImageGenerator()
        self.voice_engine = VoiceEngine()
        self.base_system_prompt = (
            "Você é Lucy IA, assistente pessoal espirituosa, calorosa e carismática (estilo J.A.R.V.I.S. + anime). "
            "Responda SEMPRE 100% em Português do Brasil natural, vivo e direto (1 a 2 frases curtas). "
            "NUNCA simule diálogos ('Usuário:', 'Lucy:'), NUNCA use colchetes como '[...]' e NUNCA invente preços ou valores ausentes.\n"
            "Para ações, responda APENAS JSON: {\"tool\": \"nome\", \"args\": {...}}.\n"
            "Ferramentas: "
            "open_app(app_name), "
            "write_or_edit_file(filename, content, mode, location='desktop'|'workspace'), "
            "read_file(filename, location), open_file_in_os(filename, location), list_files(location), "
            "read_screen_or_image(), system_control(action='status'|'aumentar volume'|'diminuir volume'|'mutar'|'pausar'|'proxima'), "
            "web_search(query), remember_fact(fact), generate_image(prompt)."
        )

    def _build_system_prompt(self) -> str:
        mem_ctx = self.memory.get_context_block()
        if mem_ctx:
            return f"{self.base_system_prompt}\n\n{mem_ctx}"
        return self.base_system_prompt

    def _load_llm(self):
        with self._llm_lock:
            if self.llm is None:
                self.llm = Llama(
                    model_path=self.model_path,
                    n_gpu_layers=-1,
                    n_ctx=4096,
                    n_threads=8,
                    use_mmap=True,
                    chat_format="chatml",
                    verbose=False
                )

    def _preload_llm_async(self):
        threading.Thread(target=self._load_llm, daemon=True).start()

    def _unload_all_models(self):
        with self._llm_lock:
            if self.llm is not None:
                del self.llm
                self.llm = None
        self.voice_engine.offload_to_ram()
        gc.collect()
        torch.cuda.empty_cache()

    def _normalize_content(self, content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    parts.append(str(block["text"]))
                elif isinstance(block, str):
                    parts.append(block)
            return " ".join(parts).strip()
        return str(content)

    def _extract_tool_call(self, text: str):
        known_tools = {
            "open_app", "write_or_edit_file", "read_file", "open_file_in_os",
            "list_files", "read_screen_or_image", "system_control",
            "web_search", "remember_fact", "generate_image"
        }
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
            if not isinstance(data, dict):
                return None
            if "tool" in data and data["tool"] in known_tools:
                return {"tool": data["tool"], "args": data.get("args", {})}
            for key in known_tools:
                if key in data:
                    val = data[key]
                    return {"tool": key, "args": val if isinstance(val, dict) else {}}
        except json.JSONDecodeError:
            return None
        return None

    def _prepare_english_image_prompt(self, raw_prompt: str, fallback_user_msg: str) -> str:
        src = (raw_prompt or fallback_user_msg or "").strip()
        cleaned = re.sub(
            r'^(crie|gere|faça|faca|desenhe|mostre)\s+(uma\s+|um\s+)?(imagem|arte|foto|ilustração|ilustracao)\s+(de\s+|do\s+|da\s+)?',
            '',
            src,
            flags=re.IGNORECASE
        ).strip() or src

        if not re.search(r'[áéíóúâêîôûãõçÁÉÍÓÚÂÊÎÔÛÃÕÇ]', cleaned) and re.search(r'\b(girl|boy|woman|man|warrior|vampire|cyberpunk|neon|city|cat|dragon)\b', cleaned.lower()):
            return f"{cleaned}, detailed face and subject focus, high detail"

        translated = cleaned.lower()
        for pat, rep in _PT_TO_EN_IMAGE_MAP:
            translated = re.sub(pat, rep, translated)

        translated = re.sub(r'\b(em|uma|um|de|do|da|com|e|no|na|para|por|ao|aos)\b', ' ', translated)
        tokens = [t.strip(" ,.") for t in re.split(r'[,\n]+', translated) if t.strip(" ,.")]

        subject_keywords = ("warrior", "vampire", "girl", "boy", "woman", "man", "cat", "dog", "dragon", "character", "personagem")
        priority = []
        others = []
        for tok in tokens:
            t_clean = re.sub(r'\s+', ' ', tok).strip()
            if not t_clean:
                continue
            if any(k in t_clean for k in subject_keywords):
                priority.append(f"{t_clean}, centered subject, detailed character")
            else:
                others.append(t_clean)

        ordered = priority + others
        if not ordered:
            ordered = ["detailed subject, dynamic lighting, high quality"]
        return ", ".join(dict.fromkeys(ordered))

    def _fallback_intent_router(self, user_msg: str):
        low = user_msg.lower()
        if any(k in low for k in ("traduza minha tela", "traduzir a tela", "o que esta na tela", "o que está na tela", "leia minha tela", "texto do jogo", "traduzir tela")):
            return {"tool": "read_screen_or_image", "args": {}}
        if any(k in low for k in ("gere uma imagem", "crie uma imagem", "crie a imagem", "faça uma imagem", "faca uma imagem", "desenhe", "gerar imagem", "criar imagem")):
            return {"tool": "generate_image", "args": {"prompt": user_msg}}
        if any(k in low for k in ("cotacao", "cotação", "preco", "preço", "valor", "fipe", "noticias", "notícias", "quanto esta", "quanto está")):
            return {"tool": "web_search", "args": {"query": user_msg}}
        if "calculadora" in low and any(v in low for v in ("abr", "inici", "exec")):
            return {"tool": "open_app", "args": {"app_name": "calculadora"}}
        if "bloco de notas" in low and any(v in low for v in ("abr", "inici")):
            return {"tool": "open_app", "args": {"app_name": "bloco de notas"}}
        return None

    def process_turn_stream(
        self,
        user_message: str,
        history: list,
        enable_voice: bool,
        library_voice: str | None,
        custom_voice_path: str | None,
        ref_text_hint: str = DEFAULT_REF_PHRASE,
        tts_backend: str = "Chatterbox PT-BR (Expressivo)",
        exaggeration: float = 0.65,
        image_model: str = "Anime (AnimaPencil XL)",
        image_styles: list[str] | None = None,
        image_resolution: str = "1920x1080 (Full HD 16:9 - Paisagem)",
        image_seed: int = -1,
        uploaded_image_path: str | None = None
    ):
        self._load_llm()
        messages = [{"role": "system", "content": self._build_system_prompt()}]
        for item in history[-4:]:
            if isinstance(item, dict) and item.get("content"):
                clean_hist = self._normalize_content(item["content"])
                if clean_hist:
                    messages.append({"role": item["role"], "content": clean_hist})

        effective_prompt = user_message
        if uploaded_image_path:
            effective_prompt = f"{user_message}\n[Imagem anexada: {uploaded_image_path}]"
        messages.append({"role": "user", "content": effective_prompt})

        direct_tool = self._fallback_intent_router(user_message)
        if uploaded_image_path:
            tool_data = {"tool": "read_screen_or_image", "args": {"image_path": uploaded_image_path}}
            raw_reply = json.dumps(tool_data, ensure_ascii=False)
        elif direct_tool:
            tool_data = direct_tool
            raw_reply = json.dumps(tool_data, ensure_ascii=False)
        else:
            response = self.llm.create_chat_completion(
                messages=messages,
                temperature=0.65,
                max_tokens=200,
                stop=STOP_TOKENS
            )
            raw_reply = (response["choices"][0]["message"].get("content") or "").strip()
            tool_data = self._extract_tool_call(raw_reply)

        print(f"[ENGINE] msg='{user_message[:70]}' | tool_data={tool_data}")

        final_image_path = None
        reply_text = ""
        generated_image = False

        if tool_data:
            fn_name = tool_data["tool"]
            args = tool_data.get("args", {})

            if fn_name == "generate_image":
                generated_image = True
                en_prompt = self._prepare_english_image_prompt(args.get("prompt", ""), user_message)
                print(f"[IMAGE] Iniciando geração | modelo={image_model} | prompt='{en_prompt}'")
                self._unload_all_models()
                try:
                    for preview_frame, done_path, status_msg in self.image_gen.generate_stream(
                        en_prompt,
                        model_label=image_model,
                        selected_styles=image_styles,
                        resolution_label=image_resolution,
                        seed=image_seed
                    ):
                        if done_path:
                            final_image_path = done_path
                        yield None, preview_frame, None, status_msg
                except Exception as e:
                    print(f"[ERRO IMAGE]: {e}")
                    traceback.print_exc()
                    final_image_path = None

                if final_image_path:
                    print(f"[IMAGE] Concluída em: {final_image_path}")
                    reply_text = random.choice(IMAGE_DONE_REPLIES)
                else:
                    print("[IMAGE] Falhou ou não retornou arquivo final.")
                    reply_text = "Tive um problema ao gerar a imagem agora e não consegui concluir o arquivo. Quer tentar novamente?"
            else:
                if fn_name == "open_app":
                    tool_result = tools.open_app(args.get("app_name", ""))
                elif fn_name == "write_or_edit_file":
                    loc = args.get("location", "desktop" if any(k in user_message.lower() for k in ("desktop", "área de trabalho", "area de trabalho")) else "workspace")
                    tool_result = tools.write_or_edit_file(
                        filename=args.get("filename", "arquivo.txt"),
                        content=args.get("content", ""),
                        mode=args.get("mode", "w"),
                        location=loc
                    )
                elif fn_name == "read_file":
                    tool_result = tools.read_file(**args)
                elif fn_name == "open_file_in_os":
                    tool_result = tools.open_file_in_os(**args)
                elif fn_name == "list_files":
                    tool_result = tools.list_files(**args)
                elif fn_name == "read_screen_or_image":
                    tool_result = tools.read_screen_or_image(args.get("image_path") or uploaded_image_path)
                elif fn_name == "system_control":
                    tool_result = tools.system_control(args.get("action", "status"))
                elif fn_name == "web_search":
                    tool_result = tools.web_search(args.get("query", user_message))
                elif fn_name == "remember_fact":
                    tool_result = self.memory.add_fact(args.get("fact", user_message))
                else:
                    tool_result = "Ação executada."

                if fn_name == "web_search" and _has_placeholder_or_unverified_claim("", tool_result):
                    reply_text = "Não consegui obter dados confiáveis e atualizados na internet agora para confirmar esse valor com segurança."
                else:
                    messages.append({"role": "assistant", "content": raw_reply})
                    messages.append({
                        "role": "user",
                        "content": (
                            f"[SISTEMA]: {tool_result}\n"
                            "Responda em 1 a 2 frases curtas, diretas e 100% em Português do Brasil. "
                            "Use APENAS o que consta em [SISTEMA]; nunca invente preços, faixas ou valores ausentes."
                        )
                    })
                    final_res = self.llm.create_chat_completion(
                        messages=messages,
                        temperature=0.55,
                        max_tokens=180,
                        stop=STOP_TOKENS
                    )
                    candidate = _sanitize_output(final_res["choices"][0]["message"].get("content") or "")
                    if fn_name == "web_search" and _has_placeholder_or_unverified_claim(candidate, tool_result):
                        reply_text = "Não encontrei um valor confirmado nos resultados da busca agora para responder com precisão."
                    else:
                        reply_text = candidate or "Tudo certo por aqui!"
        else:
            reply_text = _sanitize_output(raw_reply) if raw_reply else "Oi! Estou aqui e pronta para ajudar. O que vamos fazer agora?"

        self.memory.log_turn(user_message, reply_text)

        if enable_voice and reply_text:
            yield reply_text, final_image_path, None, "Sintetizando voz..."
            audio_path = self.voice_engine.synthesize(
                reply_text,
                selected_library_voice=library_voice,
                uploaded_wav=custom_voice_path,
                ref_text_hint=ref_text_hint,
                tts_backend=tts_backend,
                exaggeration=exaggeration
            )
            if generated_image:
                self._preload_llm_async()
            yield reply_text, final_image_path, audio_path, "Pronto"
        else:
            if generated_image:
                self._preload_llm_async()
            yield reply_text, final_image_path, None, "Pronto"
