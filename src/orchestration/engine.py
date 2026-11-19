import gc
import json
import random
import re
import torch
from llama_cpp import Llama
from src.orchestration import tools
from src.orchestration.memory import MemoryManager
from src.orchestration.vision_gen import ImageGenerator
from src.orchestration.voice import VoiceEngine, DEFAULT_REF_PHRASE

IMAGE_DONE_REPLIES = [
    "Prontinho! Acabei de finalizar sua arte em alta definição. O que você achou do resultado?",
    "Tcharam! Sua imagem ficou pronta e já está na tela. Se quiser mudar algum detalhe ou testar outra seed, é só me falar!",
    "Missão cumprida! Renderizei a imagem exatamente como você pediu. Ficou incrível, né?"
]

class LucyEngine:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.llm = None
        self._load_llm()
        self.memory = MemoryManager()
        self.image_gen = ImageGenerator()
        self.voice_engine = VoiceEngine()
        self.base_system_prompt = (
            "Você é Lucy IA, uma assistente pessoal inteligente, espirituosa, calorosa e carismática "
            "(inspirada na inteligência do J.A.R.V.I.S. com o carisma e expressividade de uma protagonista de anime).\n"
            "Fale de forma natural, viva, acolhedora e objetiva (2 a 3 frases bem expressivas, sem enrolação longa) em Português do Brasil.\n\n"
            "REGRA CRÍTICA DE AÇÕES E FERRAMENTAS:\n"
            "Você NUNCA deve dizer que abriu um programa, criou um arquivo, pesquisou ou leu a tela sem antes emitir o JSON da ferramenta!\n"
            "Sempre que o usuário pedir uma ação abaixo, responda EXCLUSIVAMENTE com um JSON válido no formato:\n"
            '{"tool": "nome_da_ferramenta", "args": {...}}\n\n'
            "FERRAMENTAS REAIS DISPONÍVEIS:\n"
            '1. "open_app" -> args: {"app_name": "calculadora"} (abre calculadora, bloco de notas, paint, explorer, chrome, edge, steam, discord, spotify, youtube ou qualquer app/URL)\n'
            '2. "write_or_edit_file" -> args: {"filename": "nota.txt", "content": "texto", "mode": "w", "location": "desktop"} (use location="desktop" se o usuário mencionar área de trabalho/desktop, ou "workspace" caso contrário)\n'
            '3. "read_file" -> args: {"filename": "arquivo.txt", "location": "desktop"} (lê txt, pdf, docx, py)\n'
            '4. "open_file_in_os" -> args: {"filename": "arquivo.txt", "location": "desktop"} (abre o arquivo na tela do Windows)\n'
            '5. "list_files" -> args: {"location": "desktop"} (lista arquivos do desktop ou workspace)\n'
            '6. "read_screen_or_image" -> args: {} (captura a tela atual do computador/jogo e extrai todo o texto para você traduzir ou explicar)\n'
            '7. "system_control" -> args: {"action": "status"} (verifica CPU/RAM/RTX 3070 Ti, ou "aumentar volume", "diminuir volume", "mutar", "pausar", "proxima")\n'
            '8. "web_search" -> args: {"query": "termo"} (pesquisa na internet em tempo real)\n'
            '9. "remember_fact" -> args: {"fact": "fato importante sobre o usuário"} (salva na memória de longo prazo)\n'
            '10. "generate_image" -> args: {"prompt": "descrição detalhada em inglês"} (gera arte/imagem)\n\n'
            "Se for apenas conversa normal, responda diretamente com simpatia e personalidade!"
        )

    def _build_system_prompt(self) -> str:
        mem_ctx = self.memory.get_context_block()
        if mem_ctx:
            return f"{self.base_system_prompt}\n\n=== MEMÓRIA DE LONGO PRAZO ===\n{mem_ctx}"
        return self.base_system_prompt

    def _load_llm(self):
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

    def _unload_all_models(self):
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

    def _fallback_intent_router(self, user_msg: str):
        low = user_msg.lower()
        if any(k in low for k in ("traduza minha tela", "traduzir a tela", "o que esta na tela", "o que está na tela", "leia minha tela", "texto do jogo", "traduzir tela")):
            return {"tool": "read_screen_or_image", "args": {}}
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
        for item in history:
            if isinstance(item, dict) and item.get("content"):
                clean_hist = self._normalize_content(item["content"])
                if clean_hist:
                    messages.append({"role": item["role"], "content": clean_hist})

        effective_prompt = user_message
        if uploaded_image_path:
            effective_prompt = f"{user_message}\n[O usuário anexou uma imagem em: {uploaded_image_path}]"
        messages.append({"role": "user", "content": effective_prompt})

        if uploaded_image_path:
            tool_data = {"tool": "read_screen_or_image", "args": {"image_path": uploaded_image_path}}
            raw_reply = json.dumps(tool_data, ensure_ascii=False)
        else:
            response = self.llm.create_chat_completion(
                messages=messages,
                temperature=0.75,
                max_tokens=512
            )
            raw_reply = (response["choices"][0]["message"].get("content") or "").strip()
            tool_data = self._extract_tool_call(raw_reply) or self._fallback_intent_router(user_message)

        final_image_path = None
        reply_text = ""

        if tool_data:
            fn_name = tool_data["tool"]
            args = tool_data.get("args", {})

            if fn_name == "generate_image":
                self._unload_all_models()
                for preview_frame, done_path, status_msg in self.image_gen.generate_stream(
                    args.get("prompt", user_message),
                    model_label=image_model,
                    selected_styles=image_styles,
                    resolution_label=image_resolution,
                    seed=image_seed
                ):
                    final_image_path = done_path
                    yield None, preview_frame, None, status_msg
                # Resposta imediata sem esperar recarregar o LLM para confirmar a imagem
                reply_text = random.choice(IMAGE_DONE_REPLIES)
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

                messages.append({"role": "assistant", "content": raw_reply})
                messages.append({
                    "role": "user",
                    "content": (
                        f"[RESULTADO REAL DO SISTEMA]: {tool_result}\n"
                        "Responda agora em 2 ou 3 frases naturais, simpáticas e expressivas (sem mostrar JSON). "
                        "Se houver texto extraído da tela/jogo em outro idioma, traduza-o claramente para o Português do Brasil."
                    )
                })
                final_res = self.llm.create_chat_completion(messages=messages, temperature=0.8, max_tokens=320)
                reply_text = (final_res["choices"][0]["message"].get("content") or tool_result).strip()
        else:
            reply_text = raw_reply if raw_reply else "Oi! Estou aqui e pronta para o que der e vier. O que vamos aprontar agora?"

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
            yield reply_text, final_image_path, audio_path, "Pronto"
        else:
            yield reply_text, final_image_path, None, "Pronto"
