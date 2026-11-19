import gc
import json
import re
import torch
from llama_cpp import Llama
from src.orchestration import tools
from src.orchestration.vision_gen import ImageGenerator
from src.orchestration.voice import VoiceEngine, DEFAULT_REF_PHRASE

class LucyEngine:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.llm = None
        self._load_llm()
        self.image_gen = ImageGenerator()
        self.voice_engine = VoiceEngine()
        self.system_prompt = (
            "Você é Lucy IA, uma assistente pessoal avançada estilo J.A.R.V.I.S. com personalidade cativante, "
            "inteligente e expressiva, inspirada em personagens de anime.\n"
            "Responda SEMPRE em Português do Brasil de forma natural e direta.\n\n"
            "FERRAMENTAS DISPONÍVEIS:\n"
            "Se o usuário pedir para realizar uma ação abaixo ou perguntar algo atual (como cotação do dólar, notícias, clima), "
            "responda APENAS com um bloco JSON exatamente no formato:\n"
            '{"tool": "nome_da_ferramenta", "args": {...}}\n\n'
            "Lista de ferramentas:\n"
            '1. "web_search" -> args: {"query": "termo de pesquisa"} (pesquisa na internet em tempo real, cotação do dólar, notícias)\n'
            '2. "list_files" -> args: {} (lista arquivos da pasta workspace)\n'
            '3. "read_file" -> args: {"filename": "nome.ext"} (lê txt, pdf, docx, py)\n'
            '4. "write_or_edit_file" -> args: {"filename": "nome.ext", "content": "texto", "mode": "w"} (cria ou edita arquivo)\n'
            '5. "open_file_in_os" -> args: {"filename": "nome.ext"} (abre arquivo no Windows)\n'
            '6. "generate_image" -> args: {"prompt": "descrição detalhada em inglês"} (gera uma imagem/arte)\n\n'
            "Se for apenas uma conversa normal que não precisa de internet nem arquivos, NÃO use JSON; responda normalmente em texto!"
        )

    def _load_llm(self):
        if self.llm is None:
            self.llm = Llama(
                model_path=self.model_path,
                n_gpu_layers=-1,
                n_ctx=4096,
                chat_format="chatml",
                verbose=False
            )

    def _unload_all_models(self):
        if self.llm is not None:
            del self.llm
            self.llm = None
        self.voice_engine.unload()
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
        known_tools = {"web_search", "list_files", "read_file", "write_or_edit_file", "open_file_in_os", "generate_image"}
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
        image_resolution: str = "1920x1080 (Full HD 16:9 - Paisagem)"
    ):
        self._load_llm()
        messages = [{"role": "system", "content": self.system_prompt}]
        for item in history:
            if isinstance(item, dict) and item.get("content"):
                clean_hist = self._normalize_content(item["content"])
                if clean_hist:
                    messages.append({"role": item["role"], "content": clean_hist})
        messages.append({"role": "user", "content": user_message})

        response = self.llm.create_chat_completion(
            messages=messages,
            temperature=0.7,
            max_tokens=1024
        )
        raw_reply = (response["choices"][0]["message"].get("content") or "").strip()
        final_image_path = None

        tool_data = self._extract_tool_call(raw_reply)
        if tool_data:
            fn_name = tool_data["tool"]
            args = tool_data.get("args", {})

            if fn_name == "web_search":
                tool_result = tools.web_search(args.get("query", user_message))
            elif fn_name == "list_files":
                tool_result = tools.list_files()
            elif fn_name == "write_or_edit_file":
                tool_result = tools.write_or_edit_file(**args)
            elif fn_name == "read_file":
                tool_result = tools.read_file(**args)
            elif fn_name == "open_file_in_os":
                tool_result = tools.open_file_in_os(**args)
            elif fn_name == "generate_image":
                self._unload_all_models()
                for preview_frame, done_path, status_msg in self.image_gen.generate_stream(
                    args.get("prompt", user_message),
                    model_label=image_model,
                    selected_styles=image_styles,
                    resolution_label=image_resolution
                ):
                    final_image_path = done_path
                    yield None, preview_frame, None, status_msg
                self._load_llm()
                tool_result = "Imagem gerada e exibida na tela com sucesso."
            else:
                tool_result = "Ação concluída."

            messages.append({"role": "assistant", "content": raw_reply})
            messages.append({
                "role": "user",
                "content": f"[SISTEMA]: {tool_result} Agora responda ao usuário de forma simpática e natural usando esses dados, sem mostrar JSON."
            })
            final_res = self.llm.create_chat_completion(messages=messages, temperature=0.7, max_tokens=512)
            reply_text = (final_res["choices"][0]["message"].get("content") or tool_result).strip()
        else:
            reply_text = raw_reply if raw_reply else "Oi! Estou aqui e pronta para ajudar. O que vamos fazer hoje?"

        audio_path = None
        if enable_voice and reply_text:
            audio_path = self.voice_engine.synthesize(
                reply_text,
                selected_library_voice=library_voice,
                uploaded_wav=custom_voice_path,
                ref_text_hint=ref_text_hint,
                tts_backend=tts_backend,
                exaggeration=exaggeration
            )

        yield reply_text, final_image_path, audio_path, "Pronto"
