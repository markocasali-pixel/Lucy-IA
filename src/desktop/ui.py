import base64
import json
import shutil
from pathlib import Path
import gradio as gr
from src.orchestration.engine import LucyEngine
from src.orchestration.vision_gen import CHECKPOINT_MODELS, FOOOCUS_STYLES, RESOLUTIONS
from src.orchestration.voice import list_saved_voices, VOICES_DIR, DEFAULT_REF_PHRASE

CSS_PATH = Path("src/desktop/style.css")

def _build_style_preview_map() -> str:
    preview_map = {}
    for name, info in FOOOCUS_STYLES.items():
        sample_path = info.get("sample")
        if sample_path and Path(sample_path).exists():
            try:
                raw = Path(sample_path).read_bytes()
                b64 = base64.b64encode(raw).decode("ascii")
                preview_map[name] = f"data:image/jpeg;base64,{b64}"
            except Exception:
                continue
    return json.dumps(preview_map)

def build_ui(engine: LucyEngine):
    custom_css = CSS_PATH.read_text(encoding="utf-8") + """
    #lucy-style-hover-card {
        position: fixed;
        z-index: 99999;
        pointer-events: none;
        background: rgba(15, 23, 42, 0.95);
        border: 1px solid #ec4899;
        border-radius: 12px;
        padding: 6px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.75);
        display: none;
        width: 170px;
        text-align: center;
    }
    #lucy-style-hover-card img {
        width: 156px;
        height: 156px;
        object-fit: cover;
        border-radius: 8px;
        display: block;
    }
    #lucy-style-hover-card span {
        display: block;
        font-size: 11px;
        color: #f8fafc;
        margin-top: 4px;
        font-weight: 600;
    }
    """

    preview_json = _build_style_preview_map()
    hover_js = f"""
    () => {{
        const samples = {preview_json};
        let card = document.getElementById('lucy-style-hover-card');
        if (!card) {{
            card = document.createElement('div');
            card.id = 'lucy-style-hover-card';
            card.innerHTML = '<img id="lucy-hover-img" src=""/><span id="lucy-hover-title"></span>';
            document.body.appendChild(card);
        }}
        const imgEl = document.getElementById('lucy-hover-img');
        const titleEl = document.getElementById('lucy-hover-title');

        document.addEventListener('mouseover', (e) => {{
            const opt = e.target.closest('li[role="option"], .item, label');
            if (!opt) {{
                card.style.display = 'none';
                return;
            }}
            const labelText = (opt.innerText || '').trim().split('\\n')[0].trim();
            if (samples[labelText]) {{
                imgEl.src = samples[labelText];
                titleEl.textContent = labelText;
                card.style.display = 'block';
            }} else {{
                card.style.display = 'none';
            }}
        }});

        document.addEventListener('mousemove', (e) => {{
            if (card.style.display === 'block') {{
                card.style.left = Math.max(12, e.clientX - 195) + 'px';
                card.style.top = Math.min(window.innerHeight - 200, Math.max(12, e.clientY - 80)) + 'px';
            }}
        }});
    }}
    """

    initial_voices = list_saved_voices()

    with gr.Blocks(title="Lucy IA") as app:
        with gr.Row(elem_classes="lucy-header"):
            gr.HTML('<h1 class="lucy-title">✦ LUCY IA</h1>')
            with gr.Row():
                status_badge = gr.Markdown("**Status:** Pronto")
                voice_toggle = gr.Checkbox(label="Voz Ativa", value=True, scale=1)
                clear_btn = gr.Button("Limpar Chat", size="sm", scale=1)

        with gr.Row():
            with gr.Column(scale=7):
                chatbot = gr.Chatbot(elem_classes="chatbot", height=580)
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Converse com a Lucy, peça para criar/ler arquivos ou gerar uma imagem...",
                        show_label=False,
                        scale=8,
                        container=False
                    )
                    send_btn = gr.Button("Enviar", variant="primary", scale=2)

            with gr.Column(scale=3):
                audio_player = gr.Audio(label="Voz da Lucy", autoplay=True, interactive=False)
                with gr.Accordion("🎙️ Biblioteca de Vozes (Chatterbox / F5-TTS)", open=True):
                    with gr.Row():
                        voice_library_dropdown = gr.Dropdown(
                            choices=initial_voices,
                            value=initial_voices[0] if initial_voices else None,
                            label="Vozes Salvas (Data/voices)",
                            scale=4
                        )
                        refresh_voices_btn = gr.Button("🔄", size="sm", scale=1)
                    tts_backend_dropdown = gr.Dropdown(
                        choices=["Chatterbox PT-BR (Expressivo)", "F5-TTS PT-BR (Rápido)"],
                        value="Chatterbox PT-BR (Expressivo)",
                        label="Motor de Voz"
                    )
                    exaggeration_slider = gr.Slider(
                        minimum=0.2,
                        maximum=1.0,
                        value=0.65,
                        step=0.05,
                        label="Expressividade / Emoção Anime (Chatterbox)"
                    )
                    voice_clone_input = gr.Audio(
                        label="Adicionar/Testar Novo Áudio (salva na biblioteca)",
                        type="filepath"
                    )
                    ref_text_input = gr.Textbox(
                        value=DEFAULT_REF_PHRASE,
                        label="Frase de Referência (para F5-TTS)",
                        lines=2
                    )
                with gr.Accordion("🎨 Configurações de Imagem (Fooocus)", open=False):
                    model_dropdown = gr.Dropdown(
                        choices=list(CHECKPOINT_MODELS.keys()),
                        value="Anime (AnimaPencil XL)",
                        label="Modelo Base (Checkpoint)"
                    )
                    resolution_dropdown = gr.Dropdown(
                        choices=list(RESOLUTIONS.keys()),
                        value="1920x1080 (Full HD 16:9 - Paisagem)",
                        label="Tamanho / Resolução"
                    )
                    styles_dropdown = gr.Dropdown(
                        choices=list(FOOOCUS_STYLES.keys()),
                        value=["Fooocus Enhance", "Fooocus Sharp", "sai-anime"],
                        multiselect=True,
                        label="Styles do Fooocus (Passe o mouse para ver)"
                    )
                image_output = gr.Image(label="Visualizador / Imagem Gerada", height=250)

        def save_uploaded_voice(uploaded_path):
            if not uploaded_path or not Path(uploaded_path).exists():
                voices = list_saved_voices()
                return gr.update(choices=voices)
            src = Path(uploaded_path)
            dest = VOICES_DIR / src.name
            if src.resolve() != dest.resolve():
                shutil.copy2(src, dest)
            voices = list_saved_voices()
            return gr.update(choices=voices, value=dest.name)

        def refresh_voices():
            voices = list_saved_voices()
            return gr.update(choices=voices, value=voices[0] if voices else None)

        voice_clone_input.change(save_uploaded_voice, inputs=[voice_clone_input], outputs=[voice_library_dropdown])
        refresh_voices_btn.click(refresh_voices, outputs=[voice_library_dropdown])

        def handle_submit(
            user_msg, history, use_voice, lib_voice, custom_voice, ref_txt,
            tts_backend, exag, selected_model, selected_styles, selected_res
        ):
            if not user_msg.strip():
                yield "", history, None, None, "**Status:** Pronto"
                return

            history = history or []
            temp_history = history + [{"role": "user", "content": user_msg}]
            yield "", temp_history, None, None, "**Status:** Processando..."

            for reply_text, img_frame, audio_path, status_msg in engine.process_turn_stream(
                user_msg, history, use_voice, lib_voice, custom_voice, ref_txt,
                tts_backend, exag, selected_model, selected_styles, selected_res
            ):
                if reply_text is not None:
                    final_history = temp_history + [{"role": "assistant", "content": reply_text}]
                    yield "", final_history, audio_path, img_frame, f"**Status:** {status_msg}"
                else:
                    yield "", temp_history, None, img_frame, f"**Status:** {status_msg}"

        send_btn.click(
            handle_submit,
            inputs=[
                msg_input, chatbot, voice_toggle, voice_library_dropdown, voice_clone_input,
                ref_text_input, tts_backend_dropdown, exaggeration_slider,
                model_dropdown, styles_dropdown, resolution_dropdown
            ],
            outputs=[msg_input, chatbot, audio_player, image_output, status_badge]
        )
        msg_input.submit(
            handle_submit,
            inputs=[
                msg_input, chatbot, voice_toggle, voice_library_dropdown, voice_clone_input,
                ref_text_input, tts_backend_dropdown, exaggeration_slider,
                model_dropdown, styles_dropdown, resolution_dropdown
            ],
            outputs=[msg_input, chatbot, audio_player, image_output, status_badge]
        )
        clear_btn.click(lambda: ([], None, None, "**Status:** Pronto"), outputs=[chatbot, audio_player, image_output, status_badge])
        app.load(None, None, None, js=hover_js)

    return app, custom_css
