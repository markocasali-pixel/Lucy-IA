import base64
import json
from pathlib import Path
import gradio as gr
from src.orchestration.engine import LucyEngine
from src.orchestration.vision_gen import CHECKPOINT_MODELS, FOOOCUS_STYLES

CSS_PATH = Path("src/desktop/style.css")

def _build_style_preview_map() -> str:
    """Gera um mapa JSON com miniaturas base64 leves para hover instantâneo nos estilos."""
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

    with gr.Blocks(title="Lucy IA") as app:
        with gr.Row(elem_classes="lucy-header"):
            gr.HTML('<h1 class="lucy-title">✦ LUCY IA</h1>')
            with gr.Row():
                status_badge = gr.Markdown("**Status:** Pronto")
                voice_toggle = gr.Checkbox(label="Voz Ativa", value=True, scale=1)
                clear_btn = gr.Button("Limpar Chat", size="sm", scale=1)

        with gr.Row():
            with gr.Column(scale=7):
                chatbot = gr.Chatbot(elem_classes="chatbot", height=550)
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
                voice_clone_input = gr.Audio(
                    label="Clonar Voz / Referência Anime (5-10s)",
                    type="filepath"
                )
                with gr.Accordion("🎨 Configurações de Imagem (Fooocus)", open=True):
                    model_dropdown = gr.Dropdown(
                        choices=list(CHECKPOINT_MODELS.keys()),
                        value="Anime (AnimaPencil XL)",
                        label="Modelo Base (Checkpoint)"
                    )
                    styles_dropdown = gr.Dropdown(
                        choices=list(FOOOCUS_STYLES.keys()),
                        value=["Fooocus Enhance", "Fooocus Sharp", "sai-anime"],
                        multiselect=True,
                        label="Styles do Fooocus (Passe o mouse para ver o estilo)"
                    )
                image_output = gr.Image(label="Visualizador / Imagem Gerada", height=290)

        def handle_submit(user_msg, history, use_voice, custom_voice, selected_model, selected_styles):
            if not user_msg.strip():
                yield "", history, None, None, "**Status:** Pronto"
                return

            history = history or []
            temp_history = history + [{"role": "user", "content": user_msg}]
            yield "", temp_history, None, None, "**Status:** Processando..."

            for reply_text, img_frame, audio_path, status_msg in engine.process_turn_stream(
                user_msg, history, use_voice, custom_voice, selected_model, selected_styles
            ):
                if reply_text is not None:
                    final_history = temp_history + [{"role": "assistant", "content": reply_text}]
                    yield "", final_history, audio_path, img_frame, f"**Status:** {status_msg}"
                else:
                    yield "", temp_history, None, img_frame, f"**Status:** {status_msg}"

        send_btn.click(
            handle_submit,
            inputs=[msg_input, chatbot, voice_toggle, voice_clone_input, model_dropdown, styles_dropdown],
            outputs=[msg_input, chatbot, audio_player, image_output, status_badge]
        )
        msg_input.submit(
            handle_submit,
            inputs=[msg_input, chatbot, voice_toggle, voice_clone_input, model_dropdown, styles_dropdown],
            outputs=[msg_input, chatbot, audio_player, image_output, status_badge]
        )
        clear_btn.click(lambda: ([], None, None, "**Status:** Pronto"), outputs=[chatbot, audio_player, image_output, status_badge])
        app.load(None, None, None, js=hover_js)

    return app, custom_css

