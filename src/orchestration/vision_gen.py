import gc
import json
import queue
import threading
import uuid
from pathlib import Path
import torch
from diffusers import (
    StableDiffusionXLPipeline,
    DPMSolverMultistepScheduler,
    AutoencoderKL,
    AutoencoderTiny
)

OUTPUT_DIR = Path("Data/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FOOOCUS_BASE = Path(r"D:\Fooocus_win64_2-5-0\Fooocus")
FOOOCUS_CKPT_DIR = FOOOCUS_BASE / "models" / "checkpoints"
FOOOCUS_STYLES_DIR = FOOOCUS_BASE / "sdxl_styles"
FOOOCUS_SAMPLES_DIR = FOOOCUS_STYLES_DIR / "samples"

CHECKPOINT_MODELS = {
    "Anime (AnimaPencil XL)": "animaPencilXL_v500.safetensors",
    "Anime Expressivo (Pony V6 XL)": "ponyDiffusionV6XL.safetensors",
    "Realista (Juggernaut XL)": "juggernautXL_v8Rundiffusion.safetensors",
    "Artístico (Playground v2.5)": "playground-v2.5-1024px-aesthetic.fp16.safetensors",
}

def _style_to_sample_path(style_name: str) -> str | None:
    if not FOOOCUS_SAMPLES_DIR.exists():
        return None
    slug = style_name.lower().replace(" ", "_").replace("-", "_")
    for ext in (".jpg", ".png", ".webp"):
        cand = FOOOCUS_SAMPLES_DIR / f"{slug}{ext}"
        if cand.exists():
            return str(cand)
    return None

def load_fooocus_styles() -> dict[str, dict]:
    styles = {}
    if FOOOCUS_STYLES_DIR.exists():
        for json_file in sorted(FOOOCUS_STYLES_DIR.glob("*.json")):
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    for item in data:
                        name = item.get("name")
                        if name and name != "Fooocus V2":
                            styles[name] = {
                                "prompt": item.get("prompt", "{prompt}"),
                                "negative_prompt": item.get("negative_prompt", ""),
                                "sample": _style_to_sample_path(name)
                            }
            except Exception:
                continue
    return styles

FOOOCUS_STYLES = load_fooocus_styles()

def apply_fooocus_styles(base_prompt: str, selected_styles: list[str] | None) -> tuple[str, str]:
    """Aplica os estilos escolhidos priorizando os últimos marcados para não estourar os 77 tokens do CLIP."""
    prompt = base_prompt.strip()
    neg_parts = []

    # Inverte a lista para que o estilo mais recente selecionado tenha prioridade máxima nos tokens
    for style_name in reversed(selected_styles or []):
        style = FOOOCUS_STYLES.get(style_name)
        if not style:
            continue
        tmpl = style.get("prompt", "{prompt}")
        if "{prompt}" in tmpl:
            prompt = tmpl.replace("{prompt}", prompt)
        else:
            prompt = f"{tmpl}, {prompt}"
        if style.get("negative_prompt"):
            neg_parts.append(style["negative_prompt"])

    # Remove palavras duplicadas mantendo a ordem para economizar tokens do CLIP
    seen = set()
    clean_tokens = []
    for part in prompt.split(","):
        p = part.strip()
        if p and p.lower() not in seen:
            seen.add(p.lower())
            clean_tokens.append(p)
    final_prompt = ", ".join(clean_tokens)

    neg_prompt = ", ".join(dict.fromkeys(p.strip() for p in neg_parts if p.strip()))
    if not neg_prompt:
        neg_prompt = "lowres, bad anatomy, bad hands, text, error, worst quality, low quality, blurry, deformed"
    return final_prompt, neg_prompt

class ImageGenerator:
    def __init__(self):
        self.pipe = None
        self.taesd = None

    def _load_pipeline(self, model_label: str):
        filename = CHECKPOINT_MODELS.get(model_label, "animaPencilXL_v500.safetensors")
        ckpt_path = FOOOCUS_CKPT_DIR / filename

        gc.collect()
        torch.cuda.empty_cache()

        print(f"Carregando checkpoint do Fooocus: {filename}...")
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        self.pipe = StableDiffusionXLPipeline.from_single_file(
            str(ckpt_path),
            vae=vae,
            torch_dtype=torch.float16,
            use_safetensors=True
        )
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            use_karras_sigmas=True
        )
        # Mantém pico abaixo de 5.3GB VRAM na RTX 3070 Ti (evita estourar para RAM compartilhada lenta)
        self.pipe.vae.enable_slicing()
        self.pipe.enable_attention_slicing()
        self.pipe.enable_model_cpu_offload()

        if self.taesd is None:
            self.taesd = AutoencoderTiny.from_pretrained(
                "madebyollin/taesdxl", torch_dtype=torch.float16
            ).to("cuda")

    def generate_stream(self, prompt: str, model_label: str = "Anime (AnimaPencil XL)", selected_styles: list[str] | None = None):
        self._load_pipeline(model_label)

        full_prompt, neg_prompt = apply_fooocus_styles(prompt, selected_styles)
        if "Pony" in model_label:
            full_prompt = f"score_9, score_8_up, score_7_up, source_anime, {full_prompt}"

        print(f"[PROMPT FINAL APLICADO]: {full_prompt[:220]}...")

        preview_queue = queue.Queue()
        final_holder = {"path": None, "done": False}

        def step_callback(pipe, step_index, timestep, callback_kwargs):
            if step_index % 2 == 0:
                latents = callback_kwargs["latents"]
                with torch.no_grad():
                    decoded = self.taesd.decode(latents).sample
                    decoded = (decoded / 2 + 0.5).clamp(0, 1)
                    img_np = (decoded[0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
                    preview_queue.put((step_index + 1, img_np))
            return callback_kwargs

        def run_inference():
            image = self.pipe(
                prompt=full_prompt,
                negative_prompt=neg_prompt,
                width=768,
                height=1024,
                num_inference_steps=16,
                guidance_scale=6.5,
                callback_on_step_end=step_callback
            ).images[0]

            file_path = OUTPUT_DIR / f"img_{uuid.uuid4().hex[:8]}.png"
            image.save(file_path)
            final_holder["path"] = str(file_path)
            final_holder["done"] = True

        worker = threading.Thread(target=run_inference)
        worker.start()

        while not final_holder["done"] or not preview_queue.empty():
            try:
                step_num, frame = preview_queue.get(timeout=0.15)
                yield frame, None, f"Gerando imagem ao vivo... Passo {step_num}/16"
            except queue.Empty:
                continue

        worker.join()

        del self.pipe
        self.pipe = None
        gc.collect()
        torch.cuda.empty_cache()

        yield final_holder["path"], final_holder["path"], "Concluído!"

