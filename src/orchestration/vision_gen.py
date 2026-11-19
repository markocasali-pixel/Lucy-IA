import gc
import json
import queue
import random
import threading
import uuid
from pathlib import Path
from PIL import Image, ImageEnhance
import torch
from compel import CompelForSDXL
from diffusers import (
    StableDiffusionXLPipeline,
    DPMSolverMultistepScheduler,
    AutoencoderKL,
    AutoencoderTiny
)

OUTPUT_DIR = Path("Data/outputs").resolve()
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

RESOLUTIONS = {
    "1920x1080 (Full HD 16:9 - Paisagem)": (1024, 576, 1920, 1080),
    "1080x1920 (Full HD 9:16 - Retrato)": (576, 1024, 1080, 1920),
    "1024x1024 (Quadrado 1:1)": (1024, 1024, 1024, 1024),
    "832x1024 (Retrato Clássico)": (832, 1024, 832, 1024),
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
    prompt = (
        f"{base_prompt.strip()}, masterpiece, best quality, highly detailed face, "
        "symmetrical eyes, sharp facial features, clean lineart, rich lighting"
    )
    neg_parts = []

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

    seen = set()
    clean_tokens = []
    for part in prompt.split(","):
        p = part.strip()
        if p and p.lower() not in seen:
            seen.add(p.lower())
            clean_tokens.append(p)
    final_prompt = ", ".join(clean_tokens[:28])

    neg_parts.append(
        "bad face, distorted face, ugly face, deformed face, asymmetric eyes, cross-eyed, "
        "bad anatomy, bad hands, extra fingers, missing fingers, extra limbs, missing limbs, "
        "text, watermark, signature, error, worst quality, low quality, blurry, deformed, "
        "camera viewfinder, tv screen, monitor, phone screen, framed by screen, hands holding camera"
    )
    neg_prompt = ", ".join(dict.fromkeys(p.strip() for p in neg_parts if p.strip()))
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
        self.pipe.vae.enable_slicing()
        self.pipe.enable_attention_slicing()
        self.pipe.enable_model_cpu_offload()

        if self.taesd is None:
            self.taesd = AutoencoderTiny.from_pretrained(
                "madebyollin/taesdxl", torch_dtype=torch.float16
            ).to("cuda")

    def generate_stream(
        self,
        prompt: str,
        model_label: str = "Anime (AnimaPencil XL)",
        selected_styles: list[str] | None = None,
        resolution_label: str = "1920x1080 (Full HD 16:9 - Paisagem)",
        seed: int = -1
    ):
        try:
            self._load_pipeline(model_label)
        except Exception as e:
            yield None, None, f"Erro ao carregar modelo '{model_label}': {e}"
            return

        gen_w, gen_h, target_w, target_h = RESOLUTIONS.get(
            resolution_label, (1024, 576, 1920, 1080)
        )

        used_seed = int(seed) if (seed is not None and int(seed) >= 0) else random.randint(1, 2147483647)
        generator = torch.Generator(device="cpu").manual_seed(used_seed)

        full_prompt, neg_prompt = apply_fooocus_styles(prompt, selected_styles)
        if "Pony" in model_label:
            full_prompt = f"score_9, score_8_up, score_7_up, source_anime, {full_prompt}"

        compel = CompelForSDXL(self.pipe, device="cuda")
        with torch.no_grad():
            prompt_embeds = compel(full_prompt, negative_prompt=neg_prompt)
            conditioning = prompt_embeds.embeds.to("cuda", dtype=torch.float16)
            pooled = prompt_embeds.pooled_embeds.to("cuda", dtype=torch.float16)
            neg_conditioning = prompt_embeds.negative_embeds.to("cuda", dtype=torch.float16)
            neg_pooled = prompt_embeds.negative_pooled_embeds.to("cuda", dtype=torch.float16)

        del compel, prompt_embeds
        gc.collect()
        torch.cuda.empty_cache()

        preview_queue = queue.Queue()
        final_holder = {"path": None, "done": False, "error": None}
        total_steps = 30

        def step_callback(pipe, step_index, timestep, callback_kwargs):
            if step_index % 3 == 0:
                latents = callback_kwargs["latents"]
                with torch.no_grad():
                    decoded = self.taesd.decode(latents).sample
                    decoded = (decoded / 2 + 0.5).clamp(0, 1)
                    img_np = (decoded[0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
                    preview_queue.put((step_index + 1, img_np))
            return callback_kwargs

        def run_inference():
            try:
                image = self.pipe(
                    prompt_embeds=conditioning,
                    pooled_prompt_embeds=pooled,
                    negative_prompt_embeds=neg_conditioning,
                    negative_pooled_prompt_embeds=neg_pooled,
                    width=gen_w,
                    height=gen_h,
                    num_inference_steps=total_steps,
                    guidance_scale=7.0,
                    generator=generator,
                    callback_on_step_end=step_callback
                ).images[0]

                if (image.width, image.height) != (target_w, target_h):
                    image = image.resize((target_w, target_h), Image.Resampling.LANCZOS)
                    image = ImageEnhance.Sharpness(image).enhance(1.15)
                    image = ImageEnhance.Contrast(image).enhance(1.04)

                file_path = OUTPUT_DIR / f"img_{used_seed}_{uuid.uuid4().hex[:6]}.png"
                image.save(file_path, format="PNG")
                final_holder["path"] = str(file_path)
            except Exception as e:
                final_holder["error"] = str(e)
            finally:
                final_holder["done"] = True

        worker = threading.Thread(target=run_inference)
        worker.start()

        while not final_holder["done"] or not preview_queue.empty():
            try:
                step_num, frame = preview_queue.get(timeout=0.15)
                yield frame, None, f"Gerando em alta qualidade ({target_w}x{target_h}, Seed {used_seed})... Passo {step_num}/{total_steps}"
            except queue.Empty:
                continue

        worker.join()

        del self.pipe, conditioning, pooled, neg_conditioning, neg_pooled
        self.pipe = None
        gc.collect()
        torch.cuda.empty_cache()

        if final_holder["error"]:
            yield None, None, f"Erro na imagem: {final_holder['error']}"
        elif final_holder["path"] and Path(final_holder["path"]).exists():
            yield final_holder["path"], final_holder["path"], f"Concluído em alta qualidade ({target_w}x{target_h} | Seed: {used_seed})"
        else:
            yield None, None, "Falha ao salvar a imagem final."
