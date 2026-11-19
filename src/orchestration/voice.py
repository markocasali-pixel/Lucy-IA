import asyncio
import gc
import re
import traceback
import uuid
from pathlib import Path
import soundfile as sf
import torch
import torchaudio
import edge_tts

OUTPUT_DIR = Path("Data/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VOICES_DIR = Path("Data/voices")
VOICES_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_REF_PHRASE = "Olá! Eu sou a Lucy, sua assistente pessoal. Estou pronta para ajudar você com tudo o que precisar hoje!"

def list_saved_voices() -> list[str]:
    """Lista todos os arquivos de voz salvos em Data/voices."""
    exts = {".wav", ".mp3", ".flac", ".ogg"}
    files = [
        f.name for f in sorted(VOICES_DIR.iterdir())
        if f.is_file() and f.suffix.lower() in exts
    ]
    return files if files else ["(Nenhuma voz na pasta Data/voices)"]

def _clean_text_for_speech(text: str) -> str:
    t = text.replace("*", "").replace("#", "").replace("`", "").strip()
    def _brl_repl(m):
        reais = m.group(1)
        centavos = m.group(2)
        if centavos and int(centavos) > 0:
            return f"{reais} reais e {int(centavos)} centavos"
        return f"{reais} reais"
    t = re.sub(r'R\$\s*(\d+)[.,](\d{2})', _brl_repl, t)
    t = re.sub(r'R\$\s*(\d+)', r'\1 reais', t)
    t = re.sub(r'[^\w\s,.!?;:áéíóúâêîôûãõàçÁÉÍÓÚÂÊÎÔÛÃÕÀÇ-]', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()

def _prepare_reference_audio(src_path: str) -> str:
    wav, sr = torchaudio.load(src_path)
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    if sr != 24000:
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=24000)(wav)

    abs_wav = wav.abs().squeeze(0)
    non_silent = torch.where(abs_wav > 0.015)[0]
    if len(non_silent) > 0:
        start_idx = max(0, int(non_silent[0]) - 1200)
        end_idx = min(wav.shape[1], int(non_silent[-1]) + 2400)
        wav = wav[:, start_idx:end_idx]

    peak = torch.max(torch.abs(wav))
    if peak > 0:
        wav = (wav / peak) * 0.85

    max_samples = 24000 * 10
    if wav.shape[1] > max_samples:
        wav = wav[:, :max_samples]

    clean_ref = OUTPUT_DIR / "clean_ref_voice.wav"
    sf.write(str(clean_ref), wav.squeeze(0).numpy(), 24000, subtype="PCM_16")
    return str(clean_ref)

class VoiceEngine:
    def __init__(self):
        self.chatterbox_model = None
        self.f5_model = None
        self.default_edge_voice = "pt-BR-FranciscaNeural"

    def unload(self):
        if self.chatterbox_model is not None:
            del self.chatterbox_model
            self.chatterbox_model = None
        if self.f5_model is not None:
            del self.f5_model
            self.f5_model = None
        gc.collect()
        torch.cuda.empty_cache()

    def _load_chatterbox(self):
        if self.chatterbox_model is None:
            print("Carregando motor Chatterbox Multilingual PT-BR...")
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
            self.chatterbox_model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")

    def _load_f5(self):
        if self.f5_model is None:
            print("Carregando motor F5-TTS PT-BR...")
            from f5_tts.api import F5TTS
            from huggingface_hub import hf_hub_download, list_repo_files
            repo_id = "firstpixel/F5-TTS-pt-br"
            files = list_repo_files(repo_id)
            ckpt_name = next((f for f in files if f.endswith(".safetensors") or f.endswith(".pt")), None)
            vocab_name = next((f for f in files if f.endswith("vocab.txt")), None)
            ckpt_path = hf_hub_download(repo_id=repo_id, filename=ckpt_name)
            vocab_path = hf_hub_download(repo_id=repo_id, filename=vocab_name) if vocab_name else ""
            self.f5_model = F5TTS(model="F5TTS_Base", ckpt_file=ckpt_path, vocab_file=vocab_path)

    async def _synthesize_edge(self, text: str, out_path: Path):
        communicate = edge_tts.Communicate(text, self.default_edge_voice, pitch="+10Hz")
        await communicate.save(str(out_path))

    def _resolve_reference(self, selected_library_voice: str | None, uploaded_wav: str | None) -> str | None:
        if uploaded_wav and Path(uploaded_wav).exists():
            return uploaded_wav
        if selected_library_voice and not selected_library_voice.startswith("("):
            cand = VOICES_DIR / selected_library_voice
            if cand.exists():
                return str(cand)
        saved = list_saved_voices()
        if saved and not saved[0].startswith("("):
            return str(VOICES_DIR / saved[0])
        return None

    def synthesize(
        self,
        text: str,
        selected_library_voice: str | None = None,
        uploaded_wav: str | None = None,
        ref_text_hint: str = DEFAULT_REF_PHRASE,
        tts_backend: str = "Chatterbox PT-BR (Expressivo)",
        exaggeration: float = 0.65
    ) -> str | None:
        clean_text = _clean_text_for_speech(text)
        if not clean_text:
            return None

        ref_audio = self._resolve_reference(selected_library_voice, uploaded_wav)

        if ref_audio and "Chatterbox" in tts_backend:
            try:
                if self.f5_model is not None:
                    del self.f5_model
                    self.f5_model = None
                    torch.cuda.empty_cache()

                self._load_chatterbox()
                prepared_ref = _prepare_reference_audio(ref_audio)
                out_path = OUTPUT_DIR / f"voice_{uuid.uuid4().hex[:8]}.wav"

                wav_out = self.chatterbox_model.generate(
                    clean_text,
                    language_id="pt",
                    audio_prompt_path=prepared_ref,
                    exaggeration=float(exaggeration),
                    cfg_weight=0.5
                )
                torchaudio.save(str(out_path), wav_out.cpu(), self.chatterbox_model.sr)
                torch.cuda.empty_cache()
                return str(out_path)
            except Exception as e:
                print(f"[AVISO CHATTERBOX]: {e}. Tentando F5-TTS...")
                traceback.print_exc()

        if ref_audio:
            try:
                if self.chatterbox_model is not None:
                    del self.chatterbox_model
                    self.chatterbox_model = None
                    torch.cuda.empty_cache()

                self._load_f5()
                prepared_ref = _prepare_reference_audio(ref_audio)
                out_path = OUTPUT_DIR / f"voice_{uuid.uuid4().hex[:8]}.wav"
                self.f5_model.infer(
                    ref_file=prepared_ref,
                    ref_text=(ref_text_hint or DEFAULT_REF_PHRASE).strip(),
                    gen_text=clean_text,
                    file_wave=str(out_path),
                    nfe_step=32,
                    cfg_strength=2.0,
                    speed=0.95
                )
                torch.cuda.empty_cache()
                return str(out_path)
            except Exception as e:
                print(f"[AVISO F5-TTS]: {e}. Usando voz neural padrão.")
                traceback.print_exc()

        try:
            out_path = OUTPUT_DIR / f"voice_{uuid.uuid4().hex[:8]}.mp3"
            asyncio.run(self._synthesize_edge(clean_text, out_path))
            return str(out_path)
        except Exception as e:
            print(f"[AVISO EDGE-TTS]: Falha na sintese ({e}).")
            return None
