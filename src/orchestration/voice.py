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
DEFAULT_VOICE_WAV = Path("Data/voices/lucy_anime_default.wav")

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
    non_silent = torch.where(abs_wav > 0.02)[0]
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

    clean_ref = OUTPUT_DIR / "clean_ref_f5.wav"
    sf.write(str(clean_ref), wav.squeeze(0).numpy(), 24000, subtype="PCM_16")
    return str(clean_ref)

class VoiceEngine:
    def __init__(self):
        self.f5_model = None
        self.default_edge_voice = "pt-BR-FranciscaNeural"

    def unload(self):
        if self.f5_model is not None:
            del self.f5_model
            self.f5_model = None
            gc.collect()
            torch.cuda.empty_cache()

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
            self.f5_model = F5TTS(ckpt_file=ckpt_path, vocab_file=vocab_path)

    async def _synthesize_edge(self, text: str, out_path: Path):
        communicate = edge_tts.Communicate(text, self.default_edge_voice)
        await communicate.save(str(out_path))

    def synthesize(self, text: str, speaker_wav: str | None = None) -> str | None:
        clean_text = _clean_text_for_speech(text)
        if not clean_text:
            return None

        ref_audio = speaker_wav if (speaker_wav and Path(speaker_wav).exists()) else (
            str(DEFAULT_VOICE_WAV) if DEFAULT_VOICE_WAV.exists() else None
        )

        if ref_audio:
            try:
                self._load_f5()
                prepared_ref = _prepare_reference_audio(ref_audio)
                out_path = OUTPUT_DIR / f"voice_{uuid.uuid4().hex[:8]}.wav"
                self.f5_model.infer(
                    ref_file=prepared_ref,
                    ref_text="",
                    gen_text=clean_text,
                    file_wave=str(out_path),
                    speed=1.0
                )
                torch.cuda.empty_cache()
                return str(out_path)
            except Exception as e:
                print(f"[AVISO F5-TTS]: {e}. Tentando voz neural...")
                traceback.print_exc()

        try:
            out_path = OUTPUT_DIR / f"voice_{uuid.uuid4().hex[:8]}.mp3"
            asyncio.run(self._synthesize_edge(clean_text, out_path))
            return str(out_path)
        except Exception as e:
            print(f"[AVISO EDGE-TTS]: Falha na sintese ({e}).")
            return None
