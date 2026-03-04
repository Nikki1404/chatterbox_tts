import os
import time
import asyncio
import base64
import re
import logging
from typing import Optional, List

import numpy as np
import torch
from fastapi import FastAPI, WebSocket
from chatterbox.tts import ChatterboxTTS


# =========================
# PERFORMANCE SETTINGS
# =========================

logging.basicConfig(level=logging.INFO)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("CUDA available:", torch.cuda.is_available())

torch.set_num_threads(1)

if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

# =========================
# LATENCY CONFIG
# =========================

CHAR_SHORT = 200
VOICE_LOCK_WORDS = 12
MODEL_WARMUP_TEXT = "This is a warmup run to initialize CUDA kernels."

app = FastAPI()

# =========================
# MODEL LOAD
# =========================

print(f"Loading Chatterbox on device: {DEVICE}")
model = ChatterboxTTS.from_pretrained(device=DEVICE)
SR = model.sr

# Warmup
with torch.inference_mode():
    _ = model.generate(MODEL_WARMUP_TEXT)

print("Chatterbox ready. Sample rate:", SR)


# =========================
# UTILS
# =========================

def wav_to_b64_raw(wav: np.ndarray) -> str:
    return base64.b64encode(wav.tobytes()).decode("utf-8")


def to_np(wav) -> np.ndarray:
    if isinstance(wav, torch.Tensor):
        wav = wav.detach().cpu().numpy()
    return np.asarray(wav).squeeze().astype(np.float32)


def split_sentences(text: str) -> List[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def split_first_words(text: str, n: int):
    words = text.split()
    if len(words) <= n:
        return text, ""
    return " ".join(words[:n]), " ".join(words[n:])


async def generate_async(text: str, ref_audio: Optional[str], clone_voice: bool):
    t0 = time.perf_counter()

    with torch.inference_mode():
        if DEVICE == "cuda":
            # autocast safely (without converting model)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                wav = await asyncio.to_thread(
                    model.generate,
                    text,
                    audio_prompt_path=ref_audio,
                )
        else:
            wav = await asyncio.to_thread(
                model.generate,
                text,
                audio_prompt_path=ref_audio,
            )

    return to_np(wav), time.perf_counter() - t0


# =========================
# WEBSOCKET ENDPOINT
# =========================

@app.websocket("/tts")
async def tts(ws: WebSocket):
    await ws.accept()
    logging.info("Client connected")

    req_start = time.perf_counter()
    total_audio_samples = 0
    total_model_sec = 0.0
    ttfa_ms = None

    try:
        payload = await ws.receive_json()
        text = (payload.get("text") or "").strip()
        clone_voice = bool(payload.get("clone_voice", False))
        ref_audio = payload.get("ref_audio_path")

        if not text:
            await ws.send_json({"type": "error", "error": "text is required"})
            return

        if clone_voice:
            if not ref_audio or not os.path.exists(ref_audio):
                await ws.send_json({
                    "type": "error",
                    "error": f"ref_audio not found: {ref_audio}"
                })
                return
        else:
            ref_audio = None


        # =========================
        # BASE MODE (FAST SINGLE)
        # =========================

        if not clone_voice:
            wav, model_sec = await generate_async(text, None, False)

            total_model_sec += model_sec
            total_audio_samples += len(wav)

            ttfa_ms = (time.perf_counter() - req_start) * 1000
            audio_ms = (len(wav) / SR) * 1000
            e2e_ms = (time.perf_counter() - req_start) * 1000

            await ws.send_json({
                "type": "single",
                "audio_base64": wav_to_b64_raw(wav),
                "sample_rate": SR,
                "metrics": {
                    "mode": "single_base_fast",
                    "clone_voice": False,
                    "ttfa_ms": round(ttfa_ms, 2),
                    "model_ms": round(total_model_sec * 1000, 2),
                    "e2e_ms": round(e2e_ms, 2),
                    "audio_ms": round(audio_ms, 2),
                    "rtf": round((e2e_ms / 1000) / (audio_ms / 1000), 4),
                }
            })
            return


        # =========================
        # VOICE CLONING STREAM
        # =========================

        first_text, rest_text = split_first_words(text, VOICE_LOCK_WORDS)

        wav1, model_sec1 = await generate_async(first_text, ref_audio, True)

        total_model_sec += model_sec1
        total_audio_samples += len(wav1)

        ttfa_ms = (time.perf_counter() - req_start) * 1000

        await ws.send_json({
            "type": "chunk",
            "audio_base64": wav_to_b64_raw(wav1),
            "sample_rate": SR,
        })

        sentences = split_sentences(rest_text)

        for sent in sentences:
            wav_i, model_seci = await generate_async(sent, None, True)

            total_model_sec += model_seci
            total_audio_samples += len(wav_i)

            await ws.send_json({
                "type": "chunk",
                "audio_base64": wav_to_b64_raw(wav_i),
                "sample_rate": SR,
            })

        audio_ms = (total_audio_samples / SR) * 1000
        e2e_ms = (time.perf_counter() - req_start) * 1000

        await ws.send_json({
            "type": "done",
            "metrics": {
                "mode": "voice_locked_stream",
                "clone_voice": True,
                "ttfa_ms": round(ttfa_ms, 2),
                "model_ms": round(total_model_sec * 1000, 2),
                "e2e_ms": round(e2e_ms, 2),
                "audio_ms": round(audio_ms, 2),
                "rtf": round((e2e_ms / 1000) / (audio_ms / 1000), 4),
            }
        })

    except Exception as e:
        logging.exception("Server error")
        try:
            await ws.send_json({"type": "error", "error": str(e)})
        except Exception:
            pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass
        logging.info("Client closed")
