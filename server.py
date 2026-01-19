import os
import time
import asyncio
import base64
import io
import re
import logging
from typing import Optional, List

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, WebSocket
from chatterbox.tts import ChatterboxTTS

# --------------------------------------------------
# Config
# --------------------------------------------------
logging.basicConfig(level=logging.INFO)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(4)

# Latency tuning knobs
CHAR_SHORT = 200              # truly short → single shot
VOICE_LOCK_WORDS = 12         # minimal text to lock speaker
MODEL_WARMUP_TEXT = "Warmup."

app = FastAPI()

# --------------------------------------------------
# Model init
# --------------------------------------------------
print(f"Loading Chatterbox on device: {DEVICE}")
model = ChatterboxTTS.from_pretrained(device=DEVICE)
SR = model.sr

# Global warmup
_ = model.generate(MODEL_WARMUP_TEXT)
print("Chatterbox ready. Sample rate:", SR)

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def wav_to_b64(wav: np.ndarray, sr: int) -> str:
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


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


async def generate_async(text: str, ref_audio: Optional[str]):
    t0 = time.perf_counter()
    wav = await asyncio.to_thread(model.generate, text, audio_prompt_path=ref_audio)
    return to_np(wav), time.perf_counter() - t0


# --------------------------------------------------
# WebSocket
# --------------------------------------------------
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

        # --------------------------------------------------
        # SHORT TEXT (safe single-shot)
        # --------------------------------------------------
        if len(text) <= CHAR_SHORT:
            wav, model_sec = await generate_async(text, ref_audio)
            total_model_sec += model_sec
            total_audio_samples += len(wav)

            ttfa_ms = (time.perf_counter() - req_start) * 1000
            audio_ms = (len(wav) / SR) * 1000
            e2e_ms = (time.perf_counter() - req_start) * 1000

            await ws.send_json({
                "type": "single",
                "audio_base64": wav_to_b64(wav, SR),
                "metrics": {
                    "mode": "single",
                    "clone_voice": clone_voice,
                    "ttfa_ms": round(ttfa_ms, 2),
                    "model_ms": round(total_model_sec * 1000, 2),
                    "e2e_ms": round(e2e_ms, 2),
                    "audio_ms": round(audio_ms, 2),
                    "rtf": round((e2e_ms / 1000) / (audio_ms / 1000), 4),
                }
            })
            return

        # --------------------------------------------------
        # LONG TEXT — CORRECT LOW-LATENCY STRATEGY
        # --------------------------------------------------
        # Phase 1: VOICE LOCK (once)
        first_text, rest_text = split_first_words(text, VOICE_LOCK_WORDS)

        wav1, model_sec1 = await generate_async(first_text, ref_audio)
        total_model_sec += model_sec1
        total_audio_samples += len(wav1)

        ttfa_ms = (time.perf_counter() - req_start) * 1000

        await ws.send_json({
            "type": "chunk",
            "chunk_index": 1,
            "text": first_text,
            "audio_base64": wav_to_b64(wav1, SR),
        })

        await asyncio.sleep(0)

        # Phase 2: CONTINUATION (NO ref_audio)
        sentences = split_sentences(rest_text)
        chunk_idx = 2

        for sent in sentences:
            wav_i, model_seci = await generate_async(sent, None)
            total_model_sec += model_seci
            total_audio_samples += len(wav_i)

            await ws.send_json({
                "type": "chunk",
                "chunk_index": chunk_idx,
                "text": sent,
                "audio_base64": wav_to_b64(wav_i, SR),
            })
            chunk_idx += 1
            await asyncio.sleep(0)

        # Done
        audio_ms = (total_audio_samples / SR) * 1000
        e2e_ms = (time.perf_counter() - req_start) * 1000

        await ws.send_json({
            "type": "done",
            "metrics": {
                "mode": "voice_locked_stream",
                "clone_voice": clone_voice,
                "chunks": chunk_idx - 1,
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
