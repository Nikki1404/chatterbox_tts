FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
WORKDIR /app


RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3-pip \
    ffmpeg \
    git \
    build-essential \
    libsndfile1 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.12 /usr/bin/python && \
    python -m pip install --upgrade pip

COPY requirements.txt .

RUN pip install --no-cache-dir \
    torch==2.2.2+cu121 \
    torchaudio==2.2.2+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/audio /app/logs

ENV ENV=dev

EXPOSE 8002

CMD ["python", "main.py"]

import os
import time
import asyncio
import base64
import io
import re
import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, WebSocket
from chatterbox.tts import ChatterboxTTS

# =====================================================
# CONFIG
# =====================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(4)
MAX_CHUNK_CHARS = 160

# =====================================================
# APP + MODEL
# =====================================================
app = FastAPI()

print(f"Loading Chatterbox on device: {DEVICE}")
model = ChatterboxTTS.from_pretrained(device=DEVICE)
SR = model.sr

# Warmup
_ = model.generate("Warmup.")
print("Chatterbox ready. Sample rate:", SR)

# =====================================================
# SPEAKER PATH CACHE (SUPPORTED)
# =====================================================
# We cache *validated* reference paths to avoid repeated FS work
SPEAKER_PATH_CACHE = set()

def validate_ref_audio(path: str) -> str:
    """
    Validate and cache reference audio path.
    This is the maximum safe caching Chatterbox supports.
    """
    if path in SPEAKER_PATH_CACHE:
        return path

    if not os.path.exists(path):
        raise FileNotFoundError(f"ref_audio not found: {path}")

    SPEAKER_PATH_CACHE.add(path)
    return path

# =====================================================
# HELPERS
# =====================================================
def wav_to_b64(wav: np.ndarray) -> str:
    buf = io.BytesIO()
    sf.write(buf, wav, SR, format="WAV")
    return base64.b64encode(buf.getvalue()).decode()

def chunk_text(text: str):
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks = []
    for s in sentences:
        while len(s) > MAX_CHUNK_CHARS:
            cut = s[:MAX_CHUNK_CHARS].rsplit(" ", 1)[0]
            chunks.append(cut)
            s = s[len(cut):].strip()
        if s.strip():
            chunks.append(s.strip())
    return chunks

# =====================================================
# WEBSOCKET
# =====================================================
@app.websocket("/tts")
async def tts(ws: WebSocket):
    await ws.accept()
    print("Client connected")

    try:
        payload = await ws.receive_json()

        text = payload.get("text", "").strip()
        clone_voice = payload.get("clone_voice", False)
        ref_audio = payload.get("ref_audio_path")

        if not text:
            await ws.send_json({"type": "error", "error": "text is required"})
            return

        if clone_voice:
            try:
                ref_audio = validate_ref_audio(ref_audio)
            except Exception as e:
                await ws.send_json({"type": "error", "error": str(e)})
                return
        else:
            ref_audio = None

        chunks = chunk_text(text)

        request_start = time.perf_counter()
        first_audio_time = None
        total_samples = 0
        model_time = 0.0

        # ---------------- STREAMING ----------------
        for idx, chunk in enumerate(chunks, start=1):
            t0 = time.perf_counter()

            wav = await asyncio.to_thread(
                model.generate,
                chunk,
                audio_prompt_path=ref_audio
            )

            model_time += time.perf_counter() - t0

            if isinstance(wav, torch.Tensor):
                wav = wav.detach().cpu().numpy()
            wav = np.asarray(wav).squeeze()

            total_samples += len(wav)

            if first_audio_time is None:
                first_audio_time = time.perf_counter()

            await ws.send_json({
                "type": "chunk",
                "chunk_index": idx,
                "audio_base64": wav_to_b64(wav),
            })

            await asyncio.sleep(0)

        total_latency = time.perf_counter() - request_start
        audio_sec = total_samples / SR

        await ws.send_json({
            "type": "done",
            "metrics": {
                "clone_voice": clone_voice,
                "chunks": len(chunks),
                "ttfa_ms": round((first_audio_time - request_start) * 1000, 2),
                "model_ms": round(model_time * 1000, 2),
                "e2e_ms": round(total_latency * 1000, 2),
                "audio_ms": round(audio_sec * 1000, 2),
                "rtf": round(total_latency / audio_sec, 3),
                "cached_reference_voices": len(SPEAKER_PATH_CACHE),
            },
        })

    except Exception as e:
        print("Server error:", e)

    finally:
        await ws.close()
        print("Client closed")
