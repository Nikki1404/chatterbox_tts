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

server.py- 
import os
import time
import asyncio
import base64
import io
import re
import hashlib
from collections import OrderedDict

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, WebSocket
from chatterbox.tts import ChatterboxTTS

# -------------------------
# CONFIG
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(4)

MAX_CHUNK_CHARS = 160
MAX_CACHED_VOICES = 16

app = FastAPI()

print(f"Loading Chatterbox on device: {DEVICE}")
model = ChatterboxTTS.from_pretrained(device=DEVICE)
SR = model.sr

# Warmup
_ = model.generate("Warmup.")
print("Chatterbox ready. Sample rate:", SR)

# -------------------------
# Speaker Embedding Cache (LRU)
# -------------------------
speaker_cache = OrderedDict()

def voice_fingerprint(path):
    stat = os.stat(path)
    raw = f"{path}|{stat.st_size}|{int(stat.st_mtime)}".encode()
    return hashlib.sha1(raw).hexdigest()[:12]

def get_speaker_embedding(path):
    vid = voice_fingerprint(path)

    if vid in speaker_cache:
        speaker_cache.move_to_end(vid)
        return vid, speaker_cache[vid]

    print("Extracting speaker embedding:", path)
    emb = model.extract_speaker_embedding(path)

    if len(speaker_cache) >= MAX_CACHED_VOICES:
        speaker_cache.popitem(last=False)

    speaker_cache[vid] = emb
    return vid, emb

# -------------------------
# Helpers
# -------------------------
def wav_to_b64(wav):
    buf = io.BytesIO()
    sf.write(buf, wav, SR, format="WAV")
    return base64.b64encode(buf.getvalue()).decode()

def chunk_text(text):
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

# -------------------------
# WebSocket
# -------------------------
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

        speaker_embedding = None
        voice_id = None

        if clone_voice:
            if not ref_audio or not os.path.exists(ref_audio):
                await ws.send_json({"type": "error", "error": f"ref_audio not found: {ref_audio}"})
                return
            voice_id, speaker_embedding = get_speaker_embedding(ref_audio)

        chunks = chunk_text(text)

        request_start = time.perf_counter()
        first_audio_time = None
        total_model_time = 0
        total_samples = 0

        for idx, chunk in enumerate(chunks, 1):
            t0 = time.perf_counter()

            wav = await asyncio.to_thread(
                model.generate,
                chunk,
                speaker_embedding=speaker_embedding
            )

            total_model_time += time.perf_counter() - t0

            if isinstance(wav, torch.Tensor):
                wav = wav.detach().cpu().numpy()
            wav = np.asarray(wav).squeeze()

            total_samples += len(wav)

            if first_audio_time is None:
                first_audio_time = time.perf_counter()

            await ws.send_json({
                "type": "chunk",
                "chunk_index": idx,
                "text": chunk,
                "audio_base64": wav_to_b64(wav),
            })

            await asyncio.sleep(0)

        total_latency = time.perf_counter() - request_start
        audio_sec = total_samples / SR

        await ws.send_json({
            "type": "done",
            "metrics": {
                "voice_id": voice_id,
                "clone_voice": clone_voice,
                "chunks": len(chunks),
                "ttfa_ms": round((first_audio_time - request_start) * 1000, 2),
                "model_ms": round(total_model_time * 1000, 2),
                "e2e_ms": round(total_latency * 1000, 2),
                "audio_ms": round(audio_sec * 1000, 2),
                "rtf": round(total_latency / audio_sec, 3),
                "cached_voices": len(speaker_cache),
            },
        })

    except Exception as e:
        print("Server error:", e)

    finally:
        await ws.close()
        print("Client closed")
