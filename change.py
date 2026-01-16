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

# -----------------------------
# CONFIG
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(4)

# “Short” should never be chunked
SHORT_TEXT_CHAR_THRESHOLD = 200

# For long text, we generate FIRST part quickly, then the rest
FIRST_PART_MAX_CHARS = 220  # keep small for fast TTFA

# After we have a big waveform, we stream it to client in small slices
STREAM_AUDIO_SLICE_SEC = 0.35  # smaller = more “realtime”, too small = more overhead

# Prevent concurrent generate() calls in same process (GPU thrash)
GEN_LOCK = asyncio.Lock()

# -----------------------------
# APP + MODEL
# -----------------------------
app = FastAPI()

print(f"Loading Chatterbox on device: {DEVICE}")
model = ChatterboxTTS.from_pretrained(device=DEVICE)
SR = model.sr

# Warm-up helps first request latency
_ = model.generate("Warmup.")
print("Chatterbox ready. Sample rate:", SR)

# -----------------------------
# HELPERS
# -----------------------------
def wav_to_b64(wav: np.ndarray, sr: int) -> str:
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def split_sentences(text: str):
    # simple splitter
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sents if s.strip()]

def build_two_stage_parts(text: str):
    """
    Returns (first_part, rest_part).
    first_part is small to optimize TTFA.
    rest_part may be empty.
    """
    sents = split_sentences(text)
    if not sents:
        return "", ""

    # If only 1 sentence, keep it as single-shot in main logic
    first = sents[0]
    rest = " ".join(sents[1:]).strip()

    # If first sentence itself is huge, truncate first part to FIRST_PART_MAX_CHARS
    if len(first) > FIRST_PART_MAX_CHARS:
        cut = first[:FIRST_PART_MAX_CHARS].rsplit(" ", 1)[0]
        first_part = cut.strip()
        remaining_first = first[len(cut):].strip()
        rest_part = (remaining_first + " " + rest).strip()
        return first_part, rest_part

    return first, rest

def to_np_mono(wav):
    if isinstance(wav, torch.Tensor):
        wav = wav.detach().cpu().numpy()
    wav = np.asarray(wav).squeeze()
    return wav

async def generate_audio(text: str, ref_audio: str | None):
    """
    Runs model.generate in a thread with a process-wide lock
    to avoid concurrent inference thrashing.
    """
    async with GEN_LOCK:
        with torch.inference_mode():
            wav = await asyncio.to_thread(model.generate, text, audio_prompt_path=ref_audio)
    return to_np_mono(wav)

async def stream_waveform_as_chunks(ws: WebSocket, wav: np.ndarray, start_chunk_index: int):
    """
    Streams an already-generated waveform to client as chunk messages,
    slicing by STREAM_AUDIO_SLICE_SEC.
    """
    slice_len = int(SR * STREAM_AUDIO_SLICE_SEC)
    idx = start_chunk_index

    pos = 0
    n = len(wav)
    while pos < n:
        piece = wav[pos:pos + slice_len]
        pos += slice_len

        await ws.send_json({
            "type": "chunk",
            "chunk_index": idx,
            "audio_base64": wav_to_b64(piece, SR),
        })
        idx += 1

        # yield to event loop
        await asyncio.sleep(0)

    return idx  # next chunk index

# -----------------------------
# WS ENDPOINT
# -----------------------------
@app.websocket("/tts")
async def tts(ws: WebSocket):
    await ws.accept()
    print("Client connected")

    try:
        payload = await ws.receive_json()
        text = payload.get("text", "").strip()
        clone_voice = bool(payload.get("clone_voice", False))
        ref_audio = payload.get("ref_audio_path")

        if not text:
            await ws.send_json({"type": "error", "error": "text is required"})
            return

        # Validate reference only once
        if clone_voice:
            if not ref_audio:
                await ws.send_json({"type": "error", "error": "clone_voice=true but ref_audio_path not provided"})
                return
            if not os.path.exists(ref_audio):
                await ws.send_json({"type": "error", "error": f"ref_audio not found: {ref_audio}"})
                return
        else:
            ref_audio = None

        req_start = time.perf_counter()
        first_audio_sent_ts = None

        # ==========================================================
        # SHORT TEXT => SINGLE SHOT (lowest latency)
        # ==========================================================
        if len(text) <= SHORT_TEXT_CHAR_THRESHOLD or len(split_sentences(text)) <= 1:
            wav = await generate_audio(text, ref_audio)
            first_audio_sent_ts = time.perf_counter()

            audio_sec = len(wav) / SR
            total_sec = first_audio_sent_ts - req_start

            await ws.send_json({
                "type": "single",
                "audio_base64": wav_to_b64(wav, SR),
                "metrics": {
                    "mode": "single",
                    "clone_voice": clone_voice,
                    "ttfa_ms": round((first_audio_sent_ts - req_start) * 1000, 2),
                    "e2e_ms": round(total_sec * 1000, 2),
                    "audio_ms": round(audio_sec * 1000, 2),
                    "rtf": round(total_sec / max(audio_sec, 1e-6), 3),
                }
            })
            return

        # ==========================================================
        # LONG TEXT => 2-STAGE (best TTFA + lower total time)
        # ==========================================================
        first_part, rest_part = build_two_stage_parts(text)

        # Stage 1: fast TTFA
        wav1 = await generate_audio(first_part, ref_audio)
        first_audio_sent_ts = time.perf_counter()

        # Send first audio as chunk #1 immediately (TTFA measured here)
        await ws.send_json({
            "type": "chunk",
            "chunk_index": 1,
            "audio_base64": wav_to_b64(wav1, SR),
        })

        total_samples = len(wav1)
        chunk_index = 2

        # Stage 2: generate the rest in ONE go, then stream slices
        if rest_part:
            wav2 = await generate_audio(rest_part, ref_audio)
            total_samples += len(wav2)

            # Stream already-generated waveform as small slices (keeps WS alive + realtime playback)
            chunk_index = await stream_waveform_as_chunks(ws, wav2, chunk_index)

        total_time = time.perf_counter() - req_start
        total_audio_sec = total_samples / SR

        await ws.send_json({
            "type": "done",
            "metrics": {
                "mode": "two_stage_stream",
                "clone_voice": clone_voice,
                "ttfa_ms": round((first_audio_sent_ts - req_start) * 1000, 2),
                "total_latency_ms": round(total_time * 1000, 2),
                "audio_ms": round(total_audio_sec * 1000, 2),
                "rtf": round(total_time / max(total_audio_sec, 1e-6), 3),
                "note": "Long text uses 2-stage generation (fast first audio + rest in one pass).",
            }
        })

    except Exception as e:
        print("Server error:", e)

    finally:
        await ws.close()
        print("Client closed")
