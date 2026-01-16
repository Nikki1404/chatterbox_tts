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
import logging
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, WebSocket
from chatterbox.tts import ChatterboxTTS


# -----------------------------
# Config
# -----------------------------
logging.basicConfig(level=logging.INFO)

torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "4")))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Chunking policy (tuned for perceived latency)
CHAR_SHORT = int(os.getenv("CHAR_SHORT", "220"))          # small → single call
SENT_SINGLE_MAX = int(os.getenv("SENT_SINGLE_MAX", "1"))  # 1 sentence → single call
FIRST_BURST_WORDS = int(os.getenv("FIRST_BURST_WORDS", "12"))  # first burst for long text

# Warmup text
MODEL_WARMUP_TEXT = os.getenv("MODEL_WARMUP_TEXT", "Warmup.")
VOICE_WARMUP_TEXT = os.getenv("VOICE_WARMUP_TEXT", "Voice warmup.")


# -----------------------------
# Helpers
# -----------------------------
app = FastAPI()

print(f"Loading Chatterbox on device: {DEVICE}")
model = ChatterboxTTS.from_pretrained(device=DEVICE)
SR = model.sr

# Global warmup once
_ = model.generate(MODEL_WARMUP_TEXT)
print("Chatterbox ready. Sample rate:", SR)

# Cache: track which reference voices we've already "warmed"
WARMED_VOICES = set()


def wav_to_b64(wav: np.ndarray, sr: int) -> str:
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def safe_np(wav) -> np.ndarray:
    if isinstance(wav, torch.Tensor):
        wav = wav.detach().cpu().numpy()
    wav = np.asarray(wav).squeeze()
    # Ensure float32 for consistency
    return wav.astype(np.float32, copy=False)


def split_into_sentences(text: str) -> List[str]:
    # robust enough for demo; keeps punctuation
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sents if s]


def words(text: str) -> List[str]:
    return re.findall(r"\S+", text.strip())


def split_first_burst(text: str, n_words: int) -> Tuple[str, str]:
    w = words(text)
    if len(w) <= n_words:
        return text.strip(), ""
    first = " ".join(w[:n_words]).strip()
    rest = " ".join(w[n_words:]).strip()
    return first, rest


def should_single_call(text: str) -> bool:
    sents = split_into_sentences(text)
    if len(sents) <= SENT_SINGLE_MAX:
        return True
    if len(text) <= CHAR_SHORT:
        return True
    return False


def validate_ref_audio(clone_voice: bool, ref_audio: Optional[str]) -> Optional[str]:
    if not clone_voice:
        return None
    if not ref_audio:
        raise ValueError("clone_voice=true but ref_audio_path not provided")
    if not os.path.exists(ref_audio):
        raise ValueError(f"ref_audio not found: {ref_audio}")
    return ref_audio


def warm_voice_if_needed(ref_audio: Optional[str]) -> None:
    """
    This is NOT embedding caching (Chatterbox doesn't expose that).
    This reduces first-hit spikes for a given reference voice by doing a one-time warm generate.
    """
    if not ref_audio:
        return
    if ref_audio in WARMED_VOICES:
        return

    logging.info(f"Voice warmup (first use): {ref_audio}")
    _ = model.generate(VOICE_WARMUP_TEXT, audio_prompt_path=ref_audio)
    WARMED_VOICES.add(ref_audio)


async def generate_blocking(text: str, ref_audio: Optional[str]) -> Tuple[np.ndarray, float]:
    """
    Runs model.generate in a thread to avoid blocking the event loop.
    Returns (wav, model_time_sec).
    """
    t0 = time.perf_counter()
    wav = await asyncio.to_thread(model.generate, text, audio_prompt_path=ref_audio)
    dt = time.perf_counter() - t0
    return safe_np(wav), dt


# -----------------------------
# WebSocket endpoint
# -----------------------------
@app.websocket("/tts")
async def tts(ws: WebSocket):
    await ws.accept()
    logging.info("Client connected")

    # Metrics clocks
    req_start = time.perf_counter()
    ttfa_sent = False
    ttfa_ms = None

    total_audio_samples = 0
    total_model_sec = 0.0

    try:
        payload = await ws.receive_json()

        text = (payload.get("text") or "").strip()
        clone_voice = bool(payload.get("clone_voice", False))
        ref_audio = payload.get("ref_audio_path")

        if not text:
            await ws.send_json({"type": "error", "error": "text is required"})
            return

        # Validate voice cloning
        try:
            ref_audio = validate_ref_audio(clone_voice, ref_audio)
        except ValueError as e:
            await ws.send_json({"type": "error", "error": str(e)})
            return

        # Lazy warm per reference voice
        # NOTE: This is the closest practical thing to "speaker embedding cache".
        if ref_audio:
            await asyncio.to_thread(warm_voice_if_needed, ref_audio)

        # Decide strategy
        single_call = should_single_call(text)

        # ---- Strategy A: single call (best for 1-sentence / short) ----
        if single_call:
            wav, model_sec = await generate_blocking(text, ref_audio)
            total_model_sec += model_sec
            total_audio_samples += len(wav)

            # TTFA = when we send the first audio packet
            ttfa_ms = (time.perf_counter() - req_start) * 1000.0
            ttfa_sent = True

            audio_ms = (len(wav) / SR) * 1000.0
            e2e_ms = (time.perf_counter() - req_start) * 1000.0

            await ws.send_json({
                "type": "single",
                "audio_base64": wav_to_b64(wav, SR),
                "metrics": {
                    "mode": "single",
                    "clone_voice": clone_voice,
                    "ttfa_ms": round(ttfa_ms, 2),
                    "model_ms": round(total_model_sec * 1000.0, 2),
                    "e2e_ms": round(e2e_ms, 2),
                    "audio_ms": round(audio_ms, 2),
                    "rtf": round((e2e_ms / 1000.0) / max(audio_ms / 1000.0, 1e-6), 4),
                    "cached_reference_voices": len(WARMED_VOICES),
                }
            })
            return

        # ---- Strategy B: two-phase (fast first burst + stream remainder) ----
        # Phase 1: first burst to improve perceived latency
        first_text, rest_text = split_first_burst(text, FIRST_BURST_WORDS)

        # If split didn't actually split (edge), just run single call
        if not rest_text.strip():
            wav, model_sec = await generate_blocking(text, ref_audio)
            total_model_sec += model_sec
            total_audio_samples += len(wav)

            ttfa_ms = (time.perf_counter() - req_start) * 1000.0
            audio_ms = (len(wav) / SR) * 1000.0
            e2e_ms = (time.perf_counter() - req_start) * 1000.0

            await ws.send_json({
                "type": "single",
                "audio_base64": wav_to_b64(wav, SR),
                "metrics": {
                    "mode": "single_fallback",
                    "clone_voice": clone_voice,
                    "ttfa_ms": round(ttfa_ms, 2),
                    "model_ms": round(total_model_sec * 1000.0, 2),
                    "e2e_ms": round(e2e_ms, 2),
                    "audio_ms": round(audio_ms, 2),
                    "rtf": round((e2e_ms / 1000.0) / max(audio_ms / 1000.0, 1e-6), 4),
                    "cached_reference_voices": len(WARMED_VOICES),
                }
            })
            return

        # Phase 1 generate + send immediately
        wav1, model_sec1 = await generate_blocking(first_text, ref_audio)
        total_model_sec += model_sec1
        total_audio_samples += len(wav1)

        ttfa_ms = (time.perf_counter() - req_start) * 1000.0
        ttfa_sent = True

        await ws.send_json({
            "type": "chunk",
            "chunk_index": 1,
            "text": first_text,
            "audio_base64": wav_to_b64(wav1, SR),
        })

        await asyncio.sleep(0)

        # Phase 2: chunk remainder into sentences (sub-sentence chunking isn’t real streaming here;
        # sentence chunks are used just to keep WS responsive & allow realtime playback).
        remainder_sents = split_into_sentences(rest_text)
        chunk_idx = 2

        for sent in remainder_sents:
            wav_i, model_seci = await generate_blocking(sent, ref_audio)
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
        audio_ms_total = (total_audio_samples / SR) * 1000.0
        e2e_ms = (time.perf_counter() - req_start) * 1000.0

        await ws.send_json({
            "type": "done",
            "metrics": {
                "clone_voice": clone_voice,
                "mode": "two_phase_stream",
                "chunks": chunk_idx - 1,
                "ttfa_ms": round(ttfa_ms, 2) if ttfa_ms is not None else None,
                "model_ms": round(total_model_sec * 1000.0, 2),
                "e2e_ms": round(e2e_ms, 2),
                "audio_ms": round(audio_ms_total, 2),
                "rtf": round((e2e_ms / 1000.0) / max(audio_ms_total / 1000.0, 1e-6), 4),
                "cached_reference_voices": len(WARMED_VOICES),
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

(client_env) PS C:\Users\re_nikitav\Desktop\cx-speech-voice-cloning\client> python .\client.py

  Chatterbox TTS Client (Realtime Playback)
Reference voice is selected ONCE per session

Select reference voice (applies to entire session):
0 → No reference (BASE TTS)
1 → mono_44100_127389__acclivity__thetimehascome.wav
2 → mono_44100_382326__scott-simpson__crossing-the-bar.wav
3 → shashank_audio.wav
4 → Enter custom reference audio path
Your choice: 3

Locked Mode: VOICE CLONING
• Short text → single-shot
• Long text → sentence streaming + realtime audio

Enter text (end with empty line, or 'exit'):
Hello! This is a test of the text-to-speech system.
Today is Monday, January 5th, 2026, and the temperature is 24 degrees Celsius.
Dr. Smith will arrive at 10:30 a.m. for the meeting in Room B-12.
Please read the following numbers clearly: 42, 3.1416, and 1,000,000.
The quick brown fox jumps over the lazy dog.
Can you hear the difference between a question and a statement?
Thank you for listening, and have a great day!

Traceback (most recent call last):
  File "C:\Users\re_nikitav\Desktop\cx-speech-voice-cloning\client\client.py", line 172, in <module>
    asyncio.run(main())
    ~~~~~~~~~~~^^^^^^^^
  File "C:\Program Files\Python313\Lib\asyncio\runners.py", line 195, in run
    return runner.run(main)
           ~~~~~~~~~~^^^^^^
  File "C:\Program Files\Python313\Lib\asyncio\runners.py", line 118, in run
    return self._loop.run_until_complete(task)
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^
  File "C:\Program Files\Python313\Lib\asyncio\base_events.py", line 725, in run_until_complete
    return future.result()
           ~~~~~~~~~~~~~^^
  File "C:\Users\re_nikitav\Desktop\cx-speech-voice-cloning\client\client.py", line 108, in main
    async with websockets.connect(
               ~~~~~~~~~~~~~~~~~~^
        SERVER,
        ^^^^^^^
    ...<3 lines>...
        proxy=None,  # IMPORTANT for corp networks
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ) as ws:
    ^
  File "C:\Users\re_nikitav\Desktop\cx-speech-voice-cloning\client\client_env\Lib\site-packages\websockets\asyncio\client.py", line 587, in __aenter__
    return await self
           ^^^^^^^^^^
  File "C:\Users\re_nikitav\Desktop\cx-speech-voice-cloning\client\client_env\Lib\site-packages\websockets\asyncio\client.py", line 541, in __await_impl__
    self.connection = await self.create_connection()
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\re_nikitav\Desktop\cx-speech-voice-cloning\client\client_env\Lib\site-packages\websockets\asyncio\client.py", line 467, in create_connection
    _, connection = await loop.create_connection(factory, **kwargs)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Program Files\Python313\Lib\asyncio\base_events.py", line 1166, in create_connection
    raise exceptions[0]
  File "C:\Program Files\Python313\Lib\asyncio\base_events.py", line 1141, in create_connection
    sock = await self._connect_sock(
           ^^^^^^^^^^^^^^^^^^^^^^^^^
        exceptions, addrinfo, laddr_infos)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Program Files\Python313\Lib\asyncio\base_events.py", line 1044, in _connect_sock
    await self.sock_connect(sock, address)
  File "C:\Program Files\Python313\Lib\asyncio\proactor_events.py", line 726, in sock_connect
    return await self._proactor.connect(sock, address)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Program Files\Python313\Lib\asyncio\windows_events.py", line 804, in _poll
    value = callback(transferred, key, ov)
  File "C:\Program Files\Python313\Lib\asyncio\windows_events.py", line 600, in finish_connect
    ov.getresult()
    ~~~~~~~~~~~~^^
ConnectionRefusedError: [WinError 1225] The remote computer refused the network connection
