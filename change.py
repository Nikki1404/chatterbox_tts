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


if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

torch.set_num_threads(4)

SHORT_TEXT_CHAR_THRESHOLD = 200  # below this → single-shot

app = FastAPI()

print(f"Loading Chatterbox on device: {DEVICE}")
model = ChatterboxTTS.from_pretrained(device=DEVICE)
SR = model.sr

_ = model.generate("Warmup.")
print("Chatterbox ready. Sample rate:", SR)

def wav_to_b64(wav, sr):
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def split_into_sentences(text: str):
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s]


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

        # Voice cloning validation
        if clone_voice:
            if not ref_audio:
                await ws.send_json({
                    "type": "error",
                    "error": "clone_voice=true but ref_audio_path not provided"
                })
                return
            if not os.path.exists(ref_audio):
                await ws.send_json({
                    "type": "error",
                    "error": f"ref_audio not found: {ref_audio}"
                })
                return
        else:
            ref_audio = None

        t0 = time.perf_counter()

        # SHORT TEXT → single-shot
        if len(text) < SHORT_TEXT_CHAR_THRESHOLD:
            wav = await asyncio.to_thread(
                model.generate,
                text,
                audio_prompt_path=ref_audio
            )

            if isinstance(wav, torch.Tensor):
                wav = wav.detach().cpu().numpy()
            wav = np.asarray(wav).squeeze()

            latency = time.perf_counter() - t0
            audio_sec = len(wav) / SR

            await ws.send_json({
                "type": "single",
                "audio_base64": wav_to_b64(wav, SR),
                "metrics": {
                    "mode": "single",
                    "clone_voice": clone_voice,
                    "latency_sec_total": round(latency, 4),
                    "audio_sec_total": round(audio_sec, 4),
                    "rtf": round(latency / audio_sec, 4),
                },
            })
            return

        # LONG TEXT → sentence chunking
        sentences = split_into_sentences(text)

        first_chunk_time = None
        total_audio_samples = 0

        for idx, sentence in enumerate(sentences, start=1):
            wav = await asyncio.to_thread(
                model.generate,
                sentence,
                audio_prompt_path=ref_audio
            )

            if isinstance(wav, torch.Tensor):
                wav = wav.detach().cpu().numpy()
            wav = np.asarray(wav).squeeze()

            if first_chunk_time is None:
                first_chunk_time = time.perf_counter() - t0

            total_audio_samples += len(wav)

            await ws.send_json({
                "type": "chunk",
                "chunk_index": idx,
                "text": sentence,
                "audio_base64": wav_to_b64(wav, SR),
            })

            await asyncio.sleep(0)

        total_latency = time.perf_counter() - t0
        total_audio_sec = total_audio_samples / SR

        await ws.send_json({
            "type": "done",
            "metrics": {
                "mode": "sentence_chunked",
                "clone_voice": clone_voice,
                "sentences": len(sentences),
                "ttfa_sec": round(first_chunk_time, 4),
                "latency_sec_total": round(total_latency, 4),
                "audio_sec_total": round(total_audio_sec, 4),
                "rtf": round(total_latency / total_audio_sec, 4),
            },
        })

    except Exception as e:
        print("Server error:", e)

    finally:
        await ws.close()
        print("Client closed")
