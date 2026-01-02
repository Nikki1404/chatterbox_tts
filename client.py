import asyncio
import websockets
import json
import base64
import soundfile as sf
import io
import os
import numpy as np
import uuid
from datetime import datetime


SERVER = "ws://localhost:8000/tts"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)


def unique_wav_path(out_dir: str) -> str:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    uid = uuid.uuid4().hex[:6]
    return os.path.join(out_dir, f"tts_{ts}_{uid}.wav")


def decode_wav_from_b64(audio_b64: str):
    audio_bytes = base64.b64decode(audio_b64)
    buf = io.BytesIO(audio_bytes)
    wav, sr = sf.read(buf)
    return wav, sr


# -----------------------------
# Main
# -----------------------------
async def main():
    clone_input = input("Enable voice cloning? (y/n): ").strip().lower()
    clone_voice = clone_input == "y"

    ref_audio = None
    if clone_voice:
        ref_audio = input("ref_audio_path: ").strip()
        if not ref_audio:
            print("Error: voice cloning enabled but no ref_audio_path provided")
            return

    print("\nMode:", "VOICE CLONING" if clone_voice else "BASE TTS (NO CLONE)")
    print("• Short text → single-shot")
    print("• Long text → sentence streaming\n")

    while True:
        print("Enter text (end with empty line, or 'exit'):")
        lines = []

        while True:
            line = input()
            if line.strip() == "":
                break
            lines.append(line)

        text = " ".join(lines).strip()
        if text.lower() == "exit":
            print("Exiting client.")
            break

        audio_chunks = []

        async with websockets.connect(
            SERVER,
            max_size=200_000_000,
            ping_interval=None,
            ping_timeout=None,
        ) as ws:

            await ws.send(json.dumps({
                "text": text,
                "clone_voice": clone_voice,
                "ref_audio_path": ref_audio,
            }))

            while True:
                msg = await ws.recv()
                data = json.loads(msg)

                # ---- Error ----
                if data["type"] == "error":
                    print("Error:", data["error"])
                    break

                # ---- Short text ----
                if data["type"] == "single":
                    wav, sr = decode_wav_from_b64(data["audio_base64"])
                    out = unique_wav_path(OUT_DIR)
                    sf.write(out, wav, sr)

                    print("Saved:", out)
                    print("Metrics:", data["metrics"])
                    break

                # ---- Sentence chunk ----
                if data["type"] == "chunk":
                    wav, sr = decode_wav_from_b64(data["audio_base64"])
                    audio_chunks.append(wav)

                if data["type"] == "done":
                    final_wav = np.concatenate(audio_chunks)
                    out = unique_wav_path(OUT_DIR)
                    sf.write(out, final_wav, sr)

                    print("Saved:", out)
                    print("Metrics:", data["metrics"])
                    break


asyncio.run(main())
