import asyncio
import websockets
import json
import base64
import soundfile as sf
import sounddevice as sd
import io
import os
import numpy as np
import uuid
from datetime import datetime

SERVER = "ws://127.0.0.1:8003/tts"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# Predefined reference audios
REFERENCE_VOICES = {
    "1": os.path.join(BASE_DIR, "audio", "mono_001.wav"),
    "2": os.path.join(BASE_DIR, "audio", "mono_002.wav"),
}

def unique_wav_path(out_dir: str) -> str:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    uid = uuid.uuid4().hex[:6]
    return os.path.join(out_dir, f"tts_{ts}_{uid}.wav")


def decode_wav_from_b64(audio_b64: str):
    audio_bytes = base64.b64decode(audio_b64)
    buf = io.BytesIO(audio_bytes)
    wav, sr = sf.read(buf, dtype="float32")
    return wav, sr


def select_reference_audio():
    print("\nSelect reference voice:")
    print("0 → No reference (BASE TTS)")
    for k, v in REFERENCE_VOICES.items():
        print(f"{k} → {os.path.basename(v)}")
    print("3 → Enter custom reference audio path")

    choice = input("Your choice: ").strip()

    if choice == "0":
        return False, None

    if choice in REFERENCE_VOICES:
        ref = REFERENCE_VOICES[choice]
        if not os.path.exists(ref):
            print(f" Reference file not found: {ref}")
            return False, None
        return True, ref

    if choice == "3":
        ref = input("Enter reference audio path: ").strip()
        ref = os.path.normpath(ref)
        if not os.path.exists(ref):
            print(f" Reference file not found: {ref}")
            return False, None
        return True, ref

    print(" Invalid choice. Using BASE TTS.")
    return False, None



async def main():
    print("\n  Chatterbox TTS Client (Realtime Playback)")
    print("Menu-based reference voice selection enabled\n")

    while True:
        # ---- Reference selection ----
        clone_voice, ref_audio = select_reference_audio()

        mode = "VOICE CLONING" if clone_voice else "BASE TTS"
        print(f"\nMode: {mode}")
        print("• Short text → single-shot")
        print("• Long text → sentence streaming + realtime audio\n")

        print("Enter text (end with empty line, or 'exit'):")
        lines = []
        while True:
            line = input()
            if not line.strip():
                break
            lines.append(line)

        text = " ".join(lines).strip()
        if text.lower() == "exit":
            print("Exiting client.")
            break

        audio_chunks = []
        audio_stream = None

        async with websockets.connect(
            SERVER,
            max_size=200_000_000,
            ping_interval=None,
            ping_timeout=None,
            proxy=None,
        ) as ws:

            await ws.send(json.dumps({
                "text": text,
                "clone_voice": clone_voice,
                "ref_audio_path": ref_audio,
            }))

            while True:
                msg = await ws.recv()
                data = json.loads(msg)

                if data["type"] == "error":
                    print(" Error:", data["error"])
                    break

                # ---- Single-shot ----
                if data["type"] == "single":
                    wav, sr = decode_wav_from_b64(data["audio_base64"])

                    print(" Playing audio...")
                    sd.play(wav, sr)
                    sd.wait()

                    out = unique_wav_path(OUT_DIR)
                    sf.write(out, wav, sr)

                    print(" Saved:", out)
                    print(" Metrics:", data["metrics"])
                    break

                # ---- Streaming chunk ----
                if data["type"] == "chunk":
                    wav, sr = decode_wav_from_b64(data["audio_base64"])

                    if audio_stream is None:
                        audio_stream = sd.OutputStream(
                            samplerate=sr,
                            channels=1 if wav.ndim == 1 else wav.shape[1],
                            dtype="float32",
                        )
                        audio_stream.start()

                    audio_stream.write(wav)
                    audio_chunks.append(wav)

                # ---- Done ----
                if data["type"] == "done":
                    if audio_stream:
                        audio_stream.stop()
                        audio_stream.close()

                    final_wav = np.concatenate(audio_chunks)
                    out = unique_wav_path(OUT_DIR)
                    sf.write(out, final_wav, sr)

                    print("\n Saved:", out)
                    print(" Metrics:", data["metrics"])
                    break

        print("\n--- Request completed ---\n")


asyncio.run(main())
