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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))        # client/
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..")) # project root

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

VOICES_DIR = os.path.join(PROJECT_ROOT, "voices")

REFERENCE_VOICES = {
    "1": os.path.join(
        VOICES_DIR,
        "mono_44100_127389__acclivity__thetimehascome.wav",
    ),
    "2": os.path.join(
        VOICES_DIR,
        "mono_44100_127390__acclivity_the sunisrising.wav",
    ),
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


def resolve_reference_path(user_input: str) -> str:
    ref = os.path.normpath(user_input)

    if not os.path.isabs(ref):
        ref = os.path.join(PROJECT_ROOT, ref)

    return ref


def select_reference_audio_once():
    print("\nSelect reference voice (applies to entire session):")
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
            raise FileNotFoundError(f"Reference file not found:\n{ref}")
        return True, ref

    if choice == "3":
        user_path = input("Enter reference audio path: ").strip()
        ref = resolve_reference_path(user_path)

        if not os.path.exists(ref):
            raise FileNotFoundError(f"Reference file not found:\n{ref}")

        return True, ref

    raise ValueError("Invalid reference selection")

async def main():
    print("\n🎙️  Chatterbox TTS Client (Realtime Playback)")
    print("Reference voice is selected ONCE per session\n")

    try:
        clone_voice, ref_audio = select_reference_audio_once()
    except Exception as e:
        print(f"❌ {e}")
        return

    mode = "VOICE CLONING" if clone_voice else "BASE TTS"
    print(f"\nLocked Mode: {mode}")
    print("• Short text → single-shot")
    print("• Long text → sentence streaming + realtime audio\n")

    while True:
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
            proxy=None,  # avoids corporate proxy issues
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
                    print("❌ Error:", data["error"])
                    break

                # ---- Single-shot ----
                if data["type"] == "single":
                    wav, sr = decode_wav_from_b64(data["audio_base64"])

                    sd.play(wav, sr)
                    sd.wait()

                    out = unique_wav_path(OUT_DIR)
                    sf.write(out, wav, sr)

                    print("💾 Saved:", out)
                    print("📊 Metrics:", data["metrics"])
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

                    print("💾 Saved:", out)
                    print("📊 Metrics:", data["metrics"])
                    break

        print("\n--- Ready for next input ---\n")


asyncio.run(main())
