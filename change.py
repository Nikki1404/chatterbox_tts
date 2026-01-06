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


def unique_wav_path(out_dir: str) -> str:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    uid = uuid.uuid4().hex[:6]
    return os.path.join(out_dir, f"tts_{ts}_{uid}.wav")


def decode_wav_from_b64(audio_b64: str):
    audio_bytes = base64.b64decode(audio_b64)
    buf = io.BytesIO(audio_bytes)
    wav, sr = sf.read(buf, dtype="float32")
    return wav, sr


def normalize_path_for_server(path: str) -> str:
    """
    IMPORTANT RULE:
    - Convert ONLY backslashes to forward slashes
    - NEVER touch existing forward slashes
    """
    return path.replace("\\", "/")


async def main():
    print("\n🎙️  Chatterbox TTS Client (Realtime Playback)")
    print("Reference voice is selected ONCE per session\n")

    print("Select reference voice (applies to entire session):")
    print("0 → No reference (BASE TTS)")
    print("1 → mono_44100_127389__acclivity__thetimehascome.wav")
    print("2 → mono_44100_382326__scott-simpson__crossing-the-bar.wav")
    print("3 → Enter custom reference audio path")

    choice = input("Your choice: ").strip()

    clone_voice = False
    ref_audio = None

    if choice == "0":
        clone_voice = False

    elif choice == "1":
        clone_voice = True
        ref_audio = "voices/mono_44100_127389__acclivity__thetimehascome.wav"

    elif choice == "2":
        clone_voice = True
        ref_audio = "voices/mono_44100_382326__scott-simpson__crossing-the-bar.wav"

    elif choice == "3":
        clone_voice = True
        user_path = input("Enter reference audio path: ").strip()
        if not user_path:
            print(" No path provided. Exiting.")
            return
        ref_audio = normalize_path_for_server(user_path)

    else:
        print(" Invalid choice. Exiting.")
        return

    print("\nLocked Mode:", "VOICE CLONING" if clone_voice else "BASE TTS")
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
            proxy=None,  # IMPORTANT for corp networks
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
                    print("\n Error:", data["error"])
                    break

                if data["type"] == "single":
                    wav, sr = decode_wav_from_b64(data["audio_base64"])
                    sd.play(wav, sr)
                    sd.wait()

                    out = unique_wav_path(OUT_DIR)
                    sf.write(out, wav, sr)

                    print("\n Saved:", out)
                    print(" Metrics:", data["metrics"])
                    break

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

                if data["type"] == "done":
                    if audio_stream:
                        audio_stream.stop()
                        audio_stream.close()

                    final_wav = np.concatenate(audio_chunks)
                    out = unique_wav_path(OUT_DIR)
                    sf.write(out, final_wav, sr)

                    print("\n💾 Saved:", out)
                    print("📊 Metrics:", data["metrics"])
                    break

        print("\n--- Ready for next input ---\n")


asyncio.run(main())
