import asyncio
import websockets
import json
import base64
import soundfile as sf
import sounddevice as sd
import os
import numpy as np
import uuid
from datetime import datetime


SERVER = "ws://127.0.0.1:8001/tts"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def unique_wav_path(out_dir: str) -> str:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    uid = uuid.uuid4().hex[:6]
    return os.path.join(out_dir, f"tts_{ts}_{uid}.wav")


# =========================
# RAW FLOAT32 DECODER
# =========================

def decode_raw_audio(audio_b64: str, sr: int):
    audio_bytes = base64.b64decode(audio_b64)

    # Reconstruct float32 numpy array
    wav = np.frombuffer(audio_bytes, dtype=np.float32)

    return wav, sr


def normalize_path_for_server(path: str) -> str:
    return path.replace("\\", "/")


async def main():
    print("\n  Chatterbox TTS Client (Optimized Raw Audio Mode)")
    print("Reference voice is selected ONCE per session\n")

    print("Select reference voice (applies to entire session):")
    print("0 → No reference (BASE TTS)")
    print("1 → mono_44100_127389__acclivity__thetimehascome.wav")
    print("2 → mono_44100_382326__scott-simpson__crossing-the-bar.wav")
    print("3 → shashank_audio.wav")
    print("4 → Enter custom reference audio path")

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
        ref_audio = "voices/shashank_audio.wav"

    elif choice == "4":
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
    print("• Base TTS → single-shot ultra-fast")
    print("• Voice cloning → streaming mode\n")

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
        current_sr = None

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
                    print("\n Error:", data["error"])
                    break

                # =========================
                # SINGLE MODE (BASE FAST)
                # =========================
                if data["type"] == "single":
                    sr = data["sample_rate"]
                    wav, sr = decode_raw_audio(
                        data["audio_base64"],
                        sr
                    )

                    sd.play(wav, sr)
                    sd.wait()

                    out = unique_wav_path(OUT_DIR)
                    sf.write(out, wav, sr)

                    print("\n Saved:", out)
                    print(" Metrics:", data["metrics"])
                    break

                # =========================
                # STREAM CHUNK
                # =========================
                if data["type"] == "chunk":
                    sr = data["sample_rate"]
                    wav, sr = decode_raw_audio(
                        data["audio_base64"],
                        sr
                    )

                    if current_sr is None:
                        current_sr = sr

                    if audio_stream is None:
                        audio_stream = sd.OutputStream(
                            samplerate=sr,
                            channels=1,
                            dtype="float32",
                        )
                        audio_stream.start()

                    audio_stream.write(wav)
                    audio_chunks.append(wav)

                # =========================
                # DONE STREAM
                # =========================
                if data["type"] == "done":
                    if audio_stream:
                        audio_stream.stop()
                        audio_stream.close()

                    final_wav = np.concatenate(audio_chunks)

                    out = unique_wav_path(OUT_DIR)
                    sf.write(out, final_wav, current_sr)

                    print("\n Saved:", out)
                    print(" Metrics:", data["metrics"])
                    break

        print("\n--- Ready for next input ---\n")


asyncio.run(main())
