import asyncio
import json
import time
import wave
import io
from typing import Tuple

import boto3
import websockets
import jiwer
import pandas as pd
from tqdm import tqdm
from whisper_normalizer.english import EnglishTextNormalizer

# SERVER ENDPOINTS
URL_SILERO = "ws://127.0.0.1:8082/ws"
URL_CUSTOM_VAD = "ws://127.0.0.1:8002/asr/realtime-custom-vad"

BUCKET = "cx-speech"
PREFIX = "asr-realtime/benchmarking-data-3/"

TARGET_SR = 16000
CHUNK_MS = 80
CHUNK_FRAMES = int(TARGET_SR * CHUNK_MS / 1000)

s3 = boto3.client("s3")
whisper_norm = EnglishTextNormalizer()


# TEXT NORMALIZATION
def normalize(txt):
    return whisper_norm(txt or "")


transform = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(word_delimiter=" "),
])


def calculate_wer(reference_str: str, hypothesis_str: str) -> float:
    return jiwer.wer(
        reference=reference_str,
        hypothesis=hypothesis_str,
        reference_transform=transform,
        hypothesis_transform=transform
    )


# S3 HELPERS
def list_s3_subfolders():
    paginator = s3.get_paginator("list_objects_v2")
    result = paginator.paginate(
        Bucket=BUCKET,
        Prefix=PREFIX,
        Delimiter="/"
    )

    folders = []
    for page in result:
        for prefix in page.get("CommonPrefixes", []):
            folders.append(prefix["Prefix"])
    return folders


def get_s3_object_bytes(key: str) -> bytes:
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    return obj["Body"].read()


def iter_wav_chunks_from_bytes(wav_bytes: bytes):
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        while True:
            data = wf.readframes(CHUNK_FRAMES)
            if not data:
                break
            yield data


def silence_bytes(sec: float) -> bytes:
    return b"\x00\x00" * int(TARGET_SR * sec)


# SILERO SERVER
async def transcribe_ws_silero(wav_bytes: bytes) -> Tuple[str, int]:
    async with websockets.connect(URL_SILERO, max_size=None) as ws:

        finals = []
        final_received = asyncio.Event()

        first_audio_sent = None
        ttfb_ms = None

        async def receiver():
            nonlocal ttfb_ms
            async for msg in ws:
                obj = json.loads(msg)

                # measure TTFB
                if obj.get("type") in ["partial", "final"] and ttfb_ms is None:
                    ttfb_ms = int((time.time() - first_audio_sent) * 1000)

                if obj.get("type") == "final":
                    if obj.get("text"):
                        finals.append(obj["text"].strip())
                    final_received.set()

        recv_task = asyncio.create_task(receiver())

        await ws.send(json.dumps({"type": "config", "sample_rate": TARGET_SR}))

        for i, chunk in enumerate(iter_wav_chunks_from_bytes(wav_bytes)):

            if i == 0:
                first_audio_sent = time.time()

            await ws.send(chunk)
            await asyncio.sleep(CHUNK_MS / 1000.0)

        await ws.send(silence_bytes(0.7))
        await asyncio.wait_for(final_received.wait(), timeout=60)

        await ws.close()
        recv_task.cancel()

        if ttfb_ms is None:
            ttfb_ms = -1

        return " ".join(finals), ttfb_ms


# CUSTOM VAD SERVER
async def transcribe_ws_custom_vad(wav_bytes: bytes) -> Tuple[str, int]:
    async with websockets.connect(URL_CUSTOM_VAD, max_size=None) as ws:

        finals = []
        final_received = asyncio.Event()

        first_audio_sent = None
        ttfb_ms = None

        async def receiver():
            nonlocal ttfb_ms
            async for msg in ws:
                obj = json.loads(msg)

                if obj.get("type") in ["partial", "final"] and ttfb_ms is None:
                    ttfb_ms = int((time.time() - first_audio_sent) * 1000)

                if obj.get("type") == "final":
                    if obj.get("text"):
                        finals.append(obj["text"].strip())
                    final_received.set()

        recv_task = asyncio.create_task(receiver())

        # REQUIRED INIT MESSAGE
        await ws.send(json.dumps({"backend": "nemotron"}))

        for i, chunk in enumerate(iter_wav_chunks_from_bytes(wav_bytes)):

            if i == 0:
                first_audio_sent = time.time()

            await ws.send(chunk)
            await asyncio.sleep(CHUNK_MS / 1000.0)

        await ws.send(b"")  # EOS
        await asyncio.wait_for(final_received.wait(), timeout=60)

        await ws.close()
        recv_task.cancel()

        if ttfb_ms is None:
            ttfb_ms = -1

        return " ".join(finals), ttfb_ms


# PROCESS ONE AUDIO FOLDER
async def process_folder(folder_prefix: str):
    objects = s3.list_objects_v2(Bucket=BUCKET, Prefix=folder_prefix)["Contents"]

    wav_key = None
    txt_key = None

    for obj in objects:
        if obj["Key"].endswith(".wav"):
            wav_key = obj["Key"]
        if obj["Key"].endswith(".txt"):
            txt_key = obj["Key"]

    if not wav_key or not txt_key:
        return None

    wav_bytes = get_s3_object_bytes(wav_key)
    reference_text = get_s3_object_bytes(txt_key).decode("utf-8")

    # Run both models
    silero_text, silero_latency = await transcribe_ws_silero(wav_bytes)
    custom_text, custom_latency = await transcribe_ws_custom_vad(wav_bytes)

    # Normalized
    normalized_reference = normalize(reference_text)
    normalized_silero = normalize(silero_text)
    normalized_custom = normalize(custom_text)

    return {
        "filename": folder_prefix.split("/")[-2],

        "silero_latency(ttfb_ms)": silero_latency,
        "custom_vad_latency(ttfb_ms)": custom_latency,

        "silero-wer": calculate_wer(reference_text, silero_text),
        "custom-vad-wer": calculate_wer(reference_text, custom_text),

        "reference-text": reference_text,
        "silero-text": silero_text,
        "custom-vad-text": custom_text,

        "normalized-silero-wer": calculate_wer(normalized_reference, normalized_silero),
        "normalized-custom-vad-wer": calculate_wer(normalized_reference, normalized_custom),

        "normalized-reference-text": normalized_reference,
        "normalized-sielro-text": normalized_silero,
        "normalized-custom-vad-text": normalized_custom,
    }


# WRITE XLSX
def write_to_file(results):
    df = pd.DataFrame(results)
    df = df[[
        "filename",
        "silero_latency(ttfb_ms)",
        "custom_vad_latency(ttfb_ms)",
        "silero-wer",
        "custom-vad-wer",
        "reference-text",
        "silero-text",
        "custom-vad-text",
        "normalized-silero-wer",
        "normalized-custom-vad-wer",
        "normalized-reference-text",
        "normalized-sielro-text",
        "normalized-custom-vad-text",
    ]]

    df.to_excel("benchmark_nemotron_sileroVScustom_vad.xlsx", index=False)


# MAIN
async def main():
    folders = list_s3_subfolders()
    results = []

    for folder in tqdm(folders):
        try:
            result = await process_folder(folder)
            if result:
                results.append(result)
        except Exception as e:
            print("Error in folder:", folder, e)

    write_to_file(results)


if __name__ == "__main__":
    asyncio.run(main())
