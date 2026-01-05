import time
import csv
import requests
from pathlib import Path
import jiwer
from concurrent.futures import ThreadPoolExecutor, as_completed


BASE_DIR = Path(__file__).resolve().parents[1]  # utils/

DATA_DIR = BASE_DIR / "datasets" / "data" / "wav"
RAW_LIBRISPEECH_DIR = BASE_DIR / "datasets" / "data" / "raw" / "LibriSpeech"


ASR_ENDPOINT = "http://127.0.0.1:8002/asr/upload_file"
OUTPUT_CSV = "whisperx_benchmark_results.csv"

MAX_WORKERS = 4  # CPU: 2–4 | GPU: 2–4

MAX_FILES = 30

transform = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(word_delimiter=" ")
])


def get_reference_text(wav_path: Path) -> str:
    """
    Extract reference transcription from LibriSpeech chapter-level *.trans.txt
    """

    utt_id = wav_path.stem

    # Path relative to wav root
    rel = wav_path.relative_to(DATA_DIR)
    subset = rel.parts[0]        # dev-clean / dev-other / test-clean / test-other
    speaker_id = rel.parts[1]
    chapter_id = rel.parts[2]

    trans_file = (
        RAW_LIBRISPEECH_DIR
        / subset
        / speaker_id
        / chapter_id
        / f"{speaker_id}-{chapter_id}.trans.txt"
    )

    if not trans_file.exists():
        return ""

    with open(trans_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith(utt_id):
                return line.strip().split(" ", 1)[1]

    return ""


def transcribe_via_api(wav_path: Path):
    """
    Always hits WhisperX API.
    Returns transcription and latency.
    """
    start = time.time()

    with open(wav_path, "rb") as f:
        response = requests.post(
            ASR_ENDPOINT,
            headers={
                "debug": "yes",
                "diarization": "true",
                "min-speakers": "1",
                "max-speakers": "4",
            },
            files={"file": f},
            timeout=300,
        )

    latency = time.time() - start
    response.raise_for_status()

    data = response.json()

    transcription = " ".join(
        seg["sentence"] for seg in data.get("response", [])
    )

    return transcription, latency


def process_single_wav(wav_path: Path):
    """
    1) Always call ASR API
    2) Then try reference lookup
    3) Compute WER only if reference exists
    """

    hyp_text, latency = transcribe_via_api(wav_path)

    ref_text = get_reference_text(wav_path)

    if ref_text:
        wer = jiwer.wer(
            ref_text,
            hyp_text,
            reference_transform=transform,
            hypothesis_transform=transform,
        )
        wer = round(wer, 4)
    else:
        wer = None

    subset = wav_path.relative_to(DATA_DIR).parts[0]

    return [
        subset,
        wav_path.name,
        ref_text,
        hyp_text,
        round(latency, 2),
        wer,
    ]

def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"WAV directory not found: {DATA_DIR}")

    if not RAW_LIBRISPEECH_DIR.exists():
        raise FileNotFoundError(f"LibriSpeech raw directory not found: {RAW_LIBRISPEECH_DIR}")

    # 🔴 UPDATED: limit to first 30 WAV files
    wav_files = list(DATA_DIR.rglob("*.wav"))[:MAX_FILES]
    print(f"Benchmarking {len(wav_files)} WAV files (limit={MAX_FILES})")

    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_single_wav, wav): wav
            for wav in wav_files
        }

        for future in as_completed(futures):
            wav = futures[future]
            try:
                row = future.result()
                results.append(row)
                print(f"[DONE] {wav.name}")
            except Exception as e:
                print(f"[ERROR] {wav}: {e}")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "subset",
            "file",
            "reference_text",
            "predicted_text",
            "latency_sec",
            "wer",
        ])
        writer.writerows(results)

    print("\nParallel benchmark completed")
    print(f"Results saved to: {OUTPUT_CSV}")
    print(f"Total processed files: {len(results)}")

if __name__ == "__main__":
    main()
