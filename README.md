# Chatterbox TTS – Local & EC2 Benchmarking Service

This project runs **ResembleAI Chatterbox TTS** as a **WebSocket-based service** for benchmarking
latency, TTFA (time-to-first-audio), and RTF with:

- Base TTS (no voice cloning)
- Voice cloning (reference audio)
- Sentence-based streaming (low TTFA)
- Short text → single-shot
- Long text → sentence chunking
- Local & EC2 (Dockerized) execution

The system is designed for **fair latency benchmarking** rather than real-time streaming playback.

---

##  Architecture Overview

Client (CLI)  
→ WebSocket (/tts)  
→ FastAPI + Uvicorn  
→ Sentence-based TTS generation  
→ ChatterboxTTS (CPU / GPU)

- Model loads **once at startup**
- Each sentence is generated independently to reduce TTFA
- Audio chunks are streamed to keep the connection alive
- Final WAV is stitched client-side

---

##  Project Structure

```
chatterboxTTS/
├── server.py          # WebSocket TTS server
├── client.py          # CLI client + benchmarking
├── requirements.txt
├── Dockerfile
├── voices/            # Reference voice WAVs (optional)
└── outputs/           # Generated audio files
```

---

##  Local Run (Without Docker)

### 1. Create virtual environment
```bash
python3 -m venv chatterbox_env
source chatterbox_env/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Start server
```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 600
```

### 4. Run client
```bash
python client.py
```

---

##  Docker Run 

### 1. Build Docker image
```bash
docker build -t chatterbox-tts .
```

### 2. Run container
```bash
docker run -it \
  -p 8000:8000 \
  -v $(pwd)/voices:/app/voices \
  -v $(pwd)/outputs:/app/outputs \
  chatterbox-tts
```

Server will be available at:
```
ws://<EC2_PUBLIC_IP>:8000/tts
```


##  Benchmarking Modes

When running `client.py`:

### Base TTS (no cloning)
```
Enable voice cloning? (y/n): n
```

### Voice cloning
```
Enable voice cloning? (y/n): y
ref_audio_path: voices/sample.wav
```

Metrics printed per request:
- ttfa_sec
- latency_sec_total
- audio_sec_total
- rtf
- ram_mb

Generated WAVs are saved with **unique filenames** in `outputs/`.


