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

Today is January 15th, 2026. The temperature is 24.5 degrees Celsius, and the time is 3:45 PM. Testing, one, two, three.


 Saved: C:\Users\re_nikitav\Desktop\cx-speech-voice-cloning\client\outputs\tts_2026-01-16_13-15-20_224174.wav
 Metrics: {'clone_voice': True, 'chunks': 3, 'ttfa_ms': 28741.65, 'model_ms': 99381.15, 'e2e_ms': 99433.69, 'audio_ms': 12980.0, 'rtf': 7.661, 'cached_reference_voices': 1}

 Saved: C:\Users\re_nikitav\Desktop\cx-speech-voice-cloning\client\outputs\tts_2026-01-16_13-59-06_8fc4dc.wav
 Metrics: {'mode': 'single', 'clone_voice': True, 'ttfa_ms': 82698.93, 'e2e_ms': 82698.93, 'audio_ms': 13140.0, 'rtf': 6.294}
