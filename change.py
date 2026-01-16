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

Sampling:   0%|                                                                                                        | 0/1000 [00:00<?, ?it/s]We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class (https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)
Sampling:   2%|█▊                                                                                             | 19/1000 [00:02<02:06,  7.77it/s]
Chatterbox ready. Sample rate: 24000
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8002 (Press CTRL+C to quit)
INFO:     172.17.0.1:33966 - "WebSocket /tts" [accepted]
Client connected
INFO:     connection open
Extracting speaker embedding: voices/shashank_audio.wav
Server error: 'ChatterboxTTS' object has no attribute 'extract_speaker_embedding'
Client closed
INFO:     connection closed


Locked Mode: VOICE CLONING
• Short text → single-shot
• Long text → sentence streaming + realtime audio

Enter text (end with empty line, or 'exit'):
"Today is January 15th, 2026. The temperature is 24.5 degrees Celsius, and the time is 3:45 PM. Testing, one, two, three."

Traceback (most recent call last):
  File "C:\Users\re_nikitav\Desktop\cx-speech-voice-cloning\client\client.py", line 172, in <module>
    asyncio.run(main())
    ~~~~~~~~~~~^^^^^^^^
  File "C:\Program Files\Python313\Lib\asyncio\runners.py", line 195, in run
    return runner.run(main)
           ~~~~~~~~~~^^^^^^
  File "C:\Program Files\Python313\Lib\asyncio\runners.py", line 118, in run
    return self._loop.run_until_complete(task)
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^
  File "C:\Program Files\Python313\Lib\asyncio\base_events.py", line 725, in run_until_complete
    return future.result()
           ~~~~~~~~~~~~~^^
  File "C:\Users\re_nikitav\Desktop\cx-speech-voice-cloning\client\client.py", line 123, in main
    msg = await ws.recv()
          ^^^^^^^^^^^^^^^
  File "C:\Users\re_nikitav\Desktop\cx-speech-voice-cloning\client\client_env\Lib\site-packages\websockets\asyncio\connection.py", line 322, in recv
    raise self.protocol.close_exc from self.recv_exc
websockets.exceptions.ConnectionClosedOK: received 1000 (OK); then sent 1000 (OK)
(client_env) PS C:\Users\re_nikitav\Desktop\cx-speech-voice-cloning\client>
