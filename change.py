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

We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class (https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)
Sampling:   2%|█▋                                                                                             | 18/1000 [00:02<02:03,  7.98it/s]
Chatterbox ready. Sample rate: 24000
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8002 (Press CTRL+C to quit)
INFO:     172.17.0.1:52056 - "WebSocket /tts" [accepted]
Client connected
INFO:     connection open
WARNING:root:Reference mel length is not equal to 2 * reference token length.

/usr/lib/python3.10/contextlib.py:103: FutureWarning: `torch.backends.cuda.sdp_kernel()` is deprecated. In the future, this context manager will be removed. Please see `torch.nn.attention.sdpa_kernel()` for the new context manager, with updated signature.
  self.gen = func(*args, **kwds)
Sampling:   8%|███████▌                                                                                       | 80/1000 [00:10<01:55,  7.93it/s]
WARNING:root:Reference mel length is not equal to 2 * reference token length.

Sampling:  19%|█████████████████▍                                                                            | 186/1000 [00:24<01:46,  7.67it/s]
WARNING:root:Reference mel length is not equal to 2 * reference token length.

Sampling:   6%|█████▋                                                                                         | 60/1000 [00:07<02:00,  7.82it/s]
Client closed
INFO:     connection closed
