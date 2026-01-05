import os
import time
import asyncio
import base64
import io
import re
import numpy as np
import soundfile as sf
import torch

from fastapi import FastAPI, WebSocket
from chatterbox.tts import ChatterboxTTS


# -----------------------------
# Device auto-detection
# -----------------------------
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

torch.set_num_threads(4)

SHORT_TEXT_CHAR_THRESHOLD = 200  # below this → single-shot

app = FastAPI()

print(f"Loading Chatterbox on device: {DEVICE}")
model = ChatterboxTTS.from_pretrained(device=DEVICE)
SR = model.sr

# Warmup (important for GPU timing stability)
_ = model.generate("Warmup.")
print("Chatterbox ready. SR =", SR)
