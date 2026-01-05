Hi Kunal,
Sharing latency comparison results after enabling GPU vs CPU, tested on the same sentence-chunked text (numbers, dates, mixed punctuation).

GPU results:

Without voice cloning:

TTFA ~ 1.0s

Total latency ~ 22.9s

RTF ~ 0.86 (faster than real time)

With voice cloning:

TTFA ~ 4.9s

Total latency ~ 34.4s

RTF ~ 0.93

CPU results (no GPU):

Without voice cloning:

TTFA ~ 11.3s

Total latency ~ 102.9s

RTF ~ 3.58

With voice cloning:

TTFA ~ 8.3s

Total latency ~ 160.7s

RTF ~ 3.92

Key difference observed:

GPU gives ~4–5× improvement in end-to-end latency compared to CPU.

TTFA drops drastically on GPU, especially without cloning.

Voice cloning adds noticeable overhead in both cases, but on GPU it remains close to real-time, whereas on CPU it becomes significantly slower.
