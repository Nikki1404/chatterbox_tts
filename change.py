(utils_env) re_nikitav@EC03-E01-AICOE1:~/bu-digital-cx-asr-whiperx/utils/benchmark$ python3 whisperx_benchmark.py
Benchmarking 30 WAV files (limit=30)
[DONE] 5683-32865-0011.wav
[DONE] 5683-32865-0004.wav
[DONE] 5683-32865-0016.wav
[DONE] 5683-32865-0010.wav
[DONE] 5683-32865-0006.wav
[DONE] 5683-32865-0013.wav
[DONE] 5683-32865-0002.wav
[DONE] 5683-32865-0007.wav
[DONE] 5683-32865-0014.wav
[DONE] 5683-32865-0008.wav
[DONE] 5683-32865-0009.wav
[DONE] 5683-32865-0005.wav
[DONE] 5683-32865-0000.wav
[DONE] 5683-32865-0015.wav
[DONE] 5683-32865-0003.wav
[DONE] 5683-32865-0017.wav
[DONE] 5683-32865-0001.wav
[DONE] 5683-32879-0005.wav
[DONE] 5683-32865-0012.wav
[DONE] 5683-32879-0025.wav
[DONE] 5683-32879-0009.wav
[DONE] 5683-32879-0021.wav
[DONE] 5683-32879-0015.wav
[DONE] 5683-32879-0022.wav
[DONE] 5683-32879-0004.wav
[DONE] 5683-32879-0007.wav
[DONE] 5683-32879-0003.wav
[DONE] 5683-32879-0018.wav
[DONE] 5683-32879-0008.wav
[DONE] 5683-32879-0013.wav
Traceback (most recent call last):
  File "/home/CORP/re_nikitav/bu-digital-cx-asr-whiperx/utils/benchmark/whisperx_benchmark.py", line 173, in <module>
    main()
  File "/home/CORP/re_nikitav/bu-digital-cx-asr-whiperx/utils/benchmark/whisperx_benchmark.py", line 156, in main
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
PermissionError: [Errno 13] Permission denied: 'whisperx_benchmark_results.csv'
