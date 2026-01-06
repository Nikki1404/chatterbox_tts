Traceback (most recent call last):
  File "/home/CORP/re_nikitav/cx-speech-voice-cloning/client/client.py", line 102, in <module>
    asyncio.run(main())
  File "/usr/lib/python3.10/asyncio/runners.py", line 44, in run
    return loop.run_until_complete(main)
  File "/usr/lib/python3.10/asyncio/base_events.py", line 649, in run_until_complete
    return future.result()
  File "/home/CORP/re_nikitav/cx-speech-voice-cloning/client/client.py", line 84, in main
    sf.write(out, wav, sr)
  File "/home/CORP/re_nikitav/cx-speech-voice-cloning/client/client_env/lib/python3.10/site-packages/soundfile.py", line 363, in write
    with SoundFile(file, 'w', samplerate, channels,
  File "/home/CORP/re_nikitav/cx-speech-voice-cloning/client/client_env/lib/python3.10/site-packages/soundfile.py", line 690, in __init__
    self._file = self._open(file, mode_int, closefd)
  File "/home/CORP/re_nikitav/cx-speech-voice-cloning/client/client_env/lib/python3.10/site-packages/soundfile.py", line 1265, in _open
    raise LibsndfileError(err, prefix="Error opening {0!r}: ".format(self.name))
soundfile.LibsndfileError: Error opening 'outputs/tts_2026-01-06_05-54-27_9d8fac.wav': System error.
