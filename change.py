
Mode: BASE TTS
Short text → single-shot | Long text → sentence chunking

Enter text (end with empty line, or 'exit'):
hi this a tts testing

Saved: outputs/tts_2026-01-06_14-40-03_a1a699.wav
Metrics: {'mode': 'single', 'clone_voice': False, 'latency_sec_total': 2.0611, 'audio_sec_total': 2.28, 'rtf': 0.904}
Enter text (end with empty line, or 'exit'):
exit

(client_env) (base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav/cx-speech-voice-cloning/client# python3 client.py
Enable voice cloning? (y/n): y
ref_audio_path: /home/CORP/re_nikitav/cx-speech-voice-cloning/voices/shashank_audio.wav

Mode: VOICE CLONING
Short text → single-shot | Long text → sentence chunking

Enter text (end with empty line, or 'exit'):
Hello! This is a test of the text-to-speech system.
Today is Monday, January 5th, 2026, and the temperature is 24 degrees Celsius.
Dr. Smith will arrive at 10:30 a.m. for the meeting in Room B-12.
Please read the following numbers clearly: 42, 3.1416, and 1,000,000.
The quick brown fox jumps over the lazy dog.
Can you hear the difference between a question and a statement?
Thank you for listening, and have a great day!

Error: ref_audio not found: /home/CORP/re_nikitav/cx-speech-voice-cloning/voices/shashank_audio.wav
Enter text (end with empty line, or 'exit'):
^[[Aexit

Error: ref_audio not found: /home/CORP/re_nikitav/cx-speech-voice-cloning/voices/shashank_audio.wav
Enter text (end with empty line, or 'exit'):
exit

(client_env) (base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav/cx-speech-voice-cloning/client# python3 client.py
Enable voice cloning? (y/n): y
ref_audio_path: voices/shashank_audio.wav

Mode: VOICE CLONING
Short text → single-shot | Long text → sentence chunking

Enter text (end with empty line, or 'exit'):

Error: text is required
Enter text (end with empty line, or 'exit'):
hi there is a tts testing is going on

Error: ref_audio not found: voices/shashank_audio.wav
Enter text (end with empty line, or 'exit'):
