(client_env) root@EC03-E01-AICOE1:/home/CORP/re_nikitav/cx-speech-voice-cloning/client# vi client.py
(client_env) root@EC03-E01-AICOE1:/home/CORP/re_nikitav/cx-speech-voice-cloning/client# python3 client.py
Enable voice cloning? (y/n): n

Mode: BASE TTS
Short text → single-shot | Long text → sentence chunking

Enter text (end with empty line, or 'exit'):
Hello! This is a test of the text-to-speech system.
Today is Monday, January 5th, 2026, and the temperature is 24 degrees Celsius.
Dr. Smith will arrive at 10:30 a.m. for the meeting in Room B-12.
Please read the following numbers clearly: 42, 3.1416, and 1,000,000.
The quick brown fox jumps over the lazy dog.
Can you hear the difference between a question and a statement?
Thank you for listening, and have a great day!

Traceback (most recent call last):
  File "/home/CORP/re_nikitav/cx-speech-voice-cloning/client/client.py", line 102, in <module>
    asyncio.run(main())
  File "/usr/lib/python3.10/asyncio/runners.py", line 44, in run
    return loop.run_until_complete(main)
  File "/usr/lib/python3.10/asyncio/base_events.py", line 649, in run_until_complete
    return future.result()
  File "/home/CORP/re_nikitav/cx-speech-voice-cloning/client/client.py", line 60, in main
    async with websockets.connect(
  File "/home/CORP/re_nikitav/cx-speech-voice-cloning/client/client_env/lib/python3.10/site-packages/websockets/asyncio/client.py", line 587, in __aenter__
    return await self
  File "/home/CORP/re_nikitav/cx-speech-voice-cloning/client/client_env/lib/python3.10/site-packages/websockets/asyncio/client.py", line 541, in __await_impl__
    self.connection = await self.create_connection()
  File "/home/CORP/re_nikitav/cx-speech-voice-cloning/client/client_env/lib/python3.10/site-packages/websockets/asyncio/client.py", line 440, in create_connection
    transport = await connect_http_proxy(
  File "/home/CORP/re_nikitav/cx-speech-voice-cloning/client/client_env/lib/python3.10/site-packages/websockets/asyncio/client.py", line 815, in connect_http_proxy
    await protocol.response
websockets.exceptions.InvalidProxyStatus: proxy rejected connection: HTTP 400
