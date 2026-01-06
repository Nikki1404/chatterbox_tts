Enter text (end with empty line, or 'exit'):
Hello! This is a test of the text-to-speech system.

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
