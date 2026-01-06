when running via local by creating tunnel via 
PS C:\Users\re_nikitav> ssh -L 8003:localhost:8003 re_nikitav@10.90.126.61
re_nikitav@10.90.126.61's password:
Welcome to Ubuntu 22.04.5 LTS (GNU/Linux 6.8.0-1040-aws x86_64)

(client_env) PS C:\Users\re_nikitav\Desktop\cx-speech-voice-cloning\client> python .\client.py

Chatterbox TTS Client
Each request can choose BASE TTS or VOICE CLONING

Use reference voice for this request? (y/n): n

Mode: BASE TTS
• Short text → single-shot
• Long text → sentence streaming

Enter text (end with empty line, or 'exit'):
Hello! This is a test of the text-to-speech system.
Traceback (most recent call last):
  File "C:\Users\re_nikitav\Desktop\cx-speech-voice-cloning\client\client.py", line 119, in <module>
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
  File "C:\Users\re_nikitav\Desktop\cx-speech-voice-cloning\client\client.py", line 68, in main
    async with websockets.connect(
               ~~~~~~~~~~~~~~~~~~^
        SERVER,
        ^^^^^^^
    ...<3 lines>...
        proxy=None,  # important in corporate networks
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ) as ws:
    ^
  File "C:\Users\re_nikitav\Desktop\cx-speech-voice-cloning\client\client_env\Lib\site-packages\websockets\asyncio\client.py", line 587, in __aenter__
    return await self
           ^^^^^^^^^^
  File "C:\Users\re_nikitav\Desktop\cx-speech-voice-cloning\client\client_env\Lib\site-packages\websockets\asyncio\client.py", line 541, in __await_impl__
    self.connection = await self.create_connection()
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\re_nikitav\Desktop\cx-speech-voice-cloning\client\client_env\Lib\site-packages\websockets\asyncio\client.py", line 467, in create_connection
    _, connection = await loop.create_connection(factory, **kwargs)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Program Files\Python313\Lib\asyncio\base_events.py", line 1166, in create_connection
    raise exceptions[0]
  File "C:\Program Files\Python313\Lib\asyncio\base_events.py", line 1141, in create_connection
    sock = await self._connect_sock(
           ^^^^^^^^^^^^^^^^^^^^^^^^^
        exceptions, addrinfo, laddr_infos)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Program Files\Python313\Lib\asyncio\base_events.py", line 1044, in _connect_sock
    await self.sock_connect(sock, address)
  File "C:\Program Files\Python313\Lib\asyncio\proactor_events.py", line 726, in sock_connect
    return await self._proactor.connect(sock, address)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Program Files\Python313\Lib\asyncio\windows_events.py", line 804, in _poll
    value = callback(transferred, key, ov)
  File "C:\Program Files\Python313\Lib\asyncio\windows_events.py", line 600, in finish_connect
    ov.getresult()
    ~~~~~~~~~~~~^^
ConnectionRefusedError: [WinError 1225] The remote computer refused the network connection
