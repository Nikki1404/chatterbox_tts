(client_env) PS C:\Users\re_nikitav\Desktop\cx-speech-voice-cloning\client> python .\client.py

  Chatterbox TTS Client (Realtime Playback)
Reference voice is selected ONCE per session

Select reference voice (applies to entire session):
0 → No reference (BASE TTS)
1 → mono_44100_127389__acclivity__thetimehascome.wav
2 → mono_44100_382326__scott-simpson__crossing-the-bar.wav
3 → Enter custom reference audio path
Your choice: 2

Locked Mode: VOICE CLONING
• Short text → single-shot
• Long text → sentence streaming + realtime audio

Enter text (end with empty line, or 'exit'):
hi, Hello there

Traceback (most recent call last):
  File "C:\Users\re_nikitav\Desktop\cx-speech-voice-cloning\client\client.py", line 167, in <module>
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
  File "C:\Users\re_nikitav\Desktop\cx-speech-voice-cloning\client\client.py", line 103, in main
    async with websockets.connect(
               ~~~~~~~~~~~~~~~~~~^
        SERVER,
        ^^^^^^^
    ...<3 lines>...
        proxy=None,  # IMPORTANT for corp networks
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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
(client_env) PS C:\Users\re_nikitav\Desktop\cx-speech-voice-cloning\client>





