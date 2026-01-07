curl -X POST "http://127.0.0.1:8002/asr/upload_file" \
  -H "debug: yes" \
  -H "diarization: true" \
  -F "file=@sample.mp4"
