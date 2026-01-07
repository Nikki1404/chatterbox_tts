aws ssm start-session \
  --target i-05a31ee067e1506cb \
  --document-name AWS-StartPortForwardingSession \
  --parameters '{"portNumber":["22"],"localPortNumber":["2222"]}'

root@EC03-E01-AICOE1:/home/CORP/kunal259787# zip -r inspira_audio.zip inspira_audio/
  adding: inspira_audio/ (stored 0%)
  adding: inspira_audio/2025-08-05-18-21-41_679788052256_VOICE_c894aafb-12f8-4dd1-b1c7-c885427900f8.mp4
zip I/O error: No space left on device
zip error: Output file write failure (write error on zip file)
