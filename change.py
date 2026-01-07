aws ssm start-session \
  --target i-SOURCEINSTANCEID \
  --document-name AWS-StartPortForwardingSession \
  --parameters '{"portNumber":["22"],"localPortNumber":["2222"]}'
