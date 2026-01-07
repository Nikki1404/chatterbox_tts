aws ssm start-session \
  --region ap-south-1 \
  --target i-05a31ee067e1506cb \
  --document-name AWS-StartPortForwardingSession \
  --parameters '{"portNumber":["22"],"localPortNumber":["2222"]}'
