re_nikitav@EC03-E01-AICOE1:~$ sudo su
(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav#  aws ssm start-session \
  --target i-05a31ee067e1506cb \
  --document-name AWS-StartPortForwardingSession \
  --parameters '{"portNumber":["22"],"localPortNumber":["2222"]}'

You must specify a region. You can also configure your region by running "aws configure".
