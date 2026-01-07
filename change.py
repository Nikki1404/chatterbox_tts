re_nikitav@EC03-E01-AICOE1:~$ sudo su
(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav#  aws ssm start-session \
  --target i-05a31ee067e1506cb \
  --document-name AWS-StartPortForwardingSession \
  --parameters '{"portNumber":["22"],"localPortNumber":["2222"]}'

You must specify a region. You can also configure your region by running "aws configure".
(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav# curl -s http://169.254.169.254/latest/meta-data/placement/region
(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav# TOKEN=$(curl -sX PUT "http://169.254.169.254/latest/api/token" \
  -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav# curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \
  http://169.254.169.254/latest/meta-data/placement/region
(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav# aws ssm start-session \
  --region us-east-1 \
  --target i-05a31ee067e1506cb \
  --document-name AWS-StartPortForwardingSession \
  --parameters '{"portNumber":["22"],"localPortNumber":["2222"]}'

Unable to locate credentials. You can configure credentials by running "aws login".
(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav# aws login
No AWS region has been configured. The AWS region is the geographic location of your AWS resources.

If you have used AWS before and already have resources in your account, specify which region they were created in. If you have not created resources in your account before, you can pick the region closest to you: https://docs.aws.amazon.com/global-infrastructure/latest/regions/aws-regions.html.

You are able to change the region in the CLI at any time with the command "aws configure set region NEW_REGION".
AWS Region [us-east-1]:

