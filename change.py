(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav# export no_proxy=10.0.0.0/8,localhost,127.0.0.1
(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav# export NO_PROXY=10.0.0.0/8,localhost,127.0.0.1
(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav#  wget http://10.90.126.61:9000/
--2026-01-07 08:56:48--  http://10.90.126.61:9000/
Connecting to 163.116.128.80:8080... connected.
Proxy request sent, awaiting response... 400 Bad Request
2026-01-07 08:56:48 ERROR 400: Bad Request.
