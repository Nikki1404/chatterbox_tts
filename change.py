(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav# ss -tulpn | grep ssh
tcp   LISTEN 0      128              0.0.0.0:22         0.0.0.0:*    users:(("sshd",pid=2234707,fd=3))                                                      
tcp   LISTEN 0      128                 [::]:22            [::]:*    users:(("sshd",pid=2234707,fd=4))                  
