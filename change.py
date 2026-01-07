aws ssm start-session \
  --target i-05a31ee067e1506cb \
  --document-name AWS-StartPortForwardingSession \
  --parameters '{"portNumber":["22"],"localPortNumber":["2222"]}'

root@EC03-E01-AICOE1:/home/CORP/kunal259787# df -h
Filesystem       Size  Used Avail Use% Mounted on
/dev/root        518G  518G     0 100% /
tmpfs             16G  2.0M   16G   1% /dev/shm
tmpfs            6.2G  122M  6.1G   2% /run
tmpfs            5.0M     0  5.0M   0% /run/lock
/dev/nvme0n1p15  105M  6.1M   99M   6% /boot/efi
tmpfs            3.1G  8.0K  3.1G   1% /run/user/10006
overlay          518G  518G     0 100% /var/lib/docker/overlay2/c3b99c39f615851f5677aa926a723e34248b633f30888e45f5d58304d79cdba6/merged
overlay          518G  518G     0 100% /var/lib/docker/overlay2/30fc4200dcce8abc635d12ee01d50511ce25c725705bc8fcc70f74af747be09f/merged
overlay          518G  518G     0 100% /var/lib/docker/overlay2/90ae0d17f77343e1957ab5e3bedd58870fac9cfbbbb2d5b328e8eb37af340f01/merged
tmpfs            3.1G  8.0K  3.1G   1% /run/user/94440
tmpfs            3.1G  8.0K  3.1G   1% /run/user/104684
overlay          518G  518G     0 100% /var/lib/docker/overlay2/a98cdebdafba7814525a406b331f50ec1362b7f89d7051e1e644f1f16df9fe6b/merged
