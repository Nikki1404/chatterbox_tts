9.635 Reading package lists...
10.77 Building dependency tree...
11.04 Reading state information...
11.10 E: Unable to locate package python3.12-distutils
11.10 E: Couldn't find any package by glob 'python3.12-distutils'
11.10 E: Couldn't find any package by regex 'python3.12-distutils'
------
Dockerfile:13
--------------------
  12 |
  13 | >>> RUN add-apt-repository ppa:deadsnakes/ppa && \
  14 | >>>     apt-get update && \
  15 | >>>     apt-get install -y \
  16 | >>>     python3.12 \
  17 | >>>     python3.12-dev \
  18 | >>>     python3.12-venv \
  19 | >>>     python3.12-distutils && \
  20 | >>>     rm -rf /var/lib/apt/lists/*
  21 |
--------------------
ERROR: failed to build: failed to solve: process "/bin/sh -c add-apt-repository ppa:deadsnakes/ppa &&     apt-get update &&     apt-get install -y     python3.12     python3.12-dev     python3.12-venv     python3.12-distutils &&     rm -rf /var/lib/apt/lists/*" did not complete successfully: exit code: 100
