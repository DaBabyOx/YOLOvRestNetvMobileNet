FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    && ln -s /usr/bin/python3 /usr/local/bin/python \
    && rm -rf /var/lib/apt/lists/*

COPY pipeline/requirements.txt /workspace/pipeline/requirements.txt

RUN pip3 install --no-cache-dir -r /workspace/pipeline/requirements.txt

CMD ["bash"]
