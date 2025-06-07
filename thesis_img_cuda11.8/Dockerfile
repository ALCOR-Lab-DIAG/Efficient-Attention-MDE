FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

RUN apt update && apt install python3 python3-pip ffmpeg libsm6 libxext6 -y

COPY requirements.txt /

RUN pip3 install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 -f https://download.pytorch.org/whl/cu118/torch_stable.html

RUN pip install -r /requirements.txt

RUN mkdir /.cache && chmod 777 /.cache 