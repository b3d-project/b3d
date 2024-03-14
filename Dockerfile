FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

RUN apt-get update && apt-get install -y build-essential git ffmpeg libsm6 libxext6 fonts-freefont-ttf
RUN pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
RUN pip install jupyter opencv-python scikit-image filterpy
RUN cd /workspace && git clone https://github.com/facebookresearch/detectron2.git
