FROM ubuntu:focal

WORKDIR /
RUN apt-get update -y --fix-missing
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata
RUN apt-get install -y \
    software-properties-common libjpeg-dev libpng-dev libtiff-dev libfreetype6-dev \
    liblcms-dev libwebp-dev tcl libopenjp2-7-dev libimagequant-dev \
    libraqm-dev libxcb-randr0-dev libxcb-xtest0-dev libxcb-xinerama0-dev \
    libxcb-shape0-dev libxcb-xkb-dev git wget zlib1g-dev
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get install -y python3.6 python3-pip
RUN python3.6 -m pip install numpy && \
    python3.6 -m pip install torch==0.3.1 && \
    python3.6 -m pip install torchvision==0.2.0 && \
    python3.6 -m pip install pillow==6.1 && \
    python3.6 -m pip install pandas && \
    python3.6 -m pip install seaborn && \
    python3.6 -m pip install matplotlib && \
    python3.6 -m pip install tqdm && \
    python3.6 -m pip install scikit-learn

WORKDIR /
RUN mkdir covidaid
WORKDIR /covidaid
CMD [ "./docker_entrypt.sh" ]
