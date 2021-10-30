FROM ubuntu:focal

WORKDIR /
RUN apt-get update -y --fix-missing
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata
RUN apt-get install -y \
    libjpeg-dev libpng-dev libtiff-dev libfreetype6-dev \
    liblcms-dev libwebp-dev tcl libopenjp2-7-dev libimagequant-dev \
    libraqm-dev libxcb-randr0-dev libxcb-xtest0-dev libxcb-xinerama0-dev \
    libxcb-shape0-dev libxcb-xkb-dev git wget zlib1g-dev python3-pip
RUN pip install numpy && \
    pip install torch && \
    pip install torchvision && \
    pip install pillow && \
    pip install pandas && \
    pip install seaborn && \
    pip install matplotlib && \
    pip install tqdm && \
    pip install scikit-learn

WORKDIR /
RUN mkdir covidaid
WORKDIR /covidaid
CMD [ "./docker_entrypt.sh" ]
