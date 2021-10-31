FROM ubuntu:focal

WORKDIR /
RUN apt-get update -y --fix-missing
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata
RUN apt install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get install -y \
    software-properties-common libjpeg-dev libpng-dev libtiff-dev libfreetype6-dev \
    liblcms-dev libwebp-dev tcl libopenjp2-7-dev libimagequant-dev \
    libraqm-dev libxcb-randr0-dev libxcb-xtest0-dev libxcb-xinerama0-dev \
    libxcb-shape0-dev libxcb-xkb-dev git wget zlib1g-dev python3.6 python3-pip
COPY requirements.txt requirements.txt
RUN python3.6 -m pip install -r requirements.txt

WORKDIR /
RUN mkdir covidaid
WORKDIR /covidaid
CMD [ "./docker_entrypt.sh" ]
