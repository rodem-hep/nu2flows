FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update && \
    apt-get -y upgrade && \
    apt-get -y install \
        build-essential \
        wget \
        curl \
        git \
        make \
        hdf5-tools \
        gcc \
        graphviz \
        sudo \
        texlive \
        texlive-latex-extra \
        texlive-fonts-recommended \
        dvipng

COPY requirements.txt .

RUN conda update conda
RUN conda update conda-build
RUN conda install pip

RUN python -m pip install --upgrade pip
RUN python -m pip install --upgrade -r requirements.txt

RUN export CUDA_PATH=/opt/conda/
