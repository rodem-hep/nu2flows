FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

# Local and environment variables
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PIP_ROOT_USER_ACTION=ignore
ENV PIP_NO_CACHE_DIR=false
ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /nu2flows

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


# Update python pip
RUN python -m pip install --upgrade pip
RUN python --version
RUN python -m pip --version

# Install using uv
COPY requirements.txt .
RUN python -m uv install -r requirements.txt
