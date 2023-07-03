# Load the base pytorch image
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

# Local arguments and envs
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PIP_ROOT_USER_ACTION=ignore
ARG DEBIAN_FRONTEND=noninteractive

# Update and install packages
RUN apt-get -y update && apt-get -y upgrade
RUN apt-get -y install \
    build-essential \
    wget \
    curl \
    git \
    make \
    gcc \
    graphviz \
    sudo \
    h5utils \
    vim

# Update python pip
RUN python -m pip install --upgrade pip
RUN python -m pip --version

# Install dependencies
COPY requirements.txt .
RUN python -m pip install --upgrade -r requirements.txt

# Add path to cuda for pykeops
RUN export CUDA_PATH=/opt/conda/
