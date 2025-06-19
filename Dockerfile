# Use Ubuntu 24.04 as base
FROM ubuntu:24.04

# Set non-interactive mode for apt
ENV DEBIAN_FRONTEND=noninteractive

# Install required packages, including python3.12-venv
RUN apt-get update && apt-get install -y \
    git cmake g++ python3 python3-dev python3-pip python3.12-venv swig default-jdk \
    libxerces-c-dev libfox-1.6-dev libgdal-dev libproj-dev \
    libgl2ps-dev libeigen3-dev build-essential \
    libavformat-dev libswscale-dev libopenscenegraph-dev \
 && rm -rf /var/lib/apt/lists/*

# Install SUMO
WORKDIR /opt
RUN git clone --recursive https://github.com/eclipse-sumo/sumo.git && \
    cd sumo && \
    cmake -B build . && \
    cmake --build build -j$(nproc) && \
    cmake --install build

# Install URB benchmark
WORKDIR /opt
RUN git clone https://github.com/COeXISTENCE-PROJECT/URB.git

# Set environment variables
ENV SUMO_HOME=/opt/sumo
ENV PATH=$SUMO_HOME/bin:$PATH
# ENV PYTHONPATH=/opt/URB:$PYTHONPATH

# Default command
CMD ["/bin/bash"]
