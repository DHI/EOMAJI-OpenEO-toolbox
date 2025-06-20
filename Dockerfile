FROM ubuntu:24.04

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and required packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    build-essential \
    software-properties-common \
    llvm \
    && add-apt-repository ppa:ubuntugis/ubuntugis-unstable \
    && apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app

COPY . .

RUN pip install gdal==$(gdal-config --version) --break-system-packages
RUN pip install . --break-system-packages