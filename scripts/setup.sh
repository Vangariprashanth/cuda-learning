#!/bin/bash

set -e

echo "Updating system..."
apt update

echo "Installing core tools..."
apt install -y \
    build-essential \
    cmake \
    git \
    tmux \
    htop \
    nvtop \
    nsight-systems-cli

echo "Installing PCL..."
apt install -y \
    libpcl-dev \
    pcl-tools \
    libeigen3-dev \
    libvtk9-dev

echo "Done."