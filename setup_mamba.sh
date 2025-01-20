#!/bin/bash
# This script uses CUDA 12.1. You can swap with CUDA 11.8.
mamba create --name plexus_simulate \
    python=3.10 \
    -y
mamba activate plexus_extract

pip install -e . -v