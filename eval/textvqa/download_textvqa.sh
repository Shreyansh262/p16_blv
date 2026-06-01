#!/bin/bash
set -e
cd ~/p16_blv/eval/textvqa
echo "[1/2] Downloading TextVQA annotations..."
wget -q --show-progress https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json
echo "[2/2] Downloading TextVQA images (~1.7 GB)..."
wget -q --show-progress https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip
echo "Unzipping images..."
unzip -q train_val_images.zip
echo "Done."
