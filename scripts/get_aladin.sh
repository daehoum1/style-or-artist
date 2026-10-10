#!/bin/bash
# Download the official ALADIN-ViT inference code and weights (Ruta et al.; github.com/DanRuta/ALADIN).
# The upstream repository states no licence, so its files are not redistributed here.
# Pinned commit 3ddd1124bf4a009e649621c680adb03d74a27953 and release v1.0.0 (aladinVIT_bamfg_64.47).
set -e
cd "$(dirname "$0")/.."
D=third_party/aladin
mkdir -p $D/weights
curl -sL -o $D/vit_networks.py https://raw.githubusercontent.com/DanRuta/ALADIN/3ddd1124bf4a009e649621c680adb03d74a27953/vit_networks.py
curl -sL -o $D/weights/aladin.zip https://github.com/DanRuta/ALADIN/releases/download/v1.0.0/aladinVIT_bamfg_64.47.zip
(cd $D/weights && unzip -o -q aladin.zip && find . -name "aladinVIT_bamfg_64.47.pt" -exec mv {} . \; 2>/dev/null; rm -f aladin.zip)
echo "expected sha256:"
echo "ff63666fb13e69710d5f3f2bed5f038d386ddd383eb522ee22688ee6abff7ad1  $D/vit_networks.py"
echo "310835f8b65349f8900ae7f70c95ee58858537939bdd2c7b86ce36a666b2a91d  $D/weights/aladinVIT_bamfg_64.47.pt"
sha256sum $D/vit_networks.py $D/weights/aladinVIT_bamfg_64.47.pt
