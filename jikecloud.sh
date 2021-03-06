#!/usr/bin/env bash
cp /data/inputs.zip inputs.zip
unzip inputs.zip -d inputs > /dev/null
rm inputs.zip
git clone https://github.com/aboluo11/lightai.git lightai_source
git clone https://github.com/aboluo11/salt.git salt_source
cp salt_source/main_git.ipynb main.ipynb
cp salt_source/update.sh update.sh
ln -s salt_source/salt salt
ln -s lightai_source/lightai lightai
pip install --no-cache-dir tensorboardX dill torch_nightly -f https://download.pytorch.org/whl/nightly/cu90/torch_nightly.html
mkdir models
