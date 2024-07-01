#!/bin/bash

pip install -r requirements.txt

wget -O animesfw.tgz https://pub-2fdef7a2969f43289c42ac5ae3412fd4.r2.dev/animesfw.tgz
mkdir -p tmp/model tmp/dataset
tar -C tmp/model -zxvf animesfw.tgz

wget -O mmk.tgz https://pub-2fdef7a2969f43289c42ac5ae3412fd4.r2.dev/mmk.tgz
tar -C tmp/dataset -zxvf mmk.tgz

cp types.py /opt/conda/lib/python3.8/site-packages/lightning/fabric/utilities/

sed -i "1s/.*/__version__ = '1.13.0'/" /opt/conda/lib/python3.8/site-packages/torch/version.py
