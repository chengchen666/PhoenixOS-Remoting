curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source ~/.bashrc
rustup default nightly
apt install clang libclang-dev -y
sed -i "1s/.*/__version__ = '1.13.0'/" /opt/conda/lib/python3.8/site-packages/torch/version.py
