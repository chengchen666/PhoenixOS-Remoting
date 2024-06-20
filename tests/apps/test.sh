LD_LIBRARY_PATH=../../target/release:$LD_LIBRARY_PATH \
LD_PRELOAD=../../target/release/libclient.so:$LD_PRELOAD \
python3 test.py
# strace -e trace=openat python3 test.py 2>&1 | grep "\.so"

