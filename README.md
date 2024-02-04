# XPURemoting

## Minimal demo

Consits of 3 parts: `server`, `client` and `device_buffer`.

### Build

```shell
cd /path/to/xpuremoting && cargo build
```

### Test

Launch two terminals, one for server and the other for client.

- server side:

```shell
cargo run server
```

- client side:

```shell
cd tests/cuda_api
mkdir -p build && cd build
cmake .. && make
cd ..
./startclient.sh ./build/remoting_test
```
