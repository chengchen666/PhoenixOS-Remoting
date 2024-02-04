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
cmake .
make
./startclient.sh ./remoting_test
```

