# XPURemoting

## Minimal demo

Consits of 4 parts:

- `server`
- `client`
- `network`
- `codegen`

### Build

We need Rust *nightly* toolchain to build the project.

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
