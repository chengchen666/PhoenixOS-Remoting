# XPURemoting

## Minimal demo

Consits of 4 parts:

- `server`
- `client`
- `network`
- `codegen`

## Requirements

- Environment setup: please run a docker container mounting the `xpuremoting` directory and enter it. (On `meepo3` or `meepo4`, you can use image `xpu_remoting:latest`). An example command is:

```shell
export container_name=xxx
docker run -dit  --shm-size 8G  --name $container_name  --gpus all  --privileged  --network host  -v path/to/xpuremoting:/workspace  xpu_remoting:latest
```

- Note that we need Rust **nightly** toolchain to build the project, which is not installed in the docker image.

- Version checklist
  - Rust: `cargo 1.78.0-nightly`
  - CMake: `3.22.1`
  - Clang: `6.0.0-1ubuntu2`

## Build

First we should build the ELF parsing static library from source:

```shell
cd /path/to/xpuremoting/client/src/elf
make -j
```

Then we can build the project using cargo:

```shell
cd /path/to/xpuremoting && cargo build
```

## Test

### Unit test

```shell
cargo test
```

### Integration test

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

P.S. Can use `RUST_LOG` environment to control the log level (default=debug).

## Appendix

### Build the docker image

Please refer to the [link](https://x8csr71rzs.feishu.cn/docx/DdXFdGSYOo8cktxgj8hcYh12nHf), and use the Dockerfile in the root directory.