# XPURemoting

This is a virtualization framework for CUDA API.  The developer can easily add customization to the execution CUDA API, e.g., execute it as an RPC at a remote machine. 

## Minimal demo

Consits of 4 parts:

- `server`
- `client`
- `network`
- `codegen`

## Getting started

- How to add customization to a CUDA API:  [implement_new_apis.md](docs/implement_new_apis.md) 



## Requirements

- Environment setup: please run a docker container mounting the `xpuremoting` directory and enter it. (On `meepo3` or `meepo4`, you can use image `xpu_remoting:latest`). An example command is:

```shell
export container_name=xxx
docker run -dit  --shm-size 8G  --name $container_name  --gpus all  --privileged  --network host  -v path/to/xpuremoting:/workspace  xpu_remoting:latest
```

- Note that we need Rust **nightly** toolchain to build the project, which is not installed in the docker image.

```shell
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- --default-toolchain nightly
. "$HOME/.cargo/env"
# Optional: install cargo-expand
# See `codegen/README.md` for usage and alternatives
cargo +nightly install cargo-expand
```

- Version checklist
  - Rust: `nightly`
  - CMake: at least `3.22.1` (required to run some of the tests)
  - Clang: at least `6.0.0-1ubuntu2`

## Build

```shell
cd /path/to/xpuremoting && cargo build
```

You should specify `default` feature in both `client/server`'s Cargo.toml to decide which communication methods will be compiled. For example, if you want to use RDMA communication method, you should add `"rdma"` into the `default` feature in `client/Cargo.toml` and `server/Cargo.toml`. 

## Config

You can use `config.toml` file to config communication type, buffer size, RDMA server listener socket and so on.

Since `cargo` will use cwd as running root folder, you should use absolute path for config file. The default path will be `/workspace/config.toml`. If you want a specific path, you can use the environment variable `NETWORK_CONFIG` to customize it. For example: `NETWORK_CONFIG=/workspace/config.toml cargo run server`.

### Emulator

The network emulator is a component used to simulate system performance under different network conditions. When enabled, it calculates network latency and make the message receiver busy-wait, thus simulating the network overhead caused by RTT and bandwidth. Real-world tests have shown that the emulator's performance deviates from the actual conditions by no more than 5%. You can configure the `rtt` and `bandwidth` settings in `config.toml`.

This feature is only available when using shared memory. To enable the emulator, you can simply add `emulator` to the default features in `client/Cargo.toml` and `server/Cargo.toml`.

## Test

### Unit test

```shell
cargo test
```

### Integration test

Launch two terminals, one for server and the other for client.

- server side:

```shell
cargo run [--features rdma] server
```

You should use `features` to decide what communication methods will be compiled. SHM is always compiled.

- client side:

```shell
cd tests/cuda_api
mkdir -p build && cd build
cmake .. && make
cd ..
find -type f -executable -name "test_*" -exec ./startclient_debug.sh {} \;
```

P.S. Can use `RUST_LOG` environment to control the log level (default=debug).


### Application test

Please refer to [application-guild](./tests/apps/README.md) to run applicatons



## Appendix

### Build the docker image

Please refer to the [link](https://x8csr71rzs.feishu.cn/docx/DdXFdGSYOo8cktxgj8hcYh12nHf), and use the Dockerfile in the root directory.