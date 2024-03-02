# Network

This crate aims for defining common utilities for networking for remoting, including:

- A CommChannel for client to communication with the GPU server.
- Specific implementations, e.g., TCP/IP, RDMA, shared memory.

```bash
├── Cargo.toml
├── examples
│   ├── shm_client.rs
│   └── shm_server.rs
├── README.md
├── src
│   ├── buffer.rs
│   ├── lib.rs
│   ├── ringbufferchannel
│   │   ├── channel.rs
│   │   ├── mod.rs
│   │   ├── shm.rs
│   │   ├── test.rs
│   │   └── utils.rs
│   └── type_impl
│       ├── basic.rs
│       ├── cudart.rs
│       └── mod.rs
└── tests
    ├── channel_stress.rs
    └── type_transfer.rs
```

See the upper files. Most of the modules are quite self-defined by their names.

## Usage

Just include this crate to the `Cargo.toml` and use it. The detailed examples can be
found in `examples`:

- `shm_client` and `shm_server`: the client use the ringbufferchannel backed by Linux's shared memory to send a simple message to the server.

## Tests

For unit test, just use:

```bash
cargo test
```

For stress test for channel, use the following:

```bash
cargo test --test channel_stress
```
