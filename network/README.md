# Network

This crate aims for defining common utilities for networking for remoting, including: 
- A CommChannel for client to communication with the GPU server. 
- Specific implementations, e.g., TCP/IP, RDMA, shared memory.

```
├── examples
│   ├── shm_client.rs
│   └── shm_server.rs
├── src
│   ├── buffer.rs
│   ├── lib.rs
│   ├── ringbufferchannel
│   │   ├── channel.rs
│   │   ├── mod.rs
│   │   ├── shm.rs
│   │   ├── test.rs
│   │   └── utils.rs
│   └── sd_impl_common.rs
└── tests
    └── channel_stress.rs
```

See the upper files. Most of the modules are quite self-defined by their names. 

# Usage 
Just include this crate to the `Cargo.toml` and use it. The detailed examples can be 
found in `examples`:
- `shm_client` and `shm_server`: the client use the ringbufferchannel backed by Linux's shared memory to send a simple message to the server. 


# Tests

For unit test, just use: 
```
cargo test
```

For stress test for channel, use the following: 
```
cargo test --test channel_stress
```