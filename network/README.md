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

Just include this crate to the `Cargo.toml` and use it.

1. shared memory

The detailed examples can be found in `examples`:

- `shm_client` and `shm_server`: the client use the ringbufferchannel backed by Linux's shared memory to send a simple message to the server.

2. RDMA

A simple example can be found in `tests/rdma_comm.rs`.

New a RDMA server with:
``` rust
let server = RDMAChannelBufferManager::new_server(name, buf_size, socket).unwrap();
```
then new a client ot connect to it.
``` rust
let client = RDMAChannelBufferManager::new_client(name, buf_size, socket, port).unwrap();
```

Make sure that the client has same `name`, `buf_size`, `socket` to connect to the server created before.

- The server will listen at `socket`, so make sure the client connect to the correct socket.
- The server will create a memory region with a specific `name`, so the client should use same `name` to get server's memory region info like raddr and rkey.
- The server and the client will synchronize data in buffer, so they should have same `buf_size`.

The `port` is a ibv_dev port depending on your machine.


With different communication types, you can create `ChannelBufferManager` you want at server side and client side.
You can get `RingBuffer` with:
``` rust
let server_buf = RingBuffer::new(Box::new(server));
let client_buf = RingBuffer::new(Box::new(client));
```
Then you can send and receive data between client and server with `put_bytes` and `get_bytes`.

## Tests

For unit test, just use:

```bash
cargo test
```

For stress test for channel, use the following:

```bash
cargo test --test channel_stress
```
