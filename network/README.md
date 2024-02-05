# Network common

This crate aims for defining common utilities for networking for remoting, including: 
- A CommChannel for client to communication with the GPU server. 
- Specific implementations, e.g., TCP/IP, RDMA, shared memory.


# Tests

For unit test, just use: 
```
cargo test
```

For stress test for channel, use the following: 
```
cargo test --test channel_stress
```