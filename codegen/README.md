# Codegen for (most) remoting functions

The crate is a procedural macro lib for generating:

- the remoting functions in both client and server side.
- the trait for transferring the user-defined types between the client and server.

```bash
├── Cargo.toml
├── README.md
├── src
│   ├── lib.rs
│   └── utils.rs
└── tests
    ├── execution.rs
    ├── hijack.rs
    ├── serialization.rs
    └── transportable.rs
```

## Modules

Most macros are explained by the comment documentation in the source code. The tests are also good examples for the usage.

- `transportable_derive`: the macro for deriving the `Transportable` trait for user-defined types.
- `gen_hijack`: the macro for generating the hijack functions for client intercepting application calls.
- `gen_execution`: the macro for generating the execution functions for server dispatching application calls.
- `gen_unimplement`: the macro for generating the unimplemented functions for client side. If the client calls the unimplemented functions, the server will throw `unimplemented!` error.

## Tests

Minor tips: to check the expanded macros, we can use the following:

```bash
cargo expand --test serialization
```
