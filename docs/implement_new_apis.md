# Guidelines of implementing new APIs

## Introduction

An API should first be intercepted in `client` side, transfered through `network`, and finally be actually processed in `server` side. As `network` is a generic layer, we only need to focus on `client` and `server` code for implementing new APIs.

## Which API need to be implemented?

Now the implemented APIs are listed in `client/hook.toml`. All of the rest are unimplemented yet, but they are still intercepted by auto-generated placeholder functions (check [gen_unimplement macro](../codegen/src/lib.rs#L501) and [its usage](../client/src/hijack/cudart_unimplement.rs)). If we run a application and encounter an unimplemented API, the application will throw an `unimplemented!` error, so we only need to implement those APIs prompted by the error message.

## Steps of implementation

1. Add the API to `client/hook.toml`.

This will prevent system creating unimplemented placeholder and use implemented one instead. There are two ways of implementations: default and custom.

Default implementation is a fallback used when the custom implementation is not provided, and it can handle simple cases. Custom implementation is user-defined or to handle complex cases. If default is enough in your case, just set `default = true` in the API definition. Otherwise, set `default = false` and specify the custom client and server hook functions.

You can read the comments in [gen_hijack macro](../codegen/src/lib.rs#L183) and [gen_exe macro](../codegen/src/lib.rs#L325) to check if the API can be supported by default implementation.

2. Add implementation code in both client and server side.

For default way,

- client: add `gen_hijack` macro in `client/src/hijack/*_hijack.rs`.
- server: add `gen_exe` macro in `server/src/dispatcher/*_exe.rs`, add dispatch logic in `server/src/dispatcher/mod.rs`.

For custom way,

- client: add custom hijack function in `client/src/hijack/*_hijack_custom.rs`.
- server: add custom exe function in `server/src/dispatcher/*_exe_custom.rs`, add dispatch logic in `server/src/dispatcher/mod.rs`.

You can refer to existing implementations for examples.

3. Unit test.

Add simple cuda usage test in `tests/cuda_api/` to test the new API.
