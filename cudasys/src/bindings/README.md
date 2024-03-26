# Rust Bindings for C Language Libraries

This directory contains the Rust bindings for the C language libraries. The bindings are generated using the `bindgen` tool.

## Usage

In `build.rs` we have function `bind_gen` which allows to generate bindings for the C libraries before building the Rust code. Look at the `build.rs` file to see the example of how to `bind_gen` cuda_runtime library.
