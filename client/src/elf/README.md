# ELF parsing library

This library is used to parse ELF code.

## Logic

- Input fatbin data buffer, which is from `__cudaRegisterFatBinary` function.
- Extract ELF header and section headers.
- Parse the kernel informations like name, parameters, etc., and store them.
- When requested, find the kernel information by name/ptr to assit register/launch the kernel.

## Implementation

The logic is written in C++ and the library is exposed as a static library (`libelfctrl.a`), interfaces are listed in `interfaces.h`.

I wrap the raw interfaces with a struct `ElfController` (`mod.rs`).
